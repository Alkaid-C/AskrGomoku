"""
Cross-pipeline strength evaluation: mini vs vanilla vs main.

Q1: 3 finals pairwise, 256 games per pair (184 Renju + 72 empty, each played as a
    color-swapped pair: 128 distinct opening rows, 2 colors each).
Q2: stubbed — sampled checkpoints + full matrix.

Each Renju opening's center offset (in ±2) is fixed once and reused across all
pairings so that opening i is the *same physical board* in every match.
"""

import argparse
import importlib.util
import os
import sys
import time
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# gomoku.py is symlinked across pipelines; pull it from main/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'main'))
from gomoku import (
    LOGIT_MASK_VALUE,
    RENJU_OPENING_SEQUENCES,
    EvalGameState,
    GameState,
    GomokuBoard,
    Player,
    encode_observation,
    idx_to_pos,
)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
PIPELINES = {
    'mini':    {'dir': 'mini',    'final_dir': 'test'},
    'vanilla': {'dir': 'vanilla', 'final_dir': 'release'},
    'main':    {'dir': 'main',    'final_dir': 'release'},
}


# ============================================================================
# Per-pipeline model loading (each pipeline's model.py defines its own
# GomokuPolicyNet; load each as a uniquely-named module).
# ============================================================================

_pipeline_modules: dict = {}


def get_pipeline_module(name: str):
    if name in _pipeline_modules:
        return _pipeline_modules[name]
    pdir = os.path.join(PROJECT_ROOT, PIPELINES[name]['dir'])
    mpath = os.path.join(pdir, 'model.py')
    spec = importlib.util.spec_from_file_location(f'_model_{name}', mpath)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _pipeline_modules[name] = mod
    return mod


def load_checkpoint(pipeline: str, ckpt_path: str, device: torch.device) -> nn.Module:
    mod = get_pipeline_module(pipeline)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = mod.GomokuPolicyNet().to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model


def final_policy_path(pipeline: str) -> str:
    p = PIPELINES[pipeline]
    return os.path.join(PROJECT_ROOT, p['dir'], p['final_dir'], 'final_policy.pt')


# ============================================================================
# Opening table (deterministic — same offsets across all pairings)
# ============================================================================

def build_opening_table(seed: int = 20260426,
                        n_renju: int = 184,
                        n_empty: int = 72,
                        offset_range: int = 2) -> List[Tuple[int, int, int]]:
    """Return list of (opening_id, offset_r, offset_c). opening_id=-1 is empty."""
    rng = np.random.default_rng(seed)
    specs: List[Tuple[int, int, int]] = []
    for i in range(n_renju):
        dr = int(rng.integers(-offset_range, offset_range + 1))
        dc = int(rng.integers(-offset_range, offset_range + 1))
        specs.append((i, dr, dc))
    for _ in range(n_empty):
        specs.append((-1, 0, 0))
    return specs


def apply_opening(board: GomokuBoard, spec: Tuple[int, int, int]) -> None:
    opening_id, dr, dc = spec
    if opening_id < 0:
        return
    base_r, base_c = 7 + dr, 7 + dc
    for rel_r, rel_c in RENJU_OPENING_SEQUENCES[opening_id]:
        board.Move((base_r + rel_r, base_c + rel_c))


# ============================================================================
# Eval play loop with deterministic openings
# ============================================================================

def _select_action_eval(model: nn.Module, obs_list, mask_list,
                        temperature: float, device: torch.device) -> List[int]:
    with torch.inference_mode():
        obs = torch.from_numpy(np.stack(obs_list)).float().to(device)
        msk = torch.from_numpy(np.stack(mask_list)).bool().to(device)
        logits_grid = model.forward_policy_only(obs)
        logits = logits_grid.squeeze(1)
        logits = logits.masked_fill(~msk, LOGIT_MASK_VALUE)
        if temperature > 0:
            logits = logits / temperature
        flat = logits.view(len(obs_list), 225)
        probs = F.softmax(flat, dim=1)
        actions = torch.multinomial(probs, num_samples=1).squeeze(1).cpu().tolist()
    return actions


def play_eval_games_with_openings(black_white_pairs: List[Tuple[nn.Module, nn.Module]],
                                  current_is_black: List[bool],
                                  opening_specs: List[Tuple[int, int, int]],
                                  temperature: float,
                                  device: torch.device) -> List[Tuple[GameState, bool]]:
    """Like play_eval_games, but each game's board is pre-seeded with a deterministic opening."""
    games = []
    for (b, w), is_black, spec in zip(black_white_pairs, current_is_black, opening_specs):
        g = EvalGameState(b, w, is_black)
        apply_opening(g.board, spec)
        # If opening already terminated the game (shouldn't for 3 stones — too few for 5-in-a-row),
        # mark done. Renju openings are 3 stones, max len of any line is 2, so impossible.
        games.append(g)

    n_games = len(games)
    model_game_map: dict = {}
    for i, game in enumerate(games):
        bid = id(game.black_model)
        if bid not in model_game_map:
            model_game_map[bid] = (game.black_model, [], [])
        model_game_map[bid][1].append(i)
        wid = id(game.white_model)
        if wid not in model_game_map:
            model_game_map[wid] = (game.white_model, [], [])
        model_game_map[wid][2].append(i)

    active = [True] * n_games
    n_active = n_games
    while n_active > 0:
        all_actions = [None] * n_games
        for model, b_idx, w_idx in model_game_map.values():
            batch = [i for i in b_idx if active[i] and games[i].board.who_to_play == Player.BLACK]
            batch.extend(i for i in w_idx if active[i] and games[i].board.who_to_play == Player.WHITE)
            if not batch:
                continue
            obs_list, mask_list = [], []
            for i in batch:
                lm, _ = games[i].board.GetLegalMoves()
                c0, c1, _ = games[i].board.GetBoardState()
                obs_list.append(encode_observation(c0, c1))
                mask_list.append(lm)
            actions = _select_action_eval(model, obs_list, mask_list, temperature, device)
            for i, a in zip(batch, actions):
                all_actions[i] = a
        for i in range(n_games):
            if not active[i]:
                continue
            row, col = idx_to_pos(all_actions[i])
            outcome = games[i].board.Move((row, col))
            if outcome != GameState.CONTINUE:
                games[i].outcome = outcome
                games[i].done = True
                active[i] = False
                n_active -= 1

    return [(g.outcome, g.current_is_black) for g in games]


# ============================================================================
# Match: model_a vs model_b, paired colors, all openings
# ============================================================================

def play_match(model_a: nn.Module, model_b: nn.Module,
               opening_specs: List[Tuple[int, int, int]],
               device: torch.device,
               temperature: float = 1.0,
               batch_size: int = 512) -> dict:
    """Return per-game outcomes for A. 2 games per opening (A=black, A=white)."""
    pairs: List[Tuple[nn.Module, nn.Module]] = []
    a_is_black: List[bool] = []
    specs_per_game: List[Tuple[int, int, int]] = []
    opening_idx_per_game: List[int] = []
    for oi, spec in enumerate(opening_specs):
        pairs.append((model_a, model_b))
        a_is_black.append(True)
        specs_per_game.append(spec)
        opening_idx_per_game.append(oi)
        pairs.append((model_b, model_a))
        a_is_black.append(False)
        specs_per_game.append(spec)
        opening_idx_per_game.append(oi)

    n_total = len(pairs)
    a_results: List[int] = []  # 1=A win, -1=B win, 0=draw
    for start in range(0, n_total, batch_size):
        end = min(start + batch_size, n_total)
        chunk_results = play_eval_games_with_openings(
            pairs[start:end], a_is_black[start:end], specs_per_game[start:end],
            temperature, device,
        )
        for outcome, a_black in chunk_results:
            if outcome == GameState.DRAW:
                a_results.append(0)
            elif (outcome == GameState.BLACK_WIN and a_black) or \
                 (outcome == GameState.WHITE_WIN and not a_black):
                a_results.append(1)
            else:
                a_results.append(-1)

    a_wins = sum(1 for r in a_results if r == 1)
    b_wins = sum(1 for r in a_results if r == -1)
    draws  = sum(1 for r in a_results if r == 0)
    return {
        'a_wins': a_wins, 'b_wins': b_wins, 'draws': draws,
        'n_games': n_total,
        'a_score': (a_wins + 0.5 * draws) / n_total,
        'a_results': np.array(a_results, dtype=np.int8),
        'a_is_black': np.array(a_is_black, dtype=bool),
        'opening_idx': np.array(opening_idx_per_game, dtype=np.int32),
    }


# ============================================================================
# Q1: 3 finals
# ============================================================================

# ============================================================================
# Q2: sampled-checkpoint full matrix
# ============================================================================

def discover_sampled_checkpoints(pipeline: str, step: int = 256,
                                 max_update: int = 65536) -> List[Tuple[int, str]]:
    """Return [(update, path)] for every multiple of `step` from `step` to `max_update`."""
    p = PIPELINES[pipeline]
    base = os.path.join(PROJECT_ROOT, p['dir'], p['final_dir'])
    out = []
    for u in range(step, max_update + 1, step):
        path = os.path.join(base, f'checkpoint_update_{u}.pt')
        if os.path.exists(path):
            out.append((u, path))
    return out


def run_q2(device: torch.device, output_path: str, temperature: float = 1.0,
           sample_step: int = 256, n_openings: int = 128) -> None:
    """Full pairwise matrix on sampled checkpoints.

    All n*n (i=black, j=white) games for each opening are dispatched as a single
    batch so each model's per-step inference call is fat (~n active games at peak).
    Diagonals (i == j) are skipped — self-play is information-free for ranking.
    """
    print('=' * 60)
    print('Q2: sampled-checkpoint full matrix')
    print('=' * 60)

    # Discover sampled checkpoints across pipelines
    entries: List[Tuple[str, int, str]] = []  # (pipeline, update, path)
    for name in ('mini', 'vanilla', 'main'):
        for u, path in discover_sampled_checkpoints(name, step=sample_step):
            entries.append((name, u, path))
    n = len(entries)
    print(f'Sampled {n} checkpoints '
          f'(mini={sum(1 for e in entries if e[0]=="mini")}, '
          f'vanilla={sum(1 for e in entries if e[0]=="vanilla")}, '
          f'main={sum(1 for e in entries if e[0]=="main")})')

    opening_specs = build_opening_table()[:n_openings]
    print(f'Using {len(opening_specs)} openings (subset of Q1 table) — '
          f'total games per matrix entry = {2 * len(opening_specs)} '
          f'(symmetric pair contribution from (i,j) and (j,i)).')

    # Load all checkpoints
    print('Loading checkpoints...')
    t_load = time.time()
    models: List[nn.Module] = []
    for k, (name, _, path) in enumerate(entries):
        models.append(load_checkpoint(name, path, device))
        if (k + 1) % 50 == 0:
            print(f'  loaded {k+1}/{n}')
    total_params = sum(sum(p.numel() for p in m.parameters()) for m in models)
    print(f'Loaded {n} models in {time.time()-t_load:.1f}s '
          f'(total params: {total_params/1e6:.1f}M, '
          f'~{total_params*4/1e9:.2f} GB fp32 weights)')
    if torch.cuda.is_available():
        print(f'  cuda mem allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB / '
              f'reserved: {torch.cuda.memory_reserved()/1e9:.2f} GB')

    # Result matrices: M[i,j] = count over games where i=black, j=white
    black_wins = np.zeros((n, n), dtype=np.int32)
    white_wins = np.zeros((n, n), dtype=np.int32)
    draws_m   = np.zeros((n, n), dtype=np.int32)

    # Pre-build the (i, j) index list for one opening (skip diagonal)
    pair_indices: List[Tuple[int, int]] = [(i, j) for i in range(n) for j in range(n) if i != j]
    n_pairs = len(pair_indices)
    print(f'Per opening: {n_pairs} games (n*(n-1) = {n}*{n-1})')

    total_games_planned = n_pairs * len(opening_specs)
    games_done = 0
    t_start = time.time()

    for op_idx, spec in enumerate(opening_specs):
        pairs: List[Tuple[nn.Module, nn.Module]] = [(models[i], models[j]) for i, j in pair_indices]
        is_black_a = [True] * n_pairs  # bookkeeping flag, unused for matrix indexing
        specs_for_call = [spec] * n_pairs

        t_op = time.time()
        results = play_eval_games_with_openings(
            pairs, is_black_a, specs_for_call, temperature, device,
        )
        for (i, j), (outcome, _) in zip(pair_indices, results):
            if outcome == GameState.DRAW:
                draws_m[i, j] += 1
            elif outcome == GameState.BLACK_WIN:
                black_wins[i, j] += 1
            elif outcome == GameState.WHITE_WIN:
                white_wins[i, j] += 1

        games_done += n_pairs
        dt_op = time.time() - t_op
        rate = n_pairs / dt_op
        elapsed = time.time() - t_start
        eta = (total_games_planned - games_done) / max(rate, 1.0)
        print(f'  opening {op_idx+1}/{len(opening_specs)} (id={spec[0]:>3d}, '
              f'offset=({spec[1]:+d},{spec[2]:+d})): '
              f'{n_pairs} games in {dt_op:.1f}s ({rate:.0f} g/s) | '
              f'elapsed {elapsed/60:.1f}m | ETA {eta/60:.1f}m')
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f'Total: {games_done} games in {(time.time()-t_start)/60:.1f}m')

    # Save
    pipelines_arr = np.array([e[0] for e in entries])
    updates_arr   = np.array([e[1] for e in entries], dtype=np.int32)

    np.savez_compressed(
        output_path,
        pipelines=pipelines_arr,
        updates=updates_arr,
        black_wins=black_wins,
        white_wins=white_wins,
        draws=draws_m,
        opening_ids=np.array([s[0] for s in opening_specs], dtype=np.int16),
        offset_r=np.array([s[1] for s in opening_specs], dtype=np.int8),
        offset_c=np.array([s[2] for s in opening_specs], dtype=np.int8),
    )
    print(f'Saved: {output_path}')


def run_q1(device: torch.device, output_path: str, temperature: float = 1.0,
           batch_size: int = 512) -> None:
    print('=' * 60)
    print('Q1: final-vs-final matchups')
    print('=' * 60)

    opening_specs = build_opening_table()
    print(f'Opening table: {len(opening_specs)} openings '
          f'({sum(1 for s in opening_specs if s[0] >= 0)} Renju + '
          f'{sum(1 for s in opening_specs if s[0] < 0)} empty)')

    # Load all three finals once
    finals: dict = {}
    for name in ('mini', 'vanilla', 'main'):
        path = final_policy_path(name)
        print(f'Loading {name}: {path}')
        finals[name] = load_checkpoint(name, path, device)
        n_params = sum(p.numel() for p in finals[name].parameters())
        print(f'  {name} params: {n_params:,}')

    pairings = [('mini', 'vanilla'), ('mini', 'main'), ('vanilla', 'main')]
    summary = {}
    saved: dict = {
        'opening_ids': np.array([s[0] for s in opening_specs], dtype=np.int16),
        'offset_r':    np.array([s[1] for s in opening_specs], dtype=np.int8),
        'offset_c':    np.array([s[2] for s in opening_specs], dtype=np.int8),
    }

    for a, b in pairings:
        print('-' * 60)
        print(f'{a} (A) vs {b} (B)')
        print('-' * 60)
        t0 = time.time()
        res = play_match(finals[a], finals[b], opening_specs, device,
                         temperature=temperature, batch_size=batch_size)
        dt = time.time() - t0
        rate = res['n_games'] / dt
        print(f'  games={res["n_games"]} | A wins={res["a_wins"]} | '
              f'B wins={res["b_wins"]} | draws={res["draws"]} | '
              f'A score={res["a_score"]:.3f} | {dt:.1f}s ({rate:.1f} g/s)')
        summary[f'{a}_vs_{b}'] = res
        prefix = f'{a}_vs_{b}'
        saved[f'{prefix}_a_results']  = res['a_results']
        saved[f'{prefix}_a_is_black'] = res['a_is_black']
        saved[f'{prefix}_opening_idx'] = res['opening_idx']

    print('=' * 60)
    print('Summary (A score = (wins + 0.5*draws) / games):')
    for (a, b) in pairings:
        r = summary[f'{a}_vs_{b}']
        n = r['n_games']
        a_blk = r['a_is_black']
        a_res = r['a_results']
        wb = int(((a_res == 1) & a_blk).sum())
        gb = int(a_blk.sum())
        ww = int(((a_res == 1) & ~a_blk).sum())
        gw = int((~a_blk).sum())
        print(f'  {a:>7s} vs {b:<7s}  score={r["a_score"]:.3f}  '
              f'(A-black: {wb}/{gb}={wb/gb:.2%},  A-white: {ww}/{gw}={ww/gw:.2%},  '
              f'draws={r["draws"]}/{n})')

    np.savez_compressed(output_path, **saved)
    print(f'Saved: {output_path}')


# ============================================================================
# Entrypoint
# ============================================================================

def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['q1', 'q2'], help='which research question')
    p.add_argument('--out', default=None, help='output npz path')
    p.add_argument('--temperature', type=float, default=1.0)
    p.add_argument('--batch-size', type=int, default=512)
    p.add_argument('--sample-step', type=int, default=256, help='Q2 only: ckpt sampling interval (updates)')
    p.add_argument('--n-openings', type=int, default=128, help='Q2 only: number of openings (each contributes 2 games per pair via i<->j swap)')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args(argv)

    device = torch.device(args.device)
    if args.mode == 'q1':
        out = args.out or os.path.join(os.path.dirname(__file__), 'results_q1.npz')
        run_q1(device, out, temperature=args.temperature, batch_size=args.batch_size)
    else:
        out = args.out or os.path.join(os.path.dirname(__file__), 'results_q2.npz')
        run_q2(device, out, temperature=args.temperature,
               sample_step=args.sample_step, n_openings=args.n_openings)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
