"""
Top-1 flip study driver.

Plays ``final_policy.pt`` against itself with MCTS from three families of start
positions, running each ply's search to ``--total-sims`` while sampling the
played move (temperature 1) from the ``--action-sims`` snapshot of that same
search. For every ply it records significant Top-1 preference changes using a
full-distribution baseline that starts at the raw policy and resets after every
confirmed flip, plus board features that let a later analysis ask which
positions are prone to flipping.

Start families:
  * empty      : empty board (Black to move),                n = --n-empty
  * renju      : a random Renju opening + random triangular offset (White to
                 move), recording opening_id / offset_r / offset_c,  n = --n-renju
  * random3    : three distinct random squares, auto-colored Black/White/Black
                 (White to move), recording the three squares,       n = --n-random3

Output: one JSON object per ply, streamed to ``flip_data.jsonl``. Search config
matches stage 2 (c_puct, gamma, fpu); no Dirichlet noise; raw masked-softmax
priors. See mcts/CLAUDE.md.
"""

import argparse
import json
import random
import sys
from pathlib import Path

# Project modules (gomoku, model, mcts, mcts_ext) live in the parent mcts/ dir.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from gomoku import (
    OPENING_OFFSET_RANGE,
    RENJU_OPENING_SEQUENCES,
    GameState,
    GomokuBoard,
    Player,
    idx_to_pos,
)
from model import GomokuPolicyNet
from search_with_snapshots import mcts_search_with_snapshots

# Stage-2 search constants (mirrors mcts/main.py).
C_PUCT = 1.25
DISCOUNT_GAMMA = 63.0 / 64
FPU_MULTIPLIER = 0.95


def build_starts(category: str, n: int) -> list[tuple[GomokuBoard, dict]]:
    """Return n (board, start_meta) pairs for the given start family."""
    starts: list[tuple[GomokuBoard, dict]] = []
    for _ in range(n):
        if category == "empty":
            board = GomokuBoard(opening_id=-1)
            meta: dict = {}
        elif category == "renju":
            opening_id = random.randrange(len(RENJU_OPENING_SEQUENCES))
            offset_r = round(random.triangular(-OPENING_OFFSET_RANGE, OPENING_OFFSET_RANGE, 0))
            offset_c = round(random.triangular(-OPENING_OFFSET_RANGE, OPENING_OFFSET_RANGE, 0))
            board = GomokuBoard(opening_id=-1)
            base_r, base_c = 7 + offset_r, 7 + offset_c
            for rel_r, rel_c in RENJU_OPENING_SEQUENCES[opening_id]:
                board.Move((base_r + rel_r, base_c + rel_c))
            meta = {"opening_id": opening_id, "offset_r": offset_r, "offset_c": offset_c}
        elif category == "random3":
            squares = random.sample(range(225), 3)  # distinct -> non-overlapping
            board = GomokuBoard(opening_id=-1)
            for idx in squares:
                board.Move(idx_to_pos(idx))  # auto-colored Black, White, Black
            meta = {"squares": squares}
        else:
            raise ValueError(f"unknown category: {category}")
        starts.append((board, meta))
    return starts


def run_category(
    model: torch.nn.Module,
    category: str,
    n: int,
    total_sims: int,
    action_sims: int,
    margin: float,
    device: torch.device,
    out,
) -> int:
    """Self-play all n games of a category to termination, streaming ply rows.

    Returns the number of ply rows written.
    """
    starts = build_starts(category, n)
    boards = [b for b, _ in starts]
    metas = [m for _, m in starts]

    # Per-game buffer of ply rows, held until the game ends so steps_left can be
    # back-filled from the final game length. Rows hold only scalars (no arrays).
    pending: list[list[dict]] = [[] for _ in range(n)]
    active = list(range(n))
    rows_written = 0

    while active:
        active_boards = [boards[i] for i in active]
        results = mcts_search_with_snapshots(
            model, active_boards, total_sims, action_sims,
            C_PUCT, DISCOUNT_GAMMA, FPU_MULTIPLIER, margin, device,
        )

        still_active: list[int] = []
        for j, i in enumerate(active):
            r = results[j]
            board = boards[i]
            ply_idx = len(pending[i])
            color = "black" if board.who_to_play == Player.BLACK else "white"
            pending[i].append({
                "category": category,
                "game": i,
                "start": metas[i],
                "ply_idx": ply_idx,
                "stone_count": int(board.occupied_count),
                "color": color,
                "alpha": r.alpha,
                "beta_final": r.beta_final,
                "winner_final": r.winner_final,
                "flipped": bool(r.flipped),
                "flip_tracking_start_sim": r.flip_tracking_start_sim,
                "flip_sims": r.flip_sims,
                "flip_events": r.flip_events,
                "num_flips": len(r.flip_sims),
                "last_flip_sim": r.flip_sims[-1] if r.flip_sims else None,
                "top1_agree_action_final": r.beta_final == int(np.argmax(r.dist_action)),
                "p_raw_alpha": r.p_raw_alpha,
                "p_raw_beta": r.p_raw_beta,
                "p_search_alpha": r.p_search_alpha,
                "p_search_beta": r.p_search_beta,
                "p_raw_winner": r.p_raw_winner,
                "p_search_winner": r.p_search_winner,
                "raw_entropy": r.raw_entropy,
                "raw_value": r.raw_value,
                "mcts_value": r.mcts_value,
                "raw_mcts_kl_action": r.raw_mcts_kl_action,
                "raw_mcts_kl_final": r.raw_mcts_kl_final,
            })

            # Sample the played move from the action_sims visit distribution (T=1).
            action = int(np.random.choice(225, p=r.dist_action))
            outcome = board.Move(idx_to_pos(action))
            if outcome == GameState.CONTINUE:
                still_active.append(i)
            else:
                total_plies = len(pending[i])
                for p, row in enumerate(pending[i]):
                    row["total_plies"] = total_plies
                    row["steps_left"] = total_plies - 1 - p
                    row["outcome"] = outcome.name
                    out.write(json.dumps(row) + "\n")
                rows_written += total_plies
                pending[i] = []
        out.flush()
        active = still_active

    return rows_written


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", default="release/stage2/final_policy.pt")
    ap.add_argument("--out", default="flip_analysis/flip_data.jsonl")
    ap.add_argument("--total-sims", type=int, default=4096)
    ap.add_argument("--action-sims", type=int, default=2048)
    ap.add_argument("--margin", type=float, default=0.05)
    ap.add_argument("--n-empty", type=int, default=32)
    ap.add_argument("--n-renju", type=int, default=128)
    ap.add_argument("--n-random3", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = GomokuPolicyNet().to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    categories = [
        ("empty", args.n_empty),
        ("renju", args.n_renju),
        ("random3", args.n_random3),
    ]
    total_rows = 0
    with out_path.open("w") as out:
        for category, n in categories:
            if n <= 0:
                continue
            print(f"[{category}] playing {n} games "
                  f"(total_sims={args.total_sims}, action_sims={args.action_sims}) ...",
                  flush=True)
            rows = run_category(
                model, category, n, args.total_sims, args.action_sims,
                args.margin, device, out,
            )
            total_rows += rows
            print(f"[{category}] wrote {rows} ply rows", flush=True)

    print(f"done: {total_rows} ply rows -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
