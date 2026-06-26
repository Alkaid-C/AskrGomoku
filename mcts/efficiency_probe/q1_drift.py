"""
Q1 — Does the MCTS supervision signal drift slowly? (is self-play over-produced?)

See ../stage2_efficiency_plan.md §3. We fix a set S of on-distribution boards
and, for several checkpoints, run a full stage-2 MCTS on each board. The
supervision targets are the visit distribution pi(s) and root value q(s).

  - policy drift  D_pi(a,b) = E_s[ JS(pi_a(s) || pi_b(s)) ]
  - value  drift  D_q (a,b) = E_s[ (q_a(s) - q_b(s))^2 ]

Noise floor (finite sims + root Dirichlet): the SAME checkpoint searched with
several RNG seeds. With 3 seeds we get 3 pairings -> a floor mean AND spread.
The verdict uses the excess over floor:  Delta_pi(H) = D_pi(now, now-H) - F_pi.
Delta ~ 0 (inside floor) => supervision is statistically unchanged over horizon
H => fresh self-play is redundant there. The largest H still inside floor (H*)
upper-bounds STAGE2_REPLAY_BUFFER_ROUNDS.

Usage:
    python3 q1_drift.py [--now 31] [--horizons 1,4,8,12,24] [--positions 2048]
"""

import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_MCTS_DIR = os.path.dirname(_HERE)
if _MCTS_DIR not in sys.path:
    sys.path.insert(0, _MCTS_DIR)

import probe_common as pc
import torch


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Q1 supervision-drift probe")
    ap.add_argument('--data-dir', default=os.path.join(_HERE, 'data'))
    ap.add_argument('--out-dir', default=os.path.join(_HERE, 'out'))
    ap.add_argument('--now', type=int, default=31,
                    help="anchor checkpoint (S is on-distribution for this one)")
    ap.add_argument('--horizons', default='1,4,8,12,24',
                    help="comma list of H; compares now vs now-H")
    ap.add_argument('--positions', type=int, default=2048)
    ap.add_argument('--source-dump', type=int, default=None,
                    help="sample dump to draw S from (default: --now)")
    ap.add_argument('--n-seeds', type=int, default=3,
                    help="RNG seeds for the noise floor at `now`")
    ap.add_argument('--chunk', type=int, default=256, help="boards per search call")
    ap.add_argument('--seed', type=int, default=12345, help="position-sampling seed")
    ap.add_argument('--no-plot', action='store_true')
    return ap.parse_args()


# Game-phase buckets by stone count (occupied = ch0+ch1 of obs).
PHASES = [('opening', 0, 8), ('mid', 8, 20), ('end', 20, 226)]


def phase_of(occ: np.ndarray) -> dict:
    return {name: (occ >= lo) & (occ < hi) for name, lo, hi in PHASES}


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device('cuda')
    horizons = [int(h) for h in args.horizons.split(',')]
    src = args.source_dump if args.source_dump is not None else args.now

    # --- Position set S: played roots (value_weight==1) of the source dump ---
    dump = pc.load_dump(args.data_dir, src)
    played = np.where(dump['value_weight'] == 1.0)[0]
    rng = np.random.default_rng(args.seed)
    if len(played) < args.positions:
        raise SystemExit(
            f"dump {src} has only {len(played)} played roots < {args.positions}")
    sel = rng.choice(played, size=args.positions, replace=False)
    obs_S = dump['obs'][sel]                     # [N,3,15,15] uint8
    occ_S = (obs_S[:, 0] | obs_S[:, 1]).reshape(args.positions, -1).sum(-1)
    boards_S = pc.boards_from_obs(obs_S)
    print(f"S = {args.positions} played roots from dump {src} "
          f"(occ: min={occ_S.min()} median={int(np.median(occ_S))} max={occ_S.max()})")

    # --- Searches: now x {seeds}, plus each now-H once ---
    # Pack everything into a dict keyed by a label -> (dist[N,225], q[N]).
    results: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def search_ckpt(update: int, seeds: list[int], tag: str) -> None:
        path = pc.checkpoint_path(args.data_dir, update)
        model = pc.load_model(path, device)
        for k, sd in enumerate(seeds):
            np.random.seed(sd)              # Dirichlet noise is the only RNG source
            dist, q = pc.run_search(
                model, boards_S, device, chunk_size=args.chunk,
                clear_cache_first=(k == 0),  # same weights across seeds: keep cache
            )
            label = f"{tag}_s{k}" if len(seeds) > 1 else tag
            results[label] = (dist, q)
            print(f"  searched update {update} seed#{k} -> {label}")
        del model
        torch.cuda.empty_cache()

    floor_seeds = [args.seed + 1 + i for i in range(args.n_seeds)]
    print(f"Searching anchor now={args.now} with {args.n_seeds} seeds (floor)...")
    search_ckpt(args.now, floor_seeds, 'now')

    for H in horizons:
        u = args.now - H
        if u < 1:
            print(f"  skip H={H}: update {u} < 1")
            continue
        print(f"Searching now-{H} = update {u}...")
        search_ckpt(u, [args.seed + 100 + H], f'H{H}')

    # --- Metrics ---
    phase_masks = phase_of(occ_S)

    def pair_pi(a: str, b: str) -> np.ndarray:
        return pc.js_divergence(results[a][0], results[b][0])

    def pair_q(a: str, b: str) -> np.ndarray:
        return (results[a][1] - results[b][1]) ** 2

    # Noise floor: all seed pairings of `now`.
    seed_labels = [f'now_s{i}' for i in range(args.n_seeds)]
    floor_pi_pairs, floor_q_pairs = [], []
    for i in range(args.n_seeds):
        for j in range(i + 1, args.n_seeds):
            floor_pi_pairs.append(pair_pi(seed_labels[i], seed_labels[j]))
            floor_q_pairs.append(pair_q(seed_labels[i], seed_labels[j]))
    floor_pi_pool = np.concatenate(floor_pi_pairs)
    floor_q_pool = np.concatenate(floor_q_pairs)
    F_pi = float(np.mean(floor_pi_pool))
    F_q = float(np.mean(floor_q_pool))
    # per-pair means give a band on the floor estimate itself
    F_pi_band = [float(np.mean(p)) for p in floor_pi_pairs]
    F_q_band = [float(np.mean(p)) for p in floor_q_pairs]

    report = {
        'config': vars(args),
        'floor': {
            'F_pi': F_pi, 'F_pi_pair_means': F_pi_band,
            'F_q': F_q, 'F_q_pair_means': F_q_band,
            'pi_dist': pc.distribution_summary(floor_pi_pool),
            'q_dist': pc.distribution_summary(floor_q_pool),
        },
        'horizons': {},
    }

    # Drift vs floor: now_s0 (seed A) compared against each now-H.
    anchor = 'now_s0'
    H_star_pi = 0
    floor_hi = max(F_pi_band)  # "inside floor" = D within the worst floor pairing
    print("\n" + "=" * 92)
    print(f"{'H':>4} | {'D_pi':>8} {'Dlt_pi':>8} {'p90_pi':>8} | "
          f"{'D_q':>9} {'Dlt_q':>9} | inside_floor")
    print(f"{'floor':>4} | {F_pi:8.5f} {'-':>8} {pc.distribution_summary(floor_pi_pool)['p90']:8.5f} | "
          f"{F_q:9.6f} {'-':>9} |")
    print("-" * 92)
    for H in horizons:
        lbl = f'H{H}'
        if lbl not in results:
            continue
        dpi_pp = pair_pi(anchor, lbl)
        dq_pp = pair_q(anchor, lbl)
        D_pi = float(np.mean(dpi_pp))
        D_q = float(np.mean(dq_pp))
        inside = D_pi <= floor_hi
        if inside:
            H_star_pi = max(H_star_pi, H)
        per_phase = {}
        for name, m in phase_masks.items():
            if m.any():
                per_phase[name] = {
                    'n': int(m.sum()),
                    'D_pi': float(np.mean(dpi_pp[m])),
                    'D_q': float(np.mean(dq_pp[m])),
                }
        report['horizons'][H] = {
            'update': args.now - H,
            'D_pi': D_pi, 'Delta_pi': D_pi - F_pi,
            'D_q': D_q, 'Delta_q': D_q - F_q,
            'pi_dist': pc.distribution_summary(dpi_pp),
            'q_dist': pc.distribution_summary(dq_pp),
            'inside_floor': inside,
            'per_phase': per_phase,
        }
        print(f"{H:>4} | {D_pi:8.5f} {D_pi - F_pi:8.5f} "
              f"{pc.distribution_summary(dpi_pp)['p90']:8.5f} | "
              f"{D_q:9.6f} {D_q - F_q:9.6f} | {'yes' if inside else 'NO'}")
    print("=" * 92)
    report['H_star_pi'] = H_star_pi
    print(f"H*_pi (largest H whose D_pi stays within floor) = {H_star_pi}  "
          f"(buffer rounds R = {pc.hp.STAGE2_REPLAY_BUFFER_ROUNDS})")

    # --- Persist ---
    json_path = os.path.join(args.out_dir, f'q1_drift_now{args.now}.json')
    with open(json_path, 'w') as f:
        json.dump(report, f, indent=2)
    npz_path = os.path.join(args.out_dir, f'q1_raw_now{args.now}.npz')
    save_arrays: dict[str, np.ndarray] = {'occ': occ_S}
    for k, v in results.items():
        save_arrays[f'{k}_dist'] = v[0]
        save_arrays[f'{k}_q'] = v[1]
    np.savez_compressed(npz_path, **save_arrays)  # type: ignore[arg-type]
    print(f"wrote {json_path}\nwrote {npz_path}")

    if not args.no_plot:
        _plot(args, report, horizons, F_pi, F_q, F_pi_band, F_q_band)


def _plot(args, report, horizons, F_pi, F_q, F_pi_band, F_q_band) -> None:
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"(plot skipped: {e})")
        return
    Hs = [H for H in horizons if H in report['horizons']]
    D_pi = [report['horizons'][H]['D_pi'] for H in Hs]
    D_q = [report['horizons'][H]['D_q'] for H in Hs]
    fig, (axp, axq) = plt.subplots(1, 2, figsize=(12, 4.5))
    R = pc.hp.STAGE2_REPLAY_BUFFER_ROUNDS
    for ax, D, F, band, title in (
        (axp, D_pi, F_pi, F_pi_band, 'policy drift  D_pi = E[JS]'),
        (axq, D_q, F_q, F_q_band, 'value drift  D_q = E[(dq)^2]'),
    ):
        ax.plot(Hs, D, 'o-', label='D(now, now-H)')
        ax.axhspan(min(band), max(band), color='gray', alpha=0.25, label='floor band')
        ax.axhline(F, color='gray', ls='--', lw=1, label='floor mean')
        ax.axvline(R, color='red', ls=':', lw=1, label=f'R={R}')
        ax.set_xlabel('horizon H (updates)')
        ax.set_title(title)
        ax.legend(fontsize=8)
    fig.suptitle(f'Q1 supervision drift vs floor (now={args.now})')
    fig.tight_layout()
    png = os.path.join(args.out_dir, f'q1_drift_now{args.now}.png')
    fig.savefig(png, dpi=120)
    print(f"wrote {png}")


if __name__ == '__main__':
    main()
