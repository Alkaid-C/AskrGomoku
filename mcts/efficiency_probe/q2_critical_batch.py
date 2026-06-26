"""
Q2 — Is the per-step batch larger than the critical batch? (gradient redundancy)

See ../stage2_efficiency_plan.md §4. At a fixed checkpoint with a fixed replay
buffer, we take a low-noise reference gradient and ask how small a subset still
points the same way:

    cos(g_alpha, g_full) = <g_alpha, g_full> / (||g_alpha|| ||g_full||)

Past the critical batch, growing the batch mostly cuts variance without turning
the gradient, so cos saturates near 1. The smallest alpha holding cos >~ 0.95 is
the batch (i.e. STAGE2_SAMPLE_RATIO) we could drop to.

Reference g_full (per the agreed design): ALL samples in the buffer, but with
each round i (i=0 newest) reweighted by rho^i / N_i  (rho = STAGE2_DECAY_RATIO,
N_i = round size). This makes the all-samples reference equal, in expectation, to
the recency-weighted population that real training descends: training draws a
*count* k_i = k_0 * rho^i from round i regardless of N_i, so each round's total
mass must be ~ rho^i, i.e. per-sample weight ~ rho^i / N_i (on top of the
sample's own policy/value weight).

g_alpha draws alpha * k_i samples per round (the real per-step proportions, base
per-sample weights) so E[g_alpha] = g_full exactly.

Usage:
    python3 q2_critical_batch.py [--updates 12,24,31] [--alphas 1,0.5,0.25,0.125,0.0625]
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
    ap = argparse.ArgumentParser(description="Q2 critical-batch probe")
    ap.add_argument('--data-dir', default=os.path.join(_HERE, 'data'))
    ap.add_argument('--out-dir', default=os.path.join(_HERE, 'out'))
    ap.add_argument('--updates', default='25,27,29',
                    help="checkpoint/buffer ends to probe (need full R-round history)")
    ap.add_argument('--alphas', default='1,0.5,0.25,0.125',
                    help="batch scales relative to the real per-step draw")
    ap.add_argument('--resamples', type=int, default=5, help="draws per alpha")
    ap.add_argument('--cos-threshold', type=float, default=0.95)
    ap.add_argument('--seed', type=int, default=777)
    ap.add_argument('--no-plot', action='store_true')
    return ap.parse_args()


def load_buffer(data_dir: str, u: int, R: int) -> list[dict]:
    """Buffer at update u as a newest->oldest list of round dicts (round i = u-i).

    Raises if any required dump is missing.
    """
    rounds = []
    for i in range(R):
        ui = u - i
        if ui < 0 or not os.path.exists(pc.dump_path(data_dir, ui)):
            raise SystemExit(f"missing dump for round {ui} (buffer end u={u}, R={R})")
        rounds.append(pc.load_dump(data_dir, ui))
    return rounds


def per_round_k(rounds: list[dict], sample_ratio: float, decay: float) -> list[int]:
    """Real per-step draw counts k_i, matching stage2_trainer's post-warmup rule."""
    k0 = round(sample_ratio * len(rounds[0]['value']))
    return [max(0, min(len(rd['value']), round(k0 * decay ** i)))
            for i, rd in enumerate(rounds)]


def reference_arrays(rounds: list[dict], decay: float) -> dict:
    """Concatenate all rounds; per-sample weights *= rho^i / N_i (round mass ~ rho^i)."""
    obs, dist, val, pw, vw = [], [], [], [], []
    for i, rd in enumerate(rounds):
        ni = len(rd['value'])
        factor = (decay ** i) / ni
        obs.append(rd['obs'])
        dist.append(rd['dist'])
        val.append(rd['value'])
        pw.append(rd['policy_weight'] * factor)
        vw.append(rd['value_weight'] * factor)
    return {
        'obs': np.concatenate(obs), 'dist': np.concatenate(dist),
        'value': np.concatenate(val),
        'policy_weight': np.concatenate(pw), 'value_weight': np.concatenate(vw),
    }


def draw_subset(rounds: list[dict], counts: list[int], rng: np.random.Generator) -> dict:
    """Draw `counts[i]` uniform samples from round i, base weights, concatenated."""
    obs, dist, val, pw, vw = [], [], [], [], []
    for rd, k in zip(rounds, counts):
        if k <= 0:
            continue
        idx = rng.choice(len(rd['value']), size=k, replace=False)
        obs.append(rd['obs'][idx])
        dist.append(rd['dist'][idx])
        val.append(rd['value'][idx])
        pw.append(rd['policy_weight'][idx])
        vw.append(rd['value_weight'][idx])
    return {
        'obs': np.concatenate(obs), 'dist': np.concatenate(dist),
        'value': np.concatenate(val),
        'policy_weight': np.concatenate(pw), 'value_weight': np.concatenate(vw),
    }


def grad_of(model, arr, device) -> torch.Tensor:
    return pc.compute_grad_vector(
        model, arr['obs'], arr['dist'], arr['value'],
        arr['policy_weight'], arr['value_weight'],
        device, value_loss_coeff=pc.hp.VALUE_LOSS_COEFF,
    )


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device('cuda')
    R = pc.hp.STAGE2_REPLAY_BUFFER_ROUNDS
    decay = pc.hp.STAGE2_DECAY_RATIO
    sample_ratio = pc.hp.STAGE2_SAMPLE_RATIO
    updates = [int(x) for x in args.updates.split(',')]
    alphas = [float(x) for x in args.alphas.split(',')]

    report = {'config': vars(args), 'R': R, 'decay': decay,
              'sample_ratio': sample_ratio, 'updates': {}}

    for u in updates:
        print(f"\n=== checkpoint/buffer u={u} (rounds {u-R+1}..{u}) ===")
        rounds = load_buffer(args.data_dir, u, R)
        k = per_round_k(rounds, sample_ratio, decay)
        B_step = sum(k)
        buf_total = sum(len(rd['value']) for rd in rounds)
        print(f"round sizes (new->old): {[len(rd['value']) for rd in rounds]}")
        print(f"per-step k_i: {k}  -> B_step={B_step} raw  (buffer total {buf_total})")

        model = pc.load_model(pc.checkpoint_path(args.data_dir, u), device)

        print("computing reference gradient over ALL samples (rho^i/N_i weighted)...")
        ref = reference_arrays(rounds, decay)
        g_full = grad_of(model, ref, device)
        gfull_norm = float(g_full.norm())
        print(f"  ||g_full|| = {gfull_norm:.4e}  (over {buf_total} raw samples)")

        rng = np.random.default_rng(args.seed + u)
        alpha_rows = []
        # cache squared-norms for the simple noise-scale estimate
        sqnorm_by_alpha: dict[float, list[float]] = {}
        for a in alphas:
            counts = [max(0, min(len(rd['value']), round(a * ki)))
                      for rd, ki in zip(rounds, k)]
            n_raw = sum(counts)
            coss, sqn = [], []
            for _ in range(args.resamples):
                sub = draw_subset(rounds, counts, rng)
                g = grad_of(model, sub, device)
                coss.append(pc.cosine(g, g_full))
                sqn.append(float(g.norm()) ** 2)
            sqnorm_by_alpha[a] = sqn
            row = {
                'alpha': a, 'n_raw': n_raw, 'n_aug': n_raw * 8,
                'cos_mean': float(np.mean(coss)), 'cos_std': float(np.std(coss)),
                'cos_min': float(np.min(coss)),
            }
            alpha_rows.append(row)
            print(f"  alpha={a:<7} n_raw={n_raw:<6} cos={row['cos_mean']:.4f} "
                  f"+/- {row['cos_std']:.4f} (min {row['cos_min']:.4f})")

        # smallest alpha holding cos_mean >= threshold
        ok = [r for r in alpha_rows if r['cos_mean'] >= args.cos_threshold]
        crit = min(ok, key=lambda r: r['alpha']) if ok else None

        # Simple noise scale B_simple = tr(Sigma)/||G||^2 via the two-batch
        # estimator on raw draw sizes (largest vs smallest probed alpha):
        #   E||g_B||^2 = ||G||^2 + tr(Sigma)/B
        b_big_a, b_small_a = max(alphas), min(alphas)
        B_big = sum(max(0, min(len(rd['value']), round(b_big_a * ki)))
                    for rd, ki in zip(rounds, k))
        B_small = sum(max(0, min(len(rd['value']), round(b_small_a * ki)))
                      for rd, ki in zip(rounds, k))
        gb = float(np.mean(sqnorm_by_alpha[b_big_a]))
        gs = float(np.mean(sqnorm_by_alpha[b_small_a]))
        b_simple = None
        if B_big != B_small:
            G2 = (B_big * gb - B_small * gs) / (B_big - B_small)
            trS = (gs - gb) / (1.0 / B_small - 1.0 / B_big)
            if G2 > 0 and trS > 0:
                b_simple = float(trS / G2)

        report['updates'][u] = {
            'round_sizes': [len(rd['value']) for rd in rounds],
            'per_round_k': k, 'B_step': B_step, 'buffer_total': buf_total,
            'gfull_norm': gfull_norm,
            'alphas': alpha_rows,
            'critical_alpha': crit['alpha'] if crit else None,
            'critical_n_raw': crit['n_raw'] if crit else None,
            'b_simple_raw': b_simple,
        }
        if crit:
            print(f"  -> smallest alpha with cos>={args.cos_threshold}: {crit['alpha']} "
                  f"(n_raw={crit['n_raw']}); suggests SAMPLE_RATIO ~ "
                  f"{sample_ratio * crit['alpha']:.4f}")
        else:
            print(f"  -> no alpha reached cos>={args.cos_threshold}; batch is not redundant")
        if b_simple is not None:
            print(f"  B_simple ~ {b_simple:.0f} raw samples (cf B_step={B_step})")

        del model, g_full
        torch.cuda.empty_cache()

    # conservative pick = latest update probed
    last = max(updates)
    crit_last = report['updates'][last]['critical_alpha']
    report['conservative'] = {
        'update': last, 'critical_alpha': crit_last,
        'suggested_sample_ratio': (sample_ratio * crit_last) if crit_last else None,
    }
    print("\n" + "=" * 70)
    if crit_last:
        print(f"Conservative (u={last}): critical alpha={crit_last} -> "
              f"SAMPLE_RATIO {sample_ratio} -> {sample_ratio * crit_last:.4f}")
    else:
        print(f"Conservative (u={last}): no redundancy at cos>={args.cos_threshold}")

    json_path = os.path.join(args.out_dir, 'q2_critical_batch.json')
    with open(json_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"wrote {json_path}")

    if not args.no_plot:
        _plot(args, report)


def _plot(args, report) -> None:
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"(plot skipped: {e})")
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    for u, ud in report['updates'].items():
        xs = [r['n_raw'] for r in ud['alphas']]
        ys = [r['cos_mean'] for r in ud['alphas']]
        es = [r['cos_std'] for r in ud['alphas']]
        ax.errorbar(xs, ys, yerr=es, marker='o', capsize=3, label=f'u={u}')
    ax.axhline(args.cos_threshold, color='red', ls='--', lw=1,
               label=f'cos={args.cos_threshold}')
    ax.set_xscale('log')
    ax.set_xlabel('effective raw batch (n_raw)')
    ax.set_ylabel('cos(g_alpha, g_full)')
    ax.set_title('Q2 gradient alignment vs batch size')
    ax.legend(fontsize=8)
    fig.tight_layout()
    png = os.path.join(args.out_dir, 'q2_critical_batch.png')
    fig.savefig(png, dpi=120)
    print(f"wrote {png}")


if __name__ == '__main__':
    main()
