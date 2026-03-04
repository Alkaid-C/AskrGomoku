"""
KL Stability Tests

Test 1: Generate a second set of 8 trajectories from current model, recompute KL,
        compare with original KL values. Measures trajectory sampling noise.

Test 2: Generate 8 trajectory sets from 8 different historical models (current - n*2048),
        compute KL using each, compare with current-model trajectories.
        Measures systematic bias from trajectory source.
"""

import glob
import json
import os
import random
import re
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gomoku import (
    RENJU_OPENING_SEQUENCES,
    play_episodes_batched,
)
from model import GomokuPolicyNet

GAMMA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gamma")
OUTPUT_DIR = os.path.join(GAMMA_DIR, "kl_analysis")
STRIDE = 128
LOGIT_MASK_VALUE = -1e9
EVAL_TEMP = 1.0
NUM_TRAJECTORIES = 8


def load_model(checkpoint_path, device):
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model = GomokuPolicyNet().to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        return model
    except Exception as e:
        print(f"  Warning: Failed to load {checkpoint_path}: {e}")
        return None


def generate_trajectories(model, device, num_trajs=8, seed=None):
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
    opening_ids = random.sample(range(len(RENJU_OPENING_SEQUENCES)), num_trajs)
    pairs = [(model, model)] * num_trajs
    current_is_black = [True] * num_trajs
    from gomoku import select_action_batch
    trajs = play_episodes_batched(
        pairs, current_is_black, EVAL_TEMP, device,
        select_action_batch, opening_ids,
    )
    position_sequences = []
    for traj in trajs:
        positions = []
        for obs, mask in zip(traj.observations, traj.legal_masks):
            positions.append((obs.copy(), mask.copy()))
        position_sequences.append(positions)
    total = sum(len(ps) for ps in position_sequences)
    lengths = [len(ps) for ps in position_sequences]
    print(f"    {num_trajs} trajs, {total} positions, lengths={lengths}")
    return position_sequences, opening_ids


def fingerprint_model(model, all_positions, device):
    obs_list = [pos[0] for traj in all_positions for pos in traj]
    mask_list = [pos[1] for traj in all_positions for pos in traj]
    n = len(obs_list)
    all_log_probs = []
    batch_size = 256
    with torch.inference_mode():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            obs_batch = torch.from_numpy(np.stack(obs_list[start:end])).float().to(device)
            mask_batch = torch.from_numpy(np.stack(mask_list[start:end])).bool().to(device)
            logits_grid = model.forward_policy_only(obs_batch)
            logits = logits_grid.squeeze(1).view(-1, 225)
            mask_flat = mask_batch.view(-1, 225)
            logits = logits.masked_fill(~mask_flat, LOGIT_MASK_VALUE)
            log_probs = F.log_softmax(logits, dim=1)
            all_log_probs.append(log_probs.cpu().numpy())
    return np.concatenate(all_log_probs, axis=0)


def symmetric_kl(log_probs_a, log_probs_b):
    probs_a = np.exp(log_probs_a)
    probs_b = np.exp(log_probs_b)
    kl_ab = np.sum(probs_a * (log_probs_a - log_probs_b), axis=1)
    kl_ba = np.sum(probs_b * (log_probs_b - log_probs_a), axis=1)
    sym_kl = (kl_ab + kl_ba) / 2.0
    return float(np.mean(sym_kl))


def discover_checkpoints(gamma_dir, stride):
    files = glob.glob(os.path.join(gamma_dir, "checkpoint_update_*.pt"))
    pattern = re.compile(r"checkpoint_update_(\d+)\.pt")
    updates = []
    for f in files:
        m = pattern.match(os.path.basename(f))
        if m:
            u = int(m.group(1))
            if u % stride == 0:
                updates.append(u)
    return sorted(updates)


def compute_all_kls(reference_model, reference_log_probs, trajectories,
                    all_updates, device):
    """Compute KL of all checkpoints against reference, using given trajectories."""
    kls = {}
    for chunk_start in range(0, len(all_updates), 16):
        chunk = all_updates[chunk_start:chunk_start + 16]
        for u in chunk:
            path = os.path.join(GAMMA_DIR, f"checkpoint_update_{u}.pt")
            model = load_model(path, device)
            if model is None:
                continue
            lp = fingerprint_model(model, trajectories, device)
            kls[u] = symmetric_kl(reference_log_probs, lp)
            del model
        torch.cuda.empty_cache()
    return kls


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    with open(os.path.join(GAMMA_DIR, "training_state.json")) as f:
        state = json.load(f)
    current_update = state["current_update"]

    current_model = load_model(
        os.path.join(GAMMA_DIR, f"checkpoint_update_{current_update}.pt"), device
    )
    all_updates = discover_checkpoints(GAMMA_DIR, STRIDE)
    print(f"Current model: {current_update}, {len(all_updates)} checkpoints to test")

    # Load original KL values for comparison
    import csv
    original_kls = {}
    csv_path = os.path.join(OUTPUT_DIR, "kl_winrate_summary.csv")
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            original_kls[int(row["update"])] = float(row["kl_vs_current"])

    # =========================================================================
    # Test 1: Trajectory sampling noise
    # =========================================================================
    print("\n" + "=" * 70)
    print("TEST 1: Trajectory sampling noise (5 independent trajectory sets)")
    print("=" * 70)

    # Generate 5 new trajectory sets with different seeds
    test_seeds = [100, 200, 300, 400, 500]
    all_test_kls = []

    for trial, seed in enumerate(test_seeds):
        print(f"\n  Trial {trial + 1}/5 (seed={seed}):")
        trajs, _ = generate_trajectories(current_model, device, NUM_TRAJECTORIES, seed=seed)
        ref_lp = fingerprint_model(current_model, trajs, device)
        kls = compute_all_kls(current_model, ref_lp, trajs, all_updates, device)
        all_test_kls.append(kls)
        print(f"    Done ({len(kls)} checkpoints)")

    # Compare each trial against original
    print("\n  --- Noise Analysis ---")
    common_updates = sorted(set.intersection(
        set(original_kls.keys()),
        *[set(k.keys()) for k in all_test_kls]
    ))
    print(f"  Common checkpoints: {len(common_updates)}")

    orig_arr = np.array([original_kls[u] for u in common_updates])

    # Compute stats for each trial vs original
    trial_diffs = []
    for trial, kls in enumerate(all_test_kls):
        trial_arr = np.array([kls[u] for u in common_updates])
        diff = trial_arr - orig_arr
        rel_diff = np.abs(diff) / (orig_arr + 1e-10)
        trial_diffs.append(trial_arr)
        print(f"\n  Trial {trial+1} vs original:")
        print(f"    Mean abs diff:  {np.mean(np.abs(diff)):.6f} nats")
        print(f"    Mean rel diff:  {np.mean(rel_diff):.4f} ({np.mean(rel_diff)*100:.1f}%)")
        print(f"    Max abs diff:   {np.max(np.abs(diff)):.6f} nats")
        print(f"    Correlation:    {np.corrcoef(orig_arr, trial_arr)[0,1]:.6f}")

    # Cross-trial variance (how much do different trajectory sets agree?)
    trial_matrix = np.stack(trial_diffs)  # [5, N]
    per_checkpoint_std = np.std(trial_matrix, axis=0)  # [N]
    per_checkpoint_mean = np.mean(trial_matrix, axis=0)  # [N]
    coeff_of_variation = per_checkpoint_std / (per_checkpoint_mean + 1e-10)

    print(f"\n  Cross-trial statistics (5 trials):")
    print(f"    Mean per-checkpoint std:    {np.mean(per_checkpoint_std):.6f} nats")
    print(f"    Mean coeff of variation:    {np.mean(coeff_of_variation):.4f} ({np.mean(coeff_of_variation)*100:.1f}%)")
    print(f"    Max per-checkpoint std:     {np.max(per_checkpoint_std):.6f} nats")

    # By era
    print(f"\n  Noise by era (cross-trial std):")
    for era_start in range(0, max(common_updates) + 1, 4000):
        era_end = era_start + 4000
        era_mask = [(era_start <= u < era_end) for u in common_updates]
        era_stds = per_checkpoint_std[era_mask]
        era_means = per_checkpoint_mean[era_mask]
        if len(era_stds) > 0:
            era_cv = era_stds / (era_means + 1e-10)
            print(f"    {era_start:>6}-{era_end:>6}: "
                  f"mean_std={np.mean(era_stds):.4f}, "
                  f"mean_cv={np.mean(era_cv):.4f} ({np.mean(era_cv)*100:.1f}%), "
                  f"n={len(era_stds)}")

    # =========================================================================
    # Test 2: Systematic bias from trajectory source
    # =========================================================================
    print("\n" + "=" * 70)
    print("TEST 2: Systematic bias (trajectories from 8 different model eras)")
    print("=" * 70)

    # Use models at current - n*2048 for n=1..8
    source_updates = [current_update - n * 2048 for n in range(1, 9)]
    # Filter to existing checkpoints
    existing = set(u for u in source_updates
                   if os.path.exists(os.path.join(GAMMA_DIR, f"checkpoint_update_{u}.pt")))
    source_updates = sorted(existing)
    print(f"  Source models: {source_updates}")

    source_kls = {}  # source_update -> {checkpoint_update -> kl}

    for src_update in source_updates:
        print(f"\n  Source model: update {src_update}")
        src_model = load_model(
            os.path.join(GAMMA_DIR, f"checkpoint_update_{src_update}.pt"), device
        )
        if src_model is None:
            continue

        # Generate trajectories from this source model
        trajs, _ = generate_trajectories(src_model, device, NUM_TRAJECTORIES, seed=42)

        # Fingerprint current model on these trajectories (reference is still current model)
        ref_lp = fingerprint_model(current_model, trajs, device)

        # Compute KL of all checkpoints against current model, using source's trajectories
        kls = compute_all_kls(current_model, ref_lp, trajs, all_updates, device)
        source_kls[src_update] = kls
        print(f"    Done ({len(kls)} checkpoints)")

        del src_model
        torch.cuda.empty_cache()

    # Compare each source's KL values against the original (current model's trajectories)
    print("\n  --- Bias Analysis ---")
    for src_update in source_updates:
        kls = source_kls[src_update]
        common = sorted(set(original_kls.keys()) & set(kls.keys()))
        if not common:
            continue
        orig_arr = np.array([original_kls[u] for u in common])
        src_arr = np.array([kls[u] for u in common])
        diff = src_arr - orig_arr
        rel_diff = diff / (orig_arr + 1e-10)
        corr = np.corrcoef(orig_arr, src_arr)[0, 1]

        print(f"\n  Source {src_update} (age={current_update - src_update}) vs original (current's trajs):")
        print(f"    Mean diff (signed):  {np.mean(diff):+.6f} nats (bias direction)")
        print(f"    Mean abs diff:       {np.mean(np.abs(diff)):.6f} nats")
        print(f"    Mean rel diff:       {np.mean(np.abs(rel_diff)):.4f} ({np.mean(np.abs(rel_diff))*100:.1f}%)")
        print(f"    Correlation:         {corr:.6f}")

        # Signed relative diff by era — does bias favor old or new checkpoints?
        print(f"    Signed relative diff by era:")
        for era_start in range(0, max(common) + 1, 8000):
            era_end = era_start + 8000
            era_mask = np.array([(era_start <= u < era_end) for u in common])
            if np.sum(era_mask) > 0:
                era_rel = rel_diff[era_mask]
                print(f"      {era_start:>6}-{era_end:>6}: mean_rel_diff={np.mean(era_rel):+.4f} "
                      f"({np.mean(era_rel)*100:+.1f}%), n={np.sum(era_mask)}")

    # =========================================================================
    # Plots
    # =========================================================================
    print("\n=== Generating plots ===")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # --- Plot Test 1: noise ---
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Left: scatter of original vs trial KLs (one trial)
        ax = axes[0]
        trial_arr = np.array([all_test_kls[0][u] for u in common_updates])
        orig_plot = np.array([original_kls[u] for u in common_updates])
        ax.scatter(orig_plot, trial_arr, alpha=0.5, s=15)
        lims = [0, max(orig_plot.max(), trial_arr.max()) * 1.05]
        ax.plot(lims, lims, "k--", alpha=0.3)
        ax.set_xlabel("KL (original trajectories)")
        ax.set_ylabel("KL (new trajectories, trial 1)")
        ax.set_title("Test 1: Trajectory Noise (trial 1 vs original)")
        ax.set_aspect("equal")

        # Right: coefficient of variation vs KL magnitude
        ax = axes[1]
        ax.scatter(per_checkpoint_mean, coeff_of_variation, alpha=0.5, s=15)
        ax.set_xlabel("Mean KL across trials")
        ax.set_ylabel("Coefficient of variation (std/mean)")
        ax.set_title("Test 1: Noise Level vs KL Magnitude")
        ax.axhline(y=0.1, color="red", linestyle="--", alpha=0.5, label="10% CV")
        ax.legend()

        fig.tight_layout()
        fig.savefig(os.path.join(OUTPUT_DIR, "test1_noise.png"), dpi=150)
        plt.close(fig)
        print("  test1_noise.png saved")

        # --- Plot Test 2: bias ---
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Left: KL from different sources vs original, for a few sources
        ax = axes[0]
        colors_t2 = plt.cm.viridis(np.linspace(0, 1, len(source_updates)))
        for i, src_update in enumerate(source_updates):
            kls = source_kls[src_update]
            common = sorted(set(original_kls.keys()) & set(kls.keys()))
            orig_arr_t2 = np.array([original_kls[u] for u in common])
            src_arr_t2 = np.array([kls[u] for u in common])
            age = current_update - src_update
            ax.scatter(orig_arr_t2, src_arr_t2, alpha=0.3, s=10, color=colors_t2[i],
                      label=f"age={age}")
        lims = [0, 5]
        ax.plot(lims, lims, "k--", alpha=0.3)
        ax.set_xlabel("KL (current model's trajectories)")
        ax.set_ylabel("KL (other model's trajectories)")
        ax.set_title("Test 2: Trajectory Source Bias")
        ax.legend(fontsize=7)
        ax.set_aspect("equal")

        # Right: signed mean bias vs source age
        ax = axes[1]
        ages = []
        biases = []
        abs_biases = []
        for src_update in source_updates:
            kls = source_kls[src_update]
            common = sorted(set(original_kls.keys()) & set(kls.keys()))
            orig_arr_t2 = np.array([original_kls[u] for u in common])
            src_arr_t2 = np.array([kls[u] for u in common])
            ages.append(current_update - src_update)
            biases.append(np.mean(src_arr_t2 - orig_arr_t2))
            abs_biases.append(np.mean(np.abs(src_arr_t2 - orig_arr_t2)))
        ax.plot(ages, biases, "o-", label="Mean signed bias")
        ax.plot(ages, abs_biases, "s--", label="Mean abs diff")
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        ax.set_xlabel("Source model age (updates behind current)")
        ax.set_ylabel("KL difference (nats)")
        ax.set_title("Test 2: Bias vs Source Model Age")
        ax.legend()

        fig.tight_layout()
        fig.savefig(os.path.join(OUTPUT_DIR, "test2_bias.png"), dpi=150)
        plt.close(fig)
        print("  test2_bias.png saved")

    except ImportError as e:
        print(f"  Plotting skipped: {e}")

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
