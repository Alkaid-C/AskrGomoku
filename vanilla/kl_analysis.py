"""
KL Divergence & Win Rate Analysis for Historical Checkpoints

Workflow:
1. Load current model + pool from training_state.json
2. Generate 8 self-play trajectories (Renju openings) with current model
3. Fingerprint all stride-128 historical checkpoints on those trajectories
4. Play 64 games per checkpoint vs current model
5. Produce: policy vectors (.npz), scatter plots, PCA visualization, KL stats
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

# Add vanilla/ to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gomoku import (
    RENJU_OPENING_SEQUENCES,
    GameState,
    play_eval_games,
    play_episodes_batched,
    select_action_batch_eval,
)
from model import GomokuPolicyNet

# ============================================================================
# Config
# ============================================================================

GAMMA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gamma")
OUTPUT_DIR = os.path.join(GAMMA_DIR, "kl_analysis")
STRIDE = 128
NUM_TRAJECTORIES = 8
EVAL_ROUNDS = 32  # 32 rounds × 2 colors = 64 games per checkpoint
EVAL_TEMP = 1.0
CHECKPOINT_BATCH_SIZE = 16  # Models loaded simultaneously for fingerprinting/eval
LOGIT_MASK_VALUE = -1e9


def load_model(checkpoint_path, device):
    """Load a model from checkpoint."""
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model = GomokuPolicyNet().to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        return model
    except Exception as e:
        print(f"  Warning: Failed to load {checkpoint_path}: {e}")
        return None


def discover_checkpoints(gamma_dir, stride):
    """Find all stride-aligned checkpoint update numbers."""
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


# ============================================================================
# Step 1: Generate trajectories via self-play
# ============================================================================

def generate_trajectories(model, device, num_trajs=8):
    """
    Generate trajectories by self-play with random Renju openings.

    Returns list of trajectory position sequences. Each trajectory is a list of
    (obs [3,15,15], legal_mask [15,15]) tuples — one per move in the game.
    """
    # Sample random Renju openings
    opening_ids = random.sample(range(len(RENJU_OPENING_SEQUENCES)), num_trajs)

    pairs = [(model, model)] * num_trajs
    current_is_black = [True] * num_trajs  # doesn't matter for self-play

    from gomoku import select_action_batch

    trajs = play_episodes_batched(
        pairs, current_is_black, EVAL_TEMP, device,
        select_action_batch, opening_ids,
    )

    # Extract position sequences
    position_sequences = []
    for traj in trajs:
        positions = []
        for obs, mask in zip(traj.observations, traj.legal_masks):
            positions.append((obs.copy(), mask.copy()))
        position_sequences.append(positions)

    total_positions = sum(len(ps) for ps in position_sequences)
    game_lengths = [len(ps) for ps in position_sequences]
    print(f"  Generated {num_trajs} trajectories: {total_positions} total positions")
    print(f"  Game lengths: {game_lengths}")
    return position_sequences, opening_ids


# ============================================================================
# Step 2: Fingerprint a model on trajectory positions
# ============================================================================

def fingerprint_model(model, all_positions, device):
    """
    Run model on all trajectory positions, return masked log-softmax policy vectors.

    Returns: numpy array [N_positions, 225] of log-probabilities (masked, then log-softmax).
    """
    obs_list = [pos[0] for traj in all_positions for pos in traj]
    mask_list = [pos[1] for traj in all_positions for pos in traj]

    n = len(obs_list)
    all_log_probs = []

    # Process in batches to avoid OOM
    batch_size = 256
    with torch.inference_mode():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            obs_batch = torch.from_numpy(np.stack(obs_list[start:end])).float().to(device)
            mask_batch = torch.from_numpy(np.stack(mask_list[start:end])).bool().to(device)

            logits_grid = model.forward_policy_only(obs_batch)
            logits = logits_grid.squeeze(1).view(-1, 225)  # [B, 225]

            # Apply mask
            mask_flat = mask_batch.view(-1, 225)
            logits = logits.masked_fill(~mask_flat, LOGIT_MASK_VALUE)

            # Log-softmax for KL computation
            log_probs = F.log_softmax(logits, dim=1)
            all_log_probs.append(log_probs.cpu().numpy())

    return np.concatenate(all_log_probs, axis=0)  # [N, 225]


# ============================================================================
# Step 3: KL divergence computation
# ============================================================================

def symmetric_kl(log_probs_a, log_probs_b):
    """
    Compute symmetric KL divergence (Jensen-Shannon-like average of both directions).

    Args:
        log_probs_a: [N, 225] log-probabilities
        log_probs_b: [N, 225] log-probabilities

    Returns:
        Scalar: mean symmetric KL across all positions (in nats).
    """
    # Convert to probabilities for the expectation
    probs_a = np.exp(log_probs_a)
    probs_b = np.exp(log_probs_b)

    # KL(A || B) = sum_x P_A(x) * (log P_A(x) - log P_B(x))
    kl_ab = np.sum(probs_a * (log_probs_a - log_probs_b), axis=1)  # [N]
    kl_ba = np.sum(probs_b * (log_probs_b - log_probs_a), axis=1)  # [N]

    # Symmetric KL = (KL(A||B) + KL(B||A)) / 2, averaged over positions
    sym_kl = (kl_ab + kl_ba) / 2.0
    return float(np.mean(sym_kl))


# ============================================================================
# Step 4: Win rate evaluation
# ============================================================================

def evaluate_win_rates(current_model, opponents, device, num_rounds=EVAL_ROUNDS):
    """
    Evaluate current model against multiple opponents.

    Returns list of win rates (one per opponent).
    """
    pairs = []
    current_is_black_list = []
    opponent_indices = []

    for _ in range(num_rounds):
        for opp_idx, opponent in enumerate(opponents):
            pairs.append((current_model, opponent))
            current_is_black_list.append(True)
            opponent_indices.append(opp_idx)
            pairs.append((opponent, current_model))
            current_is_black_list.append(False)
            opponent_indices.append(opp_idx)

    results = play_eval_games(
        pairs, current_is_black_list, EVAL_TEMP, device,
        select_action_fn=select_action_batch_eval,
    )

    per_opp_wins = [0] * len(opponents)
    per_opp_draws = [0] * len(opponents)
    per_opp_games = [0] * len(opponents)

    for (outcome, current_is_black), opp_idx in zip(results, opponent_indices):
        per_opp_games[opp_idx] += 1
        if outcome == GameState.DRAW:
            per_opp_draws[opp_idx] += 1
        elif (outcome == GameState.BLACK_WIN and current_is_black) or \
             (outcome == GameState.WHITE_WIN and not current_is_black):
            per_opp_wins[opp_idx] += 1

    win_rates = []
    for i in range(len(opponents)):
        if per_opp_games[i] > 0:
            win_rates.append((per_opp_wins[i] + 0.5 * per_opp_draws[i]) / per_opp_games[i])
        else:
            win_rates.append(0.5)

    return win_rates


# ============================================================================
# Main
# ============================================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Load current model from training_state.json ---
    state_path = os.path.join(GAMMA_DIR, "training_state.json")
    with open(state_path) as f:
        state = json.load(f)

    current_update = state["current_update"]
    pool_updates = state["opponent_pool_updates"]
    print(f"Current model: update {current_update}")
    print(f"Pool: {pool_updates}")

    current_ckpt = os.path.join(GAMMA_DIR, f"checkpoint_update_{current_update}.pt")
    current_model = load_model(current_ckpt, device)
    if current_model is None:
        print("ERROR: Cannot load current model!")
        return

    # --- Discover historical checkpoints ---
    all_updates = discover_checkpoints(GAMMA_DIR, STRIDE)
    print(f"Found {len(all_updates)} stride-{STRIDE} checkpoints: {all_updates[0]}..{all_updates[-1]}")

    # --- Step 1: Generate trajectories ---
    print("\n=== Generating trajectories ===")
    random.seed(42)
    torch.manual_seed(42)
    trajectories, opening_ids = generate_trajectories(current_model, device, NUM_TRAJECTORIES)
    total_positions = sum(len(t) for t in trajectories)
    print(f"  Opening IDs used: {opening_ids}")

    # --- Step 2: Fingerprint current model ---
    print("\n=== Fingerprinting current model ===")
    current_log_probs = fingerprint_model(current_model, trajectories, device)
    print(f"  Current model fingerprint shape: {current_log_probs.shape}")

    # --- Step 3: Fingerprint all historical checkpoints + compute KL ---
    print(f"\n=== Fingerprinting {len(all_updates)} historical checkpoints ===")
    results = {}  # update_num -> {'log_probs': np.array, 'kl_vs_current': float}

    for chunk_start in range(0, len(all_updates), CHECKPOINT_BATCH_SIZE):
        chunk_updates = all_updates[chunk_start:chunk_start + CHECKPOINT_BATCH_SIZE]
        chunk_end = min(chunk_start + CHECKPOINT_BATCH_SIZE, len(all_updates))
        print(f"  Chunk {chunk_start//CHECKPOINT_BATCH_SIZE + 1}/{(len(all_updates) + CHECKPOINT_BATCH_SIZE - 1)//CHECKPOINT_BATCH_SIZE}: "
              f"updates {chunk_updates[0]}..{chunk_updates[-1]}")

        for update_num in chunk_updates:
            ckpt_path = os.path.join(GAMMA_DIR, f"checkpoint_update_{update_num}.pt")
            model = load_model(ckpt_path, device)
            if model is None:
                continue

            log_probs = fingerprint_model(model, trajectories, device)
            kl = symmetric_kl(current_log_probs, log_probs)

            results[update_num] = {
                "log_probs": log_probs,
                "kl_vs_current": kl,
            }

            del model
        torch.cuda.empty_cache()

    print(f"  Successfully fingerprinted {len(results)} checkpoints")

    # --- Step 4: Evaluate win rates ---
    print(f"\n=== Evaluating win rates ({EVAL_ROUNDS} rounds × 2 colors = {EVAL_ROUNDS*2} games each) ===")
    eval_updates = sorted(results.keys())
    t0 = time.time()

    for chunk_start in range(0, len(eval_updates), CHECKPOINT_BATCH_SIZE):
        chunk = eval_updates[chunk_start:chunk_start + CHECKPOINT_BATCH_SIZE]
        chunk_num = chunk_start // CHECKPOINT_BATCH_SIZE + 1
        total_chunks = (len(eval_updates) + CHECKPOINT_BATCH_SIZE - 1) // CHECKPOINT_BATCH_SIZE

        opponents = []
        valid_updates = []
        for u in chunk:
            ckpt_path = os.path.join(GAMMA_DIR, f"checkpoint_update_{u}.pt")
            model = load_model(ckpt_path, device)
            if model is not None:
                opponents.append(model)
                valid_updates.append(u)

        if opponents:
            win_rates = evaluate_win_rates(current_model, opponents, device)
            for u, wr in zip(valid_updates, win_rates):
                results[u]["win_rate"] = wr

        elapsed = time.time() - t0
        done = chunk_start + len(chunk)
        rate = done / elapsed if elapsed > 0 else 0
        eta = (len(eval_updates) - done) / rate if rate > 0 else 0
        print(f"  Chunk {chunk_num}/{total_chunks}: updates {chunk[0]}..{chunk[-1]} "
              f"({elapsed:.0f}s elapsed, ETA {eta:.0f}s)")

        del opponents
        torch.cuda.empty_cache()

    # --- Save policy vectors ---
    print("\n=== Saving results ===")

    # Save all log_probs as .npz
    log_probs_dict = {}
    for u in sorted(results.keys()):
        log_probs_dict[f"update_{u}"] = results[u]["log_probs"]
    log_probs_dict["current_model"] = current_log_probs
    log_probs_dict["updates"] = np.array(sorted(results.keys()))

    npz_path = os.path.join(OUTPUT_DIR, "policy_fingerprints.npz")
    np.savez_compressed(npz_path, **log_probs_dict)
    npz_size_mb = os.path.getsize(npz_path) / 1e6
    print(f"  Policy fingerprints saved: {npz_path} ({npz_size_mb:.1f} MB)")
    print(f"  Shape per checkpoint: {current_log_probs.shape} "
          f"(uncompressed: {current_log_probs.nbytes / 1e6:.1f} MB each, "
          f"{current_log_probs.nbytes * len(results) / 1e6:.1f} MB total)")

    # Save summary CSV
    csv_path = os.path.join(OUTPUT_DIR, "kl_winrate_summary.csv")
    with open(csv_path, "w") as f:
        f.write("update,kl_vs_current,win_rate_vs_current,in_pool\n")
        for u in sorted(results.keys()):
            r = results[u]
            in_pool = 1 if u in pool_updates else 0
            wr = r.get("win_rate", "")
            f.write(f"{u},{r['kl_vs_current']:.6f},{wr},{in_pool}\n")
    print(f"  Summary CSV saved: {csv_path}")

    # --- Print KL stats ---
    print("\n=== KL Divergence Stats ===")
    kl_values = [results[u]["kl_vs_current"] for u in sorted(results.keys())]
    updates_arr = sorted(results.keys())
    print(f"  N checkpoints: {len(kl_values)}")
    print(f"  Min KL:    {min(kl_values):.6f}")
    print(f"  Max KL:    {max(kl_values):.6f}")
    print(f"  Mean KL:   {np.mean(kl_values):.6f}")
    print(f"  Median KL: {np.median(kl_values):.6f}")
    print(f"  Std KL:    {np.std(kl_values):.6f}")

    # KL by era
    print("\n  KL by era:")
    for era_start in range(0, max(updates_arr) + 1, 4000):
        era_end = era_start + 4000
        era_kls = [results[u]["kl_vs_current"] for u in updates_arr
                   if era_start <= u < era_end]
        if era_kls:
            print(f"    {era_start:>6}-{era_end:>6}: n={len(era_kls):>3}, "
                  f"mean={np.mean(era_kls):.4f}, min={min(era_kls):.4f}, max={max(era_kls):.4f}")

    # KL percentiles
    print(f"\n  Percentiles:")
    for p in [5, 10, 25, 50, 75, 90, 95]:
        print(f"    p{p:>2}: {np.percentile(kl_values, p):.6f}")

    # --- Generate plots ---
    print("\n=== Generating plots ===")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA

        updates_sorted = sorted(results.keys())
        kls = [results[u]["kl_vs_current"] for u in updates_sorted]
        wrs = [results[u].get("win_rate", 0.5) for u in updates_sorted]
        in_pool = [u in pool_updates for u in updates_sorted]

        # --- Plot 1: Win rate vs KL ---
        fig, ax = plt.subplots(figsize=(10, 7))
        colors = ["red" if ip else "steelblue" for ip in in_pool]
        ax.scatter(wrs, kls, c=colors, alpha=0.6, s=20)
        # Label some interesting points
        for u, wr, kl in zip(updates_sorted, wrs, kls):
            if wr < 0.45 or kl > np.percentile(kls, 95):
                ax.annotate(str(u), (wr, kl), fontsize=6, alpha=0.7)
        ax.set_xlabel("Win rate of current model vs checkpoint")
        ax.set_ylabel("Symmetric KL divergence (nats)")
        ax.set_title(f"Win Rate vs KL Divergence (current model = update {current_update})")
        ax.axvline(x=0.5, color="gray", linestyle="--", alpha=0.5)
        ax.legend(["Pool member (red)", "Historical (blue)"], loc="upper right")
        fig.tight_layout()
        fig.savefig(os.path.join(OUTPUT_DIR, "scatter_winrate_vs_kl.png"), dpi=150)
        plt.close(fig)
        print("  scatter_winrate_vs_kl.png saved")

        # --- Plot 2: Update number vs KL ---
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.scatter(updates_sorted, kls, c=colors, alpha=0.6, s=20)
        ax.set_xlabel("Checkpoint update number")
        ax.set_ylabel("Symmetric KL divergence (nats)")
        ax.set_title(f"KL Divergence Over Training (current model = update {current_update})")
        ax.legend(["Pool member (red)", "Historical (blue)"], loc="upper left")
        fig.tight_layout()
        fig.savefig(os.path.join(OUTPUT_DIR, "scatter_update_vs_kl.png"), dpi=150)
        plt.close(fig)
        print("  scatter_update_vs_kl.png saved")

        # --- Plot 3: PCA of policy fingerprints ---
        # Flatten log_probs into feature vectors
        all_log_probs_matrix = np.stack(
            [results[u]["log_probs"].flatten() for u in updates_sorted]
        )
        # Also include current model
        current_flat = current_log_probs.flatten().reshape(1, -1)
        full_matrix = np.concatenate([all_log_probs_matrix, current_flat], axis=0)

        # PCA on softmax probabilities (more interpretable than log-probs)
        all_probs_matrix = np.exp(full_matrix)

        pca = PCA(n_components=2)
        coords = pca.fit_transform(all_probs_matrix)

        fig, ax = plt.subplots(figsize=(10, 8))
        # Color by update number
        update_colors = np.array(list(updates_sorted) + [current_update])
        sc = ax.scatter(
            coords[:-1, 0], coords[:-1, 1],
            c=updates_sorted, cmap="viridis", alpha=0.6, s=20,
        )
        # Current model as star
        ax.scatter(
            coords[-1, 0], coords[-1, 1],
            c="red", marker="*", s=200, zorder=5, label=f"Current ({current_update})",
        )
        # Pool members as diamonds
        pool_indices = [i for i, u in enumerate(updates_sorted) if u in pool_updates]
        if pool_indices:
            ax.scatter(
                coords[pool_indices, 0], coords[pool_indices, 1],
                facecolors="none", edgecolors="red", s=80, linewidths=1.5,
                label="Pool members",
            )
        plt.colorbar(sc, label="Update number")
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
        ax.set_title("PCA of Policy Fingerprints (probability space)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(OUTPUT_DIR, "pca_policy_fingerprints.png"), dpi=150)
        plt.close(fig)
        print("  pca_policy_fingerprints.png saved")
        print(f"  PCA explained variance: PC1={pca.explained_variance_ratio_[0]:.3f}, "
              f"PC2={pca.explained_variance_ratio_[1]:.3f}")

        # --- Plot 4: PCA colored by win rate ---
        fig, ax = plt.subplots(figsize=(10, 8))
        sc = ax.scatter(
            coords[:-1, 0], coords[:-1, 1],
            c=wrs, cmap="RdYlGn", alpha=0.6, s=20, vmin=0.3, vmax=0.8,
        )
        ax.scatter(
            coords[-1, 0], coords[-1, 1],
            c="red", marker="*", s=200, zorder=5, label=f"Current ({current_update})",
        )
        plt.colorbar(sc, label="Win rate (current vs checkpoint)")
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
        ax.set_title("PCA of Policy Fingerprints (colored by win rate)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(OUTPUT_DIR, "pca_by_winrate.png"), dpi=150)
        plt.close(fig)
        print("  pca_by_winrate.png saved")

        # --- Plot 5: Win rate vs update (for context) ---
        fig, ax1 = plt.subplots(figsize=(12, 5))
        ax1.scatter(updates_sorted, wrs, c="steelblue", alpha=0.5, s=15, label="Win rate")
        ax1.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
        ax1.set_xlabel("Checkpoint update number")
        ax1.set_ylabel("Win rate (current vs checkpoint)", color="steelblue")
        ax2 = ax1.twinx()
        ax2.scatter(updates_sorted, kls, c="orange", alpha=0.3, s=10, label="KL")
        ax2.set_ylabel("Symmetric KL (nats)", color="orange")
        ax1.set_title(f"Win Rate & KL vs Training Progress (current = {current_update})")
        fig.tight_layout()
        fig.savefig(os.path.join(OUTPUT_DIR, "winrate_kl_vs_update.png"), dpi=150)
        plt.close(fig)
        print("  winrate_kl_vs_update.png saved")

    except ImportError as e:
        print(f"  Plotting skipped (missing dependency): {e}")

    print("\n=== Done ===")
    print(f"All outputs in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
