"""
PCA analysis of policy evolution over training.

1. Randomly select 32 checkpoints → self-play → 32 trajectories (board positions).
2. Load all n*128 checkpoints, collect softmax policy (illegal moves zeroed) on every position.
3. Concatenate per checkpoint → PCA → 2D scatter colored by update count.
"""

import glob
import os
import random
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA

# Add vanilla/ to path so we can import model and gomoku
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import GomokuPolicyNet
from gomoku import GomokuBoard, GameState, encode_observation

RELEASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "release")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_TRAJ = 32
EVAL_INTERVAL = 128


def list_checkpoints():
    """Return dict of {update_id: path} for all checkpoints."""
    pattern = os.path.join(RELEASE_DIR, "checkpoint_update_*.pt")
    result = {}
    for path in glob.glob(pattern):
        m = re.search(r"checkpoint_update_(\d+)\.pt$", path)
        if m:
            result[int(m.group(1))] = path
    return result


def load_model(path):
    """Load a checkpoint into a GomokuPolicyNet on DEVICE, eval mode."""
    model = GomokuPolicyNet()
    checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()
    return model


def self_play_trajectory(model):
    """
    Play one game of self-play and return a list of (observation, legal_mask) pairs.
    observation: np.ndarray [3, 15, 15], legal_mask: np.ndarray [15, 15].
    """
    board = GomokuBoard()
    positions = []

    while True:
        c0, c1, _ = board.GetBoardState()
        legal_mask, _ = board.GetLegalMoves()
        obs = encode_observation(c0, c1)
        positions.append((obs.copy(), legal_mask.copy()))

        # Get policy and sample a move
        obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits_grid = model.forward_policy_only(obs_t)  # [1, 1, 15, 15]
        logits = logits_grid.view(1, 225)  # [1, 225]
        legal_flat = torch.from_numpy(legal_mask.reshape(225)).float().to(DEVICE)
        logits = logits.where(legal_flat.bool(), torch.tensor(-1e9, device=DEVICE))
        probs = F.softmax(logits, dim=-1)
        action = torch.multinomial(probs, 1).item()
        row, col = action // 15, action % 15

        state = board.Move((row, col))
        if state != GameState.CONTINUE:
            break

    return positions


def collect_policies(model, all_positions):
    """
    Run model on all positions, return concatenated softmax policy vector
    with illegal moves zeroed.
    all_positions: list of (obs [3,15,15], legal_mask [15,15])
    Returns: np.ndarray of shape [num_positions * 225]
    """
    batch_size = 256
    all_policy = []

    for i in range(0, len(all_positions), batch_size):
        batch = all_positions[i:i + batch_size]
        obs_batch = np.stack([obs for obs, _ in batch])
        legal_batch = np.stack([lm.reshape(225) for _, lm in batch])

        obs_t = torch.from_numpy(obs_batch).float().to(DEVICE)
        legal_t = torch.from_numpy(legal_batch).float().to(DEVICE)

        with torch.no_grad():
            logits_grid = model.forward_policy_only(obs_t)  # [B, 1, 15, 15]
        logits = logits_grid.view(-1, 225)
        logits = logits.where(legal_t.bool(), torch.tensor(-1e9, device=DEVICE))
        probs = F.softmax(logits, dim=-1)
        probs = probs * legal_t  # zero out illegal
        policy_np = probs.cpu().numpy()  # [B, 225]
        all_policy.append(policy_np)

    all_policy = np.concatenate(all_policy, axis=0)  # [num_positions, 225]
    return all_policy.reshape(-1)  # flatten to single vector


def main():
    all_ckpts = list_checkpoints()
    all_ids = sorted(all_ckpts.keys())
    print(f"Total checkpoints: {len(all_ids)}, range: {all_ids[0]} - {all_ids[-1]}")

    # Step 1: randomly select 32 checkpoints for trajectory generation
    traj_ids = sorted(random.sample(all_ids, NUM_TRAJ))
    print(f"Generating {NUM_TRAJ} trajectories from checkpoints: {traj_ids}")

    all_positions = []  # flat list of (obs, legal_mask) across all trajectories
    traj_lengths = []
    for i, uid in enumerate(traj_ids):
        model = load_model(all_ckpts[uid])
        traj = self_play_trajectory(model)
        all_positions.extend(traj)
        traj_lengths.append(len(traj))
        del model
        torch.cuda.empty_cache()
        print(f"  Traj {i+1}/{NUM_TRAJ}: checkpoint {uid}, {len(traj)} positions")

    total_positions = len(all_positions)
    print(f"Total positions across all trajectories: {total_positions}")

    # Save trajectories for reproducibility
    traj_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pca_trajectories.npz")
    obs_arr = np.stack([obs for obs, _ in all_positions])
    legal_arr = np.stack([lm for _, lm in all_positions])
    np.savez_compressed(traj_path, obs=obs_arr, legal=legal_arr,
                        traj_checkpoint_ids=np.array(traj_ids),
                        traj_lengths=np.array(traj_lengths))
    print(f"Saved trajectories to {traj_path}")

    # Step 2: select n*128 checkpoints and collect policy on all positions
    eval_ids = sorted([uid for uid in all_ids if uid % EVAL_INTERVAL == 0])
    print(f"Collecting policy from {len(eval_ids)} checkpoints (n*{EVAL_INTERVAL})")

    feature_dim = total_positions * 225
    data_matrix = np.empty((len(eval_ids), feature_dim), dtype=np.float32)

    for i, uid in enumerate(eval_ids):
        model = load_model(all_ckpts[uid])
        feature_vec = collect_policies(model, all_positions)
        data_matrix[i] = feature_vec
        del model
        torch.cuda.empty_cache()
        if (i + 1) % 20 == 0 or i == 0 or i == len(eval_ids) - 1:
            print(f"  Processed {i+1}/{len(eval_ids)} (checkpoint {uid})")

    # Save data matrix for re-analysis without re-collecting
    matrix_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pca_data_matrix.npz")
    np.savez_compressed(matrix_path, data_matrix=data_matrix, eval_ids=np.array(eval_ids),
                        traj_checkpoint_ids=np.array(traj_ids), traj_lengths=np.array(traj_lengths))
    print(f"Saved data matrix to {matrix_path}")

    # Step 3: PCA
    print("Running PCA...")
    n_components = min(10, len(eval_ids))
    pca = PCA(n_components=n_components)
    all_coords = pca.fit_transform(data_matrix)
    coords = all_coords[:, :2]
    print(f"Explained variance ratio (top {n_components}):")
    cumulative = 0.0
    for i, ratio in enumerate(pca.explained_variance_ratio_):
        cumulative += ratio
        print(f"  PC{i+1}: {ratio:.4f} ({ratio:.1%})  cumulative: {cumulative:.1%}")

    # Step 4: Plot
    fig, ax = plt.subplots(figsize=(12, 8))
    sc = ax.scatter(
        coords[:, 0], coords[:, 1],
        c=eval_ids, cmap="viridis", s=12, alpha=0.8
    )
    # Draw a line connecting consecutive checkpoints to show the trajectory
    ax.plot(coords[:, 0], coords[:, 1], color="gray", linewidth=0.5, alpha=0.4, zorder=0)

    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Update count")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
    ax.set_title("Policy Evolution via PCA (vanilla model, self-play trajectories)")

    # Mark start and end
    ax.annotate("start", (coords[0, 0], coords[0, 1]), fontsize=9, fontweight="bold")
    ax.annotate("end", (coords[-1, 0], coords[-1, 1]), fontsize=9, fontweight="bold")

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pca_policy_evolution.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {out_path}")
    plt.close(fig)

    # Also save raw data for further analysis
    npz_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pca_policy_evolution.npz")
    np.savez_compressed(
        npz_path,
        coords=coords,
        eval_ids=np.array(eval_ids),
        explained_variance_ratio=pca.explained_variance_ratio_,
        traj_checkpoint_ids=np.array(traj_ids),
        traj_lengths=np.array(traj_lengths),
    )
    print(f"Saved raw data to {npz_path}")


if __name__ == "__main__":
    main()
