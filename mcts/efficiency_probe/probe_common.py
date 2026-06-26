"""
Shared infrastructure for the stage-2 efficiency probes (Q1 drift, Q2 critical
batch). See ../stage2_efficiency_plan.md for the methodology.

This module is imported by `q1_drift.py` and `q2_critical_batch.py`. It must be
run with the parent `mcts/` directory on sys.path so the symlinked
`model.py` / `gomoku.py` / `enhancement.py` and the local `mcts.py` /
`training.py` / `main.py` all resolve. Both entry scripts handle that before
importing this module.

Conventions reused verbatim from training (so the probe measures the real
thing):
  - obs is side-to-move relative (ch0 = mover's stones, ch1 = opponent's).
    Boards are reconstructed with `board_from_observation(obs, Player.BLACK)`:
    the search is perspective-relative and stones are colour-symmetric for
    legality / win detection, so fixing current=BLACK matches the value /
    root_Q sign convention used everywhere in mcts.py.
  - MCTS is invoked with exactly the stage-2 hyperparameters from main.py.
  - The gradient is the per-sample weighted CE + MSE of train_on_mcts_batch,
    with 8-fold augmentation and global weight-total normalization, so the
    accumulated micro-batch gradient equals the exact full-batch gradient.
"""

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gomoku import LOGIT_MASK_VALUE, Player, board_from_observation
from model import GomokuPolicyNet
from training import augment_mcts_batch_8fold

import main as hp  # stage-2 hyperparameter constants (single source of truth)
from mcts import clear_nn_eval_cache, mcts_search_batched

# ============================================================================
# Checkpoint / data loading
# ============================================================================


def load_model(ckpt_path: str, device: torch.device) -> nn.Module:
    """Load a GomokuPolicyNet from a stage-2 checkpoint, in eval mode."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get('model_state_dict', ckpt)
    model = GomokuPolicyNet().to(device)
    model.load_state_dict(state)
    model.eval()
    return model


def checkpoint_path(data_dir: str, update: int) -> str:
    return os.path.join(data_dir, f"checkpoint_update_{update}.pt")


def dump_path(data_dir: str, update: int) -> str:
    return os.path.join(data_dir, "samples", f"samples_update_{update}.npz")


def load_dump(data_dir: str, update: int) -> dict:
    """Load one self-play round's flattened samples.

    Returns a dict of arrays: obs uint8[M,3,15,15], dist f32[M,225],
    value f32[M], policy_weight f32[M], value_weight f32[M].
    """
    d = np.load(dump_path(data_dir, update))
    return {k: d[k] for k in ('obs', 'dist', 'value', 'policy_weight', 'value_weight')}


def boards_from_obs(obs_arr: np.ndarray) -> list:
    """Reconstruct a list of GomokuBoard from a [N,3,15,15] uint8 obs array."""
    return [board_from_observation(o, Player.BLACK) for o in obs_arr]


# ============================================================================
# Stage-2 MCTS search (exact deployment settings)
# ============================================================================


def run_search(
    model: nn.Module,
    boards: list,
    device: torch.device,
    chunk_size: int = 256,
    clear_cache_first: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Run stage-2 MCTS over `boards` and return (visit_dists[N,225], root_q[N]).

    Settings are pinned to main.py's stage-2 values (NUM_SIMULATIONS_S2,
    C_PUCT, entropy_multiplier=None, STAGE2 Dirichlet, gamma, FPU; no harvest),
    so the visit distribution / root Q are exactly the supervision targets the
    real run would store.

    `boards` is processed in chunks of `chunk_size` to bound tree memory; the
    global NN eval cache is shared across chunks (D4 dedup) and cleared once at
    the start when `clear_cache_first` (the caller is responsible for not
    sharing a cache across different model weights).
    """
    if clear_cache_first:
        clear_nn_eval_cache()

    n = len(boards)
    all_dist = np.zeros((n, 225), dtype=np.float32)
    all_q = np.zeros(n, dtype=np.float32)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        dist, q, _ent, _kl, _harv = mcts_search_batched(
            model,
            boards[start:end],
            num_simulations=hp.NUM_SIMULATIONS_S2,
            c_puct=hp.C_PUCT,
            entropy_multiplier=None,
            device=device,
            dirichlet_alpha=hp.STAGE2_DIRICHLET_ALPHA,
            dirichlet_epsilon=hp.STAGE2_DIRICHLET_EPSILON,
            gamma=hp.DISCOUNT_GAMMA,
            fpu_multiplier=hp.FPU_MULTIPLIER,
            harvest_min_visits=None,
        )
        all_dist[start:end] = dist
        all_q[start:end] = q
    return all_dist, all_q


# ============================================================================
# Divergence metrics (Q1)
# ============================================================================


def js_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Per-row Jensen-Shannon divergence (natural log, bounded [0, ln 2]).

    p, q: [N, 225] non-negative rows that each sum to 1 (illegal squares 0).
    """
    p = np.clip(p, 0.0, None)
    q = np.clip(q, 0.0, None)
    m = 0.5 * (p + q)

    def _kl(a, b):
        # sum a * log(a / b) over support of a; terms with a==0 vanish.
        mask = a > 0
        out = np.zeros(a.shape[0], dtype=np.float64)
        ratio = np.where(mask, a / np.clip(b, eps, None), 1.0)
        out = np.sum(np.where(mask, a * np.log(np.clip(ratio, eps, None)), 0.0), axis=-1)
        return out

    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def distribution_summary(x: np.ndarray) -> dict:
    """mean / median / p90 / p99 / max of a 1-D array of per-position values."""
    return {
        'mean': float(np.mean(x)),
        'median': float(np.median(x)),
        'p90': float(np.percentile(x, 90)),
        'p99': float(np.percentile(x, 99)),
        'max': float(np.max(x)),
    }


# ============================================================================
# Gradient vector (Q2)
# ============================================================================


def _trainable_params(model: nn.Module) -> list:
    return [p for p in model.parameters() if p.requires_grad]


def compute_grad_vector(
    model: nn.Module,
    obs: np.ndarray,
    dist: np.ndarray,
    value: np.ndarray,
    policy_weight: np.ndarray,
    value_weight: np.ndarray,
    device: torch.device,
    value_loss_coeff: float,
    micro_batch: int = 512,
    raw_chunk: int = 4096,
) -> torch.Tensor:
    """Exact weighted CE+MSE gradient over the given samples, flattened to one
    fp32 vector on `device`.

    Mirrors training.train_on_mcts_batch: 8-fold augmentation, per-sample
    weighted means normalized by *global* weight totals (so the accumulated
    gradient is the exact full-batch gradient), masked logits. Does not call
    optimizer.step(); reads param.grad and flattens.

    To bound GPU memory the full sample set is processed in chunks of `raw_chunk`
    raw samples (each expanded 8x by augmentation, then forwarded in `micro_batch`
    pieces). The weight-total denominators are computed up front from the full
    weight arrays, so they are identical across chunks and the accumulation
    remains exact regardless of chunking.

    `policy_weight` / `value_weight` are the (already round-decayed, if the
    caller chose to) per-sample weights — they need not be 1.0.
    """
    model.zero_grad(set_to_none=True)

    # Global weight totals over the *augmented* set = 8 * sum(base weights).
    pw_total = float(policy_weight.sum()) * 8.0
    vw_total = float(value_weight.sum()) * 8.0
    pw_total = max(pw_total, 1.0)
    vw_total = max(vw_total, 1.0)

    n = obs.shape[0]
    for c0 in range(0, n, raw_chunk):
        c1 = min(c0 + raw_chunk, n)
        obs_t = torch.from_numpy(obs[c0:c1]).float().to(device)
        dist_t = torch.from_numpy(dist[c0:c1]).float().to(device)
        val_t = torch.from_numpy(value[c0:c1].astype(np.float32)).to(device)
        pw_t = torch.from_numpy(policy_weight[c0:c1].astype(np.float32)).to(device)
        vw_t = torch.from_numpy(value_weight[c0:c1].astype(np.float32)).to(device)
        occupied = (obs[c0:c1, 0] | obs[c0:c1, 1])
        mask_t = torch.from_numpy((1 - occupied).astype(np.uint8)).bool().to(device)

        obs_t, dist_t, mask_t, val_t, pw_t, vw_t = augment_mcts_batch_8fold(
            obs_t, dist_t, mask_t, val_t, pw_t, vw_t
        )
        n_aug = obs_t.shape[0]
        for start in range(0, n_aug, micro_batch):
            end = min(start + micro_batch, n_aug)
            mb = end - start
            logits, pred_v = model(obs_t[start:end])
            logits = logits.squeeze(1).view(mb, 225)
            pred_v = pred_v.squeeze(-1)
            logits = logits.masked_fill(~mask_t[start:end].view(mb, 225), LOGIT_MASK_VALUE)
            log_probs = F.log_softmax(logits, dim=-1)
            ce = -(dist_t[start:end] * log_probs).sum(dim=-1)
            se = (pred_v - val_t[start:end]) ** 2
            policy_term = (pw_t[start:end] * ce).sum()
            value_term = (vw_t[start:end] * se).sum()
            loss = policy_term / pw_total + value_loss_coeff * value_term / vw_total
            loss.backward()

    grads = [
        p.grad.detach().reshape(-1) if p.grad is not None
        else torch.zeros(p.numel(), device=device)
        for p in _trainable_params(model)
    ]
    g = torch.cat(grads)
    model.zero_grad(set_to_none=True)
    return g


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.dot(a, b) / (a.norm() * b.norm() + 1e-30))
