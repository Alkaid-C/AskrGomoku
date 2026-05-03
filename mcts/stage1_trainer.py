"""
Stage 1 trainer: offline distillation from precomputed teacher MCTS data.

A freshly initialized student trains on the static `(obs, visit_dist, root_Q)`
shards produced by `data_generator.generate_stage1_data`. No MCTS is run here.
Loss is plain CE + MSE; targets are the raw visit distributions (no sharpening).
"""

import glob
import json
import os
import time
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from csv_logger import Stage1CSVLogger
from gomoku import LOGIT_MASK_VALUE
from model import GomokuPolicyNet
from training import augment_mcts_batch_8fold

TRAINING_STATE_FILE = "training_state.json"
FINAL_NAME = "stage1_final.pt"


def _load_shards(data_dir: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    paths = sorted(glob.glob(os.path.join(data_dir, "stage1_shard_*.npz")))
    if not paths:
        raise RuntimeError(f"No shards found in {data_dir}")
    print(f"Loading {len(paths)} shard(s) from {data_dir}")

    obs_chunks: list[np.ndarray] = []
    dist_chunks: list[np.ndarray] = []
    q_chunks: list[np.ndarray] = []
    for p in paths:
        with np.load(p) as data:
            obs_chunks.append(data['obs'])
            dist_chunks.append(data['visit_dist'])
            q_chunks.append(data['root_Q'])

    obs = np.concatenate(obs_chunks, axis=0)
    dist = np.concatenate(dist_chunks, axis=0)
    q = np.concatenate(q_chunks, axis=0)
    print(f"  Loaded {obs.shape[0]} samples ({obs.nbytes / 1e6:.0f} MB obs, "
          f"{dist.nbytes / 1e6:.0f} MB visit_dist)")
    return obs, dist, q


def _epoch_permutation(n: int, seed: int, epoch: int) -> np.ndarray:
    rng = np.random.default_rng(seed * 1_000_003 + epoch)
    return rng.permutation(n)


def _save_checkpoint(
    path: str,
    update: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
) -> None:
    torch.save({
        'update': update,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }, path)


def _save_state(output_dir: str, update: int, kl_ema: Optional[float]) -> None:
    state = {'current_update': update, 'kl_ema': kl_ema}
    path = os.path.join(output_dir, TRAINING_STATE_FILE)
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, path)


def run_stage1(
    data_dir: str,
    output_dir: str,
    *,
    epochs: int,
    raw_batch_per_update: int,
    train_batch_size: int,
    learning_rate: float,
    min_lr: float,
    weight_decay: float,
    value_loss_coeff: float,
    grad_clip_norm: float,
    kl_ema_window: int,
    kl_ema_threshold: Optional[float],
    checkpoint_interval: int,
    seed: int,
    device: torch.device,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    obs_np, dist_np, q_np = _load_shards(data_dir)
    n_samples = obs_np.shape[0]
    updates_per_epoch = (n_samples + raw_batch_per_update - 1) // raw_batch_per_update
    total_updates = epochs * updates_per_epoch
    print(f"Stage 1: {epochs} epochs * {updates_per_epoch} upd/ep = {total_updates} total updates")
    print(f"  raw batch={raw_batch_per_update}, train batch={train_batch_size}")
    print(f"  lr={learning_rate} -> {min_lr} (cosine), wd={weight_decay}")
    print(f"  KL EMA window={kl_ema_window}, threshold={kl_ema_threshold}")
    print()

    # Pin once on CPU; per-batch slices go to GPU as float32.
    obs_cpu = torch.from_numpy(obs_np)            # uint8
    dist_cpu = torch.from_numpy(dist_np)          # float32
    q_cpu = torch.from_numpy(q_np)                # float32

    model = GomokuPolicyNet().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay, fused=True
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_updates, eta_min=min_lr
    )

    state_path = os.path.join(output_dir, TRAINING_STATE_FILE)
    start_update = 0
    kl_ema: Optional[float] = None
    if os.path.exists(state_path):
        with open(state_path) as f:
            state = json.load(f)
        start_update = state['current_update']
        kl_ema = state.get('kl_ema')
        ckpt_path = os.path.join(output_dir, f"checkpoint_update_{start_update}.pt")
        if not os.path.exists(ckpt_path):
            raise RuntimeError(f"Checkpoint not found: {ckpt_path}")
        print(f"Resuming from update {start_update} ({ckpt_path})")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        print(f"  kl_ema={kl_ema}")
    model.train()

    csv_logger = Stage1CSVLogger(output_dir)
    ema_alpha = 1.0 / kl_ema_window
    train_start = time.time()

    for update in range(start_update, total_updates):
        t0 = time.time()
        epoch = update // updates_per_epoch
        offset_in_epoch = update % updates_per_epoch
        perm = _epoch_permutation(n_samples, seed, epoch)
        start = offset_in_epoch * raw_batch_per_update
        end = min(start + raw_batch_per_update, n_samples)
        idx = perm[start:end]

        mb_obs_u8 = obs_cpu[idx].to(device, non_blocking=True)        # uint8 [B,3,15,15]
        mb_dist = dist_cpu[idx].to(device, non_blocking=True)         # float32 [B,225]
        mb_values = q_cpu[idx].to(device, non_blocking=True)          # float32 [B]

        # Legal mask from observation (empty squares = legal)
        occupied = mb_obs_u8[:, 0] | mb_obs_u8[:, 1]                  # [B,15,15] uint8
        mb_mask = (1 - occupied).bool()
        mb_obs = mb_obs_u8.float()

        obs_aug, dist_aug, mask_aug, val_aug = augment_mcts_batch_8fold(
            mb_obs, mb_dist, mb_mask, mb_values
        )
        n_aug = obs_aug.shape[0]
        shuffle = torch.randperm(n_aug, device=device)
        obs_aug = obs_aug[shuffle]
        dist_aug = dist_aug[shuffle]
        mask_aug = mask_aug[shuffle]
        val_aug = val_aug[shuffle]

        optimizer.zero_grad()
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_kl = 0.0
        total_count = 0

        for s in range(0, n_aug, train_batch_size):
            e = min(s + train_batch_size, n_aug)
            sb_obs = obs_aug[s:e]
            sb_dist = dist_aug[s:e]
            sb_mask = mask_aug[s:e]
            sb_val = val_aug[s:e]
            sb_size = e - s

            logits, pred_values = model(sb_obs)
            logits = logits.squeeze(1).view(sb_size, 225)
            pred_values = pred_values.squeeze(-1)
            logits = logits.masked_fill(~sb_mask.view(sb_size, 225), LOGIT_MASK_VALUE)

            log_probs = F.log_softmax(logits, dim=-1)
            policy_loss = -(sb_dist * log_probs).sum(dim=-1).mean()
            value_loss = F.mse_loss(pred_values, sb_val)
            loss = (policy_loss + value_loss_coeff * value_loss) * (sb_size / n_aug)
            loss.backward()

            with torch.no_grad():
                H_target = -(sb_dist * (sb_dist + 1e-10).log()).sum(dim=-1).mean()
                kl = (policy_loss - H_target).item()

            total_policy_loss += policy_loss.item() * sb_size
            total_value_loss += value_loss.item() * sb_size
            total_kl += kl * sb_size
            total_count += sb_size

        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()

        avg_policy_loss = total_policy_loss / total_count
        avg_value_loss = total_value_loss / total_count
        avg_kl = total_kl / total_count

        if kl_ema is None:
            kl_ema = avg_kl
        else:
            kl_ema = kl_ema + ema_alpha * (avg_kl - kl_ema)

        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        t_train = time.time() - t0

        csv_logger.log(update + 1, {
            'policy_loss': avg_policy_loss,
            'value_loss': avg_value_loss,
            'kl_target_student': avg_kl,
            'kl_ema': kl_ema,
            'lr': current_lr,
            'time_train': t_train,
        })

        elapsed = time.time() - train_start
        eta = elapsed / (update - start_update + 1) * (total_updates - update - 1) \
            if update > start_update else 0.0
        print(
            f"Update {update+1:5d}/{total_updates} | ep {epoch+1}/{epochs} | "
            f"PLoss {avg_policy_loss:.4f} VLoss {avg_value_loss:.4f} | "
            f"KL {avg_kl:.4f} EMA {kl_ema:.4f} | lr {current_lr:.2e} | "
            f"{t_train:.1f}s | elapsed {elapsed/60:.0f}m eta {eta/60:.0f}m"
        )

        if (update + 1) % checkpoint_interval == 0:
            ckpt_id = update + 1
            ckpt_path = os.path.join(output_dir, f"checkpoint_update_{ckpt_id}.pt")
            _save_checkpoint(ckpt_path, ckpt_id, model, optimizer, scheduler)
            _save_state(output_dir, ckpt_id, kl_ema)
            print(f"  Saved: {os.path.basename(ckpt_path)}")

        if kl_ema_threshold is not None and kl_ema < kl_ema_threshold:
            print(f"\nKL EMA {kl_ema:.4f} < threshold {kl_ema_threshold} — early exit.")
            ckpt_id = update + 1
            # Always pair training_state.json with a real checkpoint so a
            # subsequent `stage1` invocation either resumes cleanly or sees
            # only `stage1_final.pt` and treats it as a completed stage.
            ckpt_path = os.path.join(output_dir, f"checkpoint_update_{ckpt_id}.pt")
            if not os.path.exists(ckpt_path):
                _save_checkpoint(ckpt_path, ckpt_id, model, optimizer, scheduler)
            final_path = os.path.join(output_dir, FINAL_NAME)
            torch.save({'model_state_dict': model.state_dict(), 'update': ckpt_id}, final_path)
            _save_state(output_dir, ckpt_id, kl_ema)
            print(f"Saved: {final_path}")
            return

    ckpt_path = os.path.join(output_dir, f"checkpoint_update_{total_updates}.pt")
    if not os.path.exists(ckpt_path):
        _save_checkpoint(ckpt_path, total_updates, model, optimizer, scheduler)
    final_path = os.path.join(output_dir, FINAL_NAME)
    torch.save({'model_state_dict': model.state_dict(), 'update': total_updates}, final_path)
    _save_state(output_dir, total_updates, kl_ema)
    print(f"\nStage 1 complete. Final: {final_path}")
