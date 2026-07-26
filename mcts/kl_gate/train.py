"""
kl_gate training loop.

Trains KLNet to regress `log(KL + LOG_EPSILON)` under MSE, with 8-fold dihedral
augmentation of every sample. Single pass: no resume state, no per-epoch
checkpoints — only `out/kl_net.pt` at the end, plus one CSV row per epoch.

Run from mcts/:  python3 kl_gate/train.py
"""

import csv
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from config import (
    BATCH_SIZE,
    EPOCHS,
    GRAD_CLIP_NORM,
    LR,
    MIN_LR,
    OUT_DIR,
    SEED,
)
from dataset import KLGateSplit, augment_8fold, load_split
from net import KLNet

from main import DEVICE, WEIGHT_DECAY

CSV_COLUMNS = [
    'epoch', 'train_mse', 'val_mse', 'val_mae', 'val_pearson', 'lr', 'time_train',
]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model: KLNet, split: KLGateSplit) -> dict:
    """Validation metrics in identity orientation only.

    MSE is the headline number (it is the training objective); MAE and the
    Pearson correlation are secondary readouts — a predictor can have a decent
    MSE purely by tracking the mean, and r is what shows whether it separates
    positions at all.
    """
    model.eval()
    preds = np.empty(len(split), dtype=np.float32)
    for start in range(0, len(split), BATCH_SIZE):
        indices = np.arange(start, min(start + BATCH_SIZE, len(split)))
        x, _ = split.make_batch(indices, DEVICE)
        preds[indices] = model(x).float().cpu().numpy()
    model.train()

    err = preds - split.y
    return {
        'val_mse': float(np.mean(err ** 2)),
        'val_mae': float(np.mean(np.abs(err))),
        'val_pearson': float(np.corrcoef(preds, split.y)[0, 1]),
    }


def run() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    seed_everything(SEED)

    print("Loading shards...")
    train_split, val_split = load_split()
    print(f"  train: {len(train_split)} samples | val: {len(val_split)} samples")
    print(f"  target log(KL+eps): train mean {train_split.y.mean():.4f} "
          f"std {train_split.y.std():.4f}")

    # The bar every result must clear: predicting the training mean everywhere.
    baseline_mse = float(np.mean((val_split.y - train_split.y.mean()) ** 2))
    print(f"  constant-predictor baseline val MSE: {baseline_mse:.4f}")

    model = KLNet().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"KLNet: {n_params:,} parameters")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, fused=True
    )
    steps_per_epoch = len(train_split) // BATCH_SIZE
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS * steps_per_epoch, eta_min=MIN_LR
    )
    print(f"Training: {EPOCHS} epochs x {steps_per_epoch} steps "
          f"({BATCH_SIZE} base samples -> {8 * BATCH_SIZE} rows/step), "
          f"LR {LR} -> {MIN_LR} (cosine)")

    csv_path = os.path.join(OUT_DIR, "training_log.csv")
    if not os.path.exists(csv_path):
        with open(csv_path, 'w', newline='') as f:
            csv.writer(f).writerow(CSV_COLUMNS)

    model.train()
    total_start = time.time()
    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        perm = np.random.permutation(len(train_split))
        epoch_loss = 0.0

        for step in range(steps_per_epoch):
            indices = perm[step * BATCH_SIZE:(step + 1) * BATCH_SIZE]
            x, y = train_split.make_batch(indices, DEVICE)
            x = augment_8fold(x)
            y = y.repeat(8)

            loss = F.mse_loss(model(x), y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()

        metrics = {
            'train_mse': epoch_loss / steps_per_epoch,
            'lr': scheduler.get_last_lr()[0],
            'time_train': time.time() - t0,
        }
        metrics.update(evaluate(model, val_split))

        with open(csv_path, 'a', newline='') as f:
            csv.writer(f).writerow([epoch] + [metrics[c] for c in CSV_COLUMNS[1:]])
        print(
            f"Epoch {epoch:3d}/{EPOCHS} | train MSE {metrics['train_mse']:.4f} | "
            f"val MSE {metrics['val_mse']:.4f} MAE {metrics['val_mae']:.4f} "
            f"r {metrics['val_pearson']:.4f} | lr {metrics['lr']:.2e} | "
            f"{metrics['time_train']:.0f}s | total {(time.time()-total_start)/3600:.2f}h"
        )

    out_path = os.path.join(OUT_DIR, "kl_net.pt")
    torch.save({'model_state_dict': model.state_dict(), 'epoch': EPOCHS}, out_path)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
    run()
