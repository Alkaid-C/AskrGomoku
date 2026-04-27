"""
Probe one MCTS post-train step to understand rapid model entropy growth.

Loads final_policy.pt, runs self-play + a single training step, then reports:
- Per-position entropies before/after the step
- KL(target || model) before/after
- Gradient norms
- Where the model's mass actually moved
"""

import os
import sys

sys.path.insert(0, '/data/Gomoku/vibe2/mcts_post_train')
os.chdir('/data/Gomoku/vibe2/mcts_post_train')
os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'

import math

import numpy as np
import torch
import torch.nn.functional as F

from entropy_ops import rescale_to_entropy
from gomoku import LOGIT_MASK_VALUE, RENJU_OPENING_SEQUENCES, SEED_PROBABILITY
from mcts import clear_nn_eval_cache
from model import GomokuPolicyNet
from self_play import play_mcts_games
from training import (
    GRAD_CLIP_NORM,
    TRAIN_BATCH_SIZE,
    VALUE_LOSS_COEFF,
    augment_mcts_batch_8fold,
)

device = torch.device('cuda')
torch.backends.cudnn.conv.fp32_precision = 'tf32'
torch.backends.cuda.matmul.fp32_precision = 'tf32'

LEARNING_RATE = 0.5 / 8192
WEIGHT_DECAY = 1.0 / 2 ** 24
INITIAL_TEMPERATURE = 1.4
TEMP_CONVERGENCE_EXPONENT = 0.99
NUM_SIMULATIONS = 1024
C_PUCT = 1.25
DIRICHLET_ALPHA = 0.15
DIRICHLET_EPSILON = 0.25
DISCOUNT_GAMMA = 0.98
EPISODES_PER_UPDATE = 16  # smaller for speed

torch.manual_seed(42)
np.random.seed(42)
import random as _r
_r.seed(42)

ckpt = torch.load('final_policy.pt', map_location=device, weights_only=False)
model = GomokuPolicyNet().to(device)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, fused=True)

opening_ids = []
for _ in range(EPISODES_PER_UPDATE):
    if _r.random() < SEED_PROBABILITY:
        opening_ids.append(_r.randint(0, len(RENJU_OPENING_SEQUENCES) - 1))
    else:
        opening_ids.append(-1)

print('--- self-play ---')
records = play_mcts_games(
    model=model,
    num_games=EPISODES_PER_UPDATE,
    num_simulations=NUM_SIMULATIONS,
    c_puct=C_PUCT,
    entropy_multiplier=INITIAL_TEMPERATURE,
    device=device,
    opening_ids=opening_ids,
    dirichlet_alpha=DIRICHLET_ALPHA,
    dirichlet_epsilon=DIRICHLET_EPSILON,
    gamma=DISCOUNT_GAMMA,
)

# Collect samples
all_obs = []
all_dists = []
all_values = []
all_masks = []
for record in records:
    for obs, dist, val in zip(record.observations, record.visit_distributions, record.root_values):
        all_obs.append(obs)
        all_dists.append(dist)
        all_values.append(val)
        occupied = obs[0] | obs[1]
        legal_mask = (1 - occupied).astype(np.uint8)
        all_masks.append(legal_mask)

print(f'Total positions: {len(all_obs)}')

obs_tensor = torch.from_numpy(np.stack(all_obs)).float().to(device)
dist_tensor = torch.from_numpy(np.stack(all_dists)).float().to(device)
mask_tensor = torch.from_numpy(np.stack(all_masks)).bool().to(device)
value_tensor = torch.tensor(all_values, dtype=torch.float32, device=device)

# Augment 8x
obs_tensor, dist_tensor, mask_tensor, value_tensor = augment_mcts_batch_8fold(
    obs_tensor, dist_tensor, mask_tensor, value_tensor
)
n_aug = obs_tensor.shape[0]
perm = torch.randperm(n_aug, device=device)
obs_tensor = obs_tensor[perm]
dist_tensor = dist_tensor[perm]
mask_tensor = mask_tensor[perm]
value_tensor = value_tensor[perm]
print(f'Augmented total: {n_aug}')

entropy_divisor = INITIAL_TEMPERATURE ** TEMP_CONVERGENCE_EXPONENT
print(f'entropy_divisor = {entropy_divisor:.4f}')

# === BEFORE step: forward all data and compute model entropy + KL ===
def measure_model_stats(model, obs_tensor, mask_tensor, dist_tensor, entropy_divisor):
    model.eval()
    H_model_total = 0.0
    H_target_total = 0.0
    KL_total = 0.0
    n_total = 0
    for start in range(0, n_aug, TRAIN_BATCH_SIZE):
        end = min(start + TRAIN_BATCH_SIZE, n_aug)
        mb_obs = obs_tensor[start:end]
        mb_mask = mask_tensor[start:end]
        mb_dist = dist_tensor[start:end]
        mb_size = end - start
        with torch.no_grad():
            logits, _ = model(mb_obs)
            logits = logits.squeeze(1).view(mb_size, 225)
            logits = logits.masked_fill(~mb_mask.view(mb_size, 225), LOGIT_MASK_VALUE)
            log_probs = F.log_softmax(logits, dim=-1)
            probs = F.softmax(logits, dim=-1)
            H_model = -(probs * log_probs).sum(dim=-1).mean().item()
            log_dist = mb_dist.clamp(min=1e-30).log()
            H_visit = -(mb_dist * log_dist).sum(dim=-1)
            target_H = H_visit / entropy_divisor
            sharpened = rescale_to_entropy(log_dist, target_H)
            H_target = -(sharpened * (sharpened + 1e-10).log()).sum(dim=-1).mean().item()
            # KL(target || model)
            log_t = (sharpened + 1e-10).log()
            kl = (sharpened * (log_t - log_probs)).sum(dim=-1).mean().item()
        H_model_total += H_model * mb_size
        H_target_total += H_target * mb_size
        KL_total += kl * mb_size
        n_total += mb_size
    return H_model_total / n_total, H_target_total / n_total, KL_total / n_total

H_m_before, H_t_before, KL_before = measure_model_stats(model, obs_tensor, mask_tensor, dist_tensor, entropy_divisor)
print(f'\nBEFORE step: H_model={H_m_before:.4f}  H_target={H_t_before:.4f}  KL(t||m)={KL_before:.4f}')

# === Single training step ===
model.train()
optimizer.zero_grad()
total_policy_loss = 0.0
for start in range(0, n_aug, TRAIN_BATCH_SIZE):
    end = min(start + TRAIN_BATCH_SIZE, n_aug)
    mb_obs = obs_tensor[start:end]
    mb_dist = dist_tensor[start:end]
    mb_mask = mask_tensor[start:end]
    mb_values = value_tensor[start:end]
    mb_size = end - start

    log_dist = mb_dist.clamp(min=1e-30).log()
    H_visit = -(mb_dist * log_dist).sum(dim=-1)
    target_H = H_visit / entropy_divisor
    sharpened = rescale_to_entropy(log_dist, target_H)

    logits, pred_values = model(mb_obs)
    logits = logits.squeeze(1).view(mb_size, 225)
    pred_values = pred_values.squeeze(-1)
    logits = logits.masked_fill(~mb_mask.view(mb_size, 225), LOGIT_MASK_VALUE)
    log_probs = F.log_softmax(logits, dim=-1)
    policy_loss = -(sharpened * log_probs).sum(dim=-1).mean()
    value_loss = F.mse_loss(pred_values, mb_values)
    loss = (policy_loss + VALUE_LOSS_COEFF * value_loss) * (mb_size / n_aug)
    loss.backward()
    total_policy_loss += policy_loss.item() * mb_size

# inspect gradient norm (un-clipped)
total_grad_sq = 0.0
for p in model.parameters():
    if p.grad is not None:
        total_grad_sq += p.grad.detach().pow(2).sum().item()
grad_norm = total_grad_sq ** 0.5
print(f'\nGrad norm (un-clipped): {grad_norm:.4f}  (clip={GRAD_CLIP_NORM})')
torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
optimizer.step()
print(f'Avg policy_loss during step: {total_policy_loss / n_aug:.4f}')

# === AFTER step ===
H_m_after, H_t_after, KL_after = measure_model_stats(model, obs_tensor, mask_tensor, dist_tensor, entropy_divisor)
print(f'\nAFTER step:  H_model={H_m_after:.4f}  H_target={H_t_after:.4f}  KL(t||m)={KL_after:.4f}')

print(f'\nΔ H_model = {H_m_after - H_m_before:+.4f}')
print(f'Δ KL      = {KL_after - KL_before:+.4f}')
