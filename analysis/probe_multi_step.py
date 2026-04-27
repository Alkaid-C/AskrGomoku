"""
Step on the SAME batch multiple times to see whether the model entropy
eventually settles at H_target or stays above. If it settles, the rise
is just transient (mass redistribution); if not, there's a structural
issue.
"""

import os
import sys

sys.path.insert(0, '/data/Gomoku/vibe2/mcts_post_train')
os.chdir('/data/Gomoku/vibe2/mcts_post_train')
os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'

import numpy as np
import torch
import torch.nn.functional as F

from entropy_ops import rescale_to_entropy
from gomoku import LOGIT_MASK_VALUE, RENJU_OPENING_SEQUENCES, SEED_PROBABILITY
from model import GomokuPolicyNet
from self_play import play_mcts_games
from training import (
    GRAD_CLIP_NORM,
    TRAIN_BATCH_SIZE,
    VALUE_LOSS_COEFF,
    augment_mcts_batch_8fold,
)

device = torch.device('cuda')

LEARNING_RATE = 0.5 / 8192
WEIGHT_DECAY = 1.0 / 2 ** 24
INITIAL_TEMPERATURE = 1.4
TEMP_CONVERGENCE_EXPONENT = 0.99
NUM_SIMULATIONS = 1024
C_PUCT = 1.25
DIRICHLET_ALPHA = 0.15
DIRICHLET_EPSILON = 0.25
DISCOUNT_GAMMA = 0.98
EPISODES_PER_UPDATE = 16

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

records = play_mcts_games(
    model=model, num_games=EPISODES_PER_UPDATE, num_simulations=NUM_SIMULATIONS,
    c_puct=C_PUCT, entropy_multiplier=INITIAL_TEMPERATURE, device=device,
    opening_ids=opening_ids, dirichlet_alpha=DIRICHLET_ALPHA,
    dirichlet_epsilon=DIRICHLET_EPSILON, gamma=DISCOUNT_GAMMA,
)

all_obs, all_dists, all_values, all_masks = [], [], [], []
for record in records:
    for obs, dist, val in zip(record.observations, record.visit_distributions, record.root_values):
        all_obs.append(obs); all_dists.append(dist); all_values.append(val)
        all_masks.append((1 - (obs[0] | obs[1])).astype(np.uint8))

obs_t = torch.from_numpy(np.stack(all_obs)).float().to(device)
dist_t = torch.from_numpy(np.stack(all_dists)).float().to(device)
mask_t = torch.from_numpy(np.stack(all_masks)).bool().to(device)
val_t = torch.tensor(all_values, dtype=torch.float32, device=device)
obs_t, dist_t, mask_t, val_t = augment_mcts_batch_8fold(obs_t, dist_t, mask_t, val_t)
n_aug = obs_t.shape[0]
print(f'samples: {n_aug}')

entropy_divisor = INITIAL_TEMPERATURE ** TEMP_CONVERGENCE_EXPONENT


def measure(model):
    model.eval()
    Hm, Ht, KL = 0.0, 0.0, 0.0
    n = 0
    for s in range(0, n_aug, TRAIN_BATCH_SIZE):
        e = min(s + TRAIN_BATCH_SIZE, n_aug)
        mb_o = obs_t[s:e]; mb_m = mask_t[s:e]; mb_d = dist_t[s:e]; sz = e - s
        with torch.no_grad():
            lg, _ = model(mb_o)
            lg = lg.squeeze(1).view(sz, 225).masked_fill(~mb_m.view(sz, 225), LOGIT_MASK_VALUE)
            lp = F.log_softmax(lg, dim=-1); pp = F.softmax(lg, dim=-1)
            Hm += -(pp * lp).sum(-1).mean().item() * sz
            log_d = mb_d.clamp(min=1e-30).log()
            tH = -(mb_d * log_d).sum(-1) / entropy_divisor
            sh = rescale_to_entropy(log_d, tH)
            Ht += -(sh * (sh + 1e-10).log()).sum(-1).mean().item() * sz
            KL += (sh * ((sh + 1e-10).log() - lp)).sum(-1).mean().item() * sz
        n += sz
    return Hm / n, Ht / n, KL / n


def step(model):
    model.train()
    optimizer.zero_grad()
    for s in range(0, n_aug, TRAIN_BATCH_SIZE):
        e = min(s + TRAIN_BATCH_SIZE, n_aug)
        mb_o = obs_t[s:e]; mb_d = dist_t[s:e]; mb_m = mask_t[s:e]; mb_v = val_t[s:e]; sz = e - s
        log_d = mb_d.clamp(min=1e-30).log()
        tH = -(mb_d * log_d).sum(-1) / entropy_divisor
        sh = rescale_to_entropy(log_d, tH)
        lg, pv = model(mb_o)
        lg = lg.squeeze(1).view(sz, 225).masked_fill(~mb_m.view(sz, 225), LOGIT_MASK_VALUE)
        pv = pv.squeeze(-1)
        lp = F.log_softmax(lg, dim=-1)
        pl = -(sh * lp).sum(-1).mean()
        vl = F.mse_loss(pv, mb_v)
        loss = (pl + VALUE_LOSS_COEFF * vl) * (sz / n_aug)
        loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
    optimizer.step()


print(f'\n{"step":>4} | {"H_model":>8} {"H_target":>8} {"KL":>8}')
Hm, Ht, KL = measure(model)
print(f'{0:>4} | {Hm:>8.4f} {Ht:>8.4f} {KL:>8.4f}')
for i in range(1, 31):
    step(model)
    Hm, Ht, KL = measure(model)
    print(f'{i:>4} | {Hm:>8.4f} {Ht:>8.4f} {KL:>8.4f}')
