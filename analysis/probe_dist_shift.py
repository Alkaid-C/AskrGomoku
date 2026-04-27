"""
Distinguish model-change vs distribution-shift as cause of the entropy jump
between update 1 (H≈1.4) and update 2 (H≈1.7).

Procedure:
  1. Load final_policy.pt; self-play batch_1 at T=1.28; record positions.
  2. Measure H_model on batch_1 BEFORE the gradient step  (≈ update 1's logged H).
  3. Run ONE gradient step on batch_1 (CE + MSE; same as training.py).
  4. Measure H_model on batch_1 AFTER the step (same positions, new weights).
  5. Self-play batch_2 with the updated model at the same T;
     measure H_model on batch_2 (≈ update 2's logged H).

Interpretation:
  - If H_after_same_batch ≈ H_before, but H_after_new_batch >> H_before:
        the jump is *distribution shift* — flatter prior leads MCTS to
        sample positions where the model is more uncertain.
  - If H_after_same_batch ≈ H_after_new_batch >> H_before:
        the jump is genuine model change in one step.
  - In between: both contribute.
"""

import os
import sys

sys.path.insert(0, '/data/Gomoku/vibe2/mcts_post_train')
os.chdir('/data/Gomoku/vibe2/mcts_post_train')
os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'

import random as _r

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
INITIAL_TEMPERATURE = 1.28
TEMP_CONVERGENCE_EXPONENT = 0.99
NUM_SIMULATIONS = 1024
C_PUCT = 1.25
DIRICHLET_ALPHA = 0.15
DIRICHLET_EPSILON = 0.25
DISCOUNT_GAMMA = 0.98
EPISODES_PER_UPDATE = 96


def sample_opening_ids(n: int) -> list[int]:
    out = []
    for _ in range(n):
        if _r.random() < SEED_PROBABILITY:
            out.append(_r.randint(0, len(RENJU_OPENING_SEQUENCES) - 1))
        else:
            out.append(-1)
    return out


def collect_batch(model, T: float):
    """Self-play EPISODES_PER_UPDATE games; return augmented tensors."""
    model.eval()
    opening_ids = sample_opening_ids(EPISODES_PER_UPDATE)
    records = play_mcts_games(
        model=model, num_games=EPISODES_PER_UPDATE,
        num_simulations=NUM_SIMULATIONS, c_puct=C_PUCT,
        entropy_multiplier=T, device=device, opening_ids=opening_ids,
        dirichlet_alpha=DIRICHLET_ALPHA, dirichlet_epsilon=DIRICHLET_EPSILON,
        gamma=DISCOUNT_GAMMA,
    )
    clear_nn_eval_cache()

    obs_l, dist_l, val_l, mask_l = [], [], [], []
    for record in records:
        for obs, dist, val in zip(record.observations, record.visit_distributions, record.root_values):
            obs_l.append(obs); dist_l.append(dist); val_l.append(val)
            mask_l.append((1 - (obs[0] | obs[1])).astype(np.uint8))

    obs_t = torch.from_numpy(np.stack(obs_l)).float().to(device)
    dist_t = torch.from_numpy(np.stack(dist_l)).float().to(device)
    mask_t = torch.from_numpy(np.stack(mask_l)).bool().to(device)
    val_t = torch.tensor(val_l, dtype=torch.float32, device=device)
    obs_t, dist_t, mask_t, val_t = augment_mcts_batch_8fold(obs_t, dist_t, mask_t, val_t)
    return obs_t, dist_t, mask_t, val_t


def measure(model, obs_t, dist_t, mask_t, entropy_divisor):
    model.eval()
    n_aug = obs_t.shape[0]
    Hm, Ht, KL, n = 0.0, 0.0, 0.0, 0
    for s in range(0, n_aug, TRAIN_BATCH_SIZE):
        e = min(s + TRAIN_BATCH_SIZE, n_aug)
        mb_o = obs_t[s:e]; mb_m = mask_t[s:e]; mb_d = dist_t[s:e]; sz = e - s
        with torch.no_grad():
            lg, _ = model(mb_o)
            lg = lg.squeeze(1).view(sz, 225).masked_fill(~mb_m.view(sz, 225), LOGIT_MASK_VALUE)
            lp = F.log_softmax(lg, dim=-1)
            pp = F.softmax(lg, dim=-1)
            Hm += -(pp * lp).sum(-1).mean().item() * sz
            log_d = mb_d.clamp(min=1e-30).log()
            tH = -(mb_d * log_d).sum(-1) / entropy_divisor
            sh = rescale_to_entropy(log_d, tH)
            Ht += -(sh * (sh + 1e-10).log()).sum(-1).mean().item() * sz
            KL += (sh * ((sh + 1e-10).log() - lp)).sum(-1).mean().item() * sz
        n += sz
    return Hm / n, Ht / n, KL / n


def train_one_step(model, optimizer, obs_t, dist_t, mask_t, val_t, entropy_divisor):
    n_aug = obs_t.shape[0]
    perm = torch.randperm(n_aug, device=device)
    obs_t = obs_t[perm]; dist_t = dist_t[perm]; mask_t = mask_t[perm]; val_t = val_t[perm]

    model.train()
    optimizer.zero_grad()
    total_pl = 0.0
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
        total_pl += pl.item() * sz

    g2 = sum(p.grad.detach().pow(2).sum().item() for p in model.parameters() if p.grad is not None)
    grad_norm = g2 ** 0.5
    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
    optimizer.step()
    return total_pl / n_aug, grad_norm


def main():
    sys.stdout.reconfigure(line_buffering=True)
    torch.manual_seed(42); np.random.seed(42); _r.seed(42)

    ckpt = torch.load('final_policy.pt', map_location=device, weights_only=False)
    model = GomokuPolicyNet().to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                                   weight_decay=WEIGHT_DECAY, fused=True)

    T = INITIAL_TEMPERATURE
    entropy_divisor = T ** TEMP_CONVERGENCE_EXPONENT
    print(f'T={T}  entropy_divisor=T^{TEMP_CONVERGENCE_EXPONENT}={entropy_divisor:.4f}')
    print(f'games={EPISODES_PER_UPDATE}, sims={NUM_SIMULATIONS}')

    print('\n[1] self-play batch_1 with checkpoint model...')
    obs1, dist1, mask1, val1 = collect_batch(model, T)
    print(f'    batch_1 augmented samples: {obs1.shape[0]}')

    Hm_pre, Ht_pre, KL_pre = measure(model, obs1, dist1, mask1, entropy_divisor)
    print(f'    BEFORE step:    H_model={Hm_pre:.4f}  H_target={Ht_pre:.4f}  KL(t||m)={KL_pre:.4f}')

    print('\n[2] one gradient step on batch_1...')
    pl, gn = train_one_step(model, optimizer, obs1, dist1, mask1, val1, entropy_divisor)
    print(f'    policy_loss={pl:.4f}  grad_norm(unclipped)={gn:.4f}  (clip={GRAD_CLIP_NORM})')

    Hm_same, Ht_same, KL_same = measure(model, obs1, dist1, mask1, entropy_divisor)
    print(f'    AFTER step on batch_1:    H_model={Hm_same:.4f}  H_target={Ht_same:.4f}  KL(t||m)={KL_same:.4f}')

    print('\n[3] self-play batch_2 with updated model (same T)...')
    obs2, dist2, mask2, _ = collect_batch(model, T)
    print(f'    batch_2 augmented samples: {obs2.shape[0]}')

    Hm_new, Ht_new, KL_new = measure(model, obs2, dist2, mask2, entropy_divisor)
    print(f'    AFTER step on batch_2:    H_model={Hm_new:.4f}  H_target={Ht_new:.4f}  KL(t||m)={KL_new:.4f}')

    print('\n=== summary ===')
    print(f'  H_model on batch_1, pre-step  : {Hm_pre:.4f}   (≈ update 1 logged H)')
    print(f'  H_model on batch_1, post-step : {Hm_same:.4f}   (Δ from pre = {Hm_same - Hm_pre:+.4f})')
    print(f'  H_model on batch_2, post-step : {Hm_new:.4f}   (≈ update 2 logged H; Δ from pre = {Hm_new - Hm_pre:+.4f})')
    print(f'  attributable to model change   : {Hm_same - Hm_pre:+.4f}')
    print(f'  attributable to dist shift     : {Hm_new - Hm_same:+.4f}')


if __name__ == '__main__':
    main()
