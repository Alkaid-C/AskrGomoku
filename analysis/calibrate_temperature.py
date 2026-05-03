"""
Empirically determine INITIAL_TEMPERATURE for MCTS training.

Plays MCTS self-play games at several prior temperatures, measures
H_mcts / H_model on the collected positions, then finds the fixed point
T* where ratio(T*) = T*.

Run: python3 calibrate_temperature.py
"""

import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn.functional as F
from gomoku import LOGIT_MASK_VALUE, RENJU_OPENING_SEQUENCES, SEED_PROBABILITY
from model import GomokuPolicyNet
from self_play import play_mcts_games

DEVICE = torch.device("cuda")
torch.backends.cudnn.conv.fp32_precision = 'tf32'
torch.backends.cuda.matmul.fp32_precision = 'tf32'

# Match the real training config
NUM_SIMULATIONS = 400
C_PUCT = 1.25
DIRICHLET_ALPHA = 0.15
DIRICHLET_EPSILON = 0.25
ACTION_TEMPERATURE = 1.0

NUM_GAMES_PER_T = 64
T_CANDIDATES = [0.9, 1.1, 1.3, 1.5, 1.7]

SEED = 123
CHECKPOINT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "final_policy.pt")


def measure_ratio(model: torch.nn.Module, T: float, n_games: int) -> tuple[float, float, float, int]:
    """Run n_games MCTS self-play games at prior temperature T, return
    (H_model, H_mcts, ratio, n_positions)."""
    current_is_black = [random.random() < 0.5 for _ in range(n_games)]
    n_openings = len(RENJU_OPENING_SEQUENCES)
    opening_ids = [
        random.randint(0, n_openings - 1) if random.random() < SEED_PROBABILITY else -1
        for _ in range(n_games)
    ]
    # Self-play: opponent is the same model
    opponents = [model] * n_games

    model.eval()
    records = play_mcts_games(
        current_model=model,
        opponent_models=opponents,
        current_is_black=current_is_black,
        num_simulations=NUM_SIMULATIONS,
        c_puct=C_PUCT,
        prior_temperature=T,
        device=DEVICE,
        opening_ids=opening_ids,
        dirichlet_alpha=DIRICHLET_ALPHA,
        dirichlet_epsilon=DIRICHLET_EPSILON,
        action_temperature=ACTION_TEMPERATURE,
    )

    all_obs = []
    all_dists = []
    for rec in records:
        for obs, dist in zip(rec.observations, rec.visit_distributions):
            all_obs.append(obs)
            all_dists.append(dist)

    if not all_obs:
        return 0.0, 0.0, 0.0, 0

    obs_t = torch.from_numpy(np.stack(all_obs)).float().to(DEVICE)
    dist_t = torch.from_numpy(np.stack(all_dists)).float().to(DEVICE)

    # Legal mask from observation (empty = legal); obs channels are uint8
    occupied = obs_t[:, 0] + obs_t[:, 1]
    legal_mask = (occupied == 0).view(-1, 225)

    with torch.inference_mode():
        logits, _ = model(obs_t)
    logits = logits.squeeze(1).view(-1, 225)
    logits = logits.masked_fill(~legal_mask, LOGIT_MASK_VALUE)

    log_probs = F.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    model_ent = -(probs * log_probs).sum(dim=-1).mean().item()
    mcts_ent = -(dist_t * (dist_t + 1e-10).log()).sum(dim=-1).mean().item()

    return model_ent, mcts_ent, mcts_ent / model_ent, len(all_obs)


def find_fixed_point(Ts: list[float], ratios: list[float]) -> float:
    """Find T* such that ratio(T*) = T* via linear interpolation of f(T)=ratio(T)-T."""
    f = [r - T for T, r in zip(Ts, ratios)]
    # Look for a sign change
    for i in range(len(Ts) - 1):
        if f[i] == 0:
            return Ts[i]
        if f[i] * f[i + 1] < 0:
            # Linear interpolation between (Ts[i], f[i]) and (Ts[i+1], f[i+1])
            t = Ts[i] - f[i] * (Ts[i + 1] - Ts[i]) / (f[i + 1] - f[i])
            return t
    # No sign change — return the T with smallest |ratio - T|
    idx = min(range(len(Ts)), key=lambda k: abs(f[k]))
    return Ts[idx]


def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    print(f"Loading model: {CHECKPOINT}")
    blob = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False)
    state_dict = blob['model_state_dict'] if isinstance(blob, dict) and 'model_state_dict' in blob else blob
    model = GomokuPolicyNet().to(DEVICE)
    model.load_state_dict(state_dict)
    model.eval()

    print(f"NUM_SIMULATIONS={NUM_SIMULATIONS}, games per T={NUM_GAMES_PER_T}")
    print(f"T candidates: {T_CANDIDATES}")
    print()
    print(f"{'T':>6} {'H_model':>9} {'H_mcts':>9} {'ratio':>9} {'n_pos':>7} {'time':>7}")
    print("-" * 55)

    ratios = []
    for T in T_CANDIDATES:
        t0 = time.time()
        h_model, h_mcts, ratio, n_pos = measure_ratio(model, T, NUM_GAMES_PER_T)
        dt = time.time() - t0
        print(f"{T:6.3f} {h_model:9.4f} {h_mcts:9.4f} {ratio:9.4f} {n_pos:7d} {dt:6.1f}s")
        ratios.append(ratio)

    print()
    T_star = find_fixed_point(T_CANDIDATES, ratios)
    print(f"Fixed point (ratio(T*) = T*): T* ≈ {T_star:.3f}")
    print()
    print("Current INITIAL_TEMPERATURE = 1.72")
    print(f"Empirical recommendation: INITIAL_TEMPERATURE = {T_star:.2f}")


if __name__ == "__main__":
    main()
