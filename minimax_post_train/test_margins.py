"""
Test script to determine appropriate m_rank and m_sep values for post-training.

This script:
1. Loads the RL checkpoint
2. Runs self-play to generate sample positions
3. For each position, generates candidates (top-5 + 1 random neighbor)
4. Analyzes logit distributions to recommend margin values
"""

import random
from typing import List, Tuple

import numpy as np
import torch
from gomoku import LOGIT_MASK_VALUE, GomokuBoard, encode_observation, get_local_candidate_moves, idx_to_pos
from model import GomokuPolicyNet


def generate_test_positions(model, device: torch.device, num_games: int = 10) -> List[Tuple[np.ndarray, np.ndarray, int]]:
    """
    Generate test positions by running self-play games.

    Returns:
        List of (observation, legal_mask, move_count) tuples
    """
    positions = []
    model.eval()

    for game_idx in range(num_games):
        board = GomokuBoard(opening_id=-1)  # Start from empty board
        move_count = 0

        while True:
            legal_mask, _ = board.GetLegalMoves()
            c0, c1, _ = board.GetBoardState()
            obs = encode_observation(c0, c1)

            # Store position if at least 10 moves have been played
            if move_count >= 10:
                positions.append((obs.copy(), legal_mask.copy(), move_count))

            # Make a move
            with torch.no_grad():
                obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(device)
                logits_grid = model.forward_policy_only(obs_tensor)
                logits = logits_grid.squeeze()

                # Mask illegal moves
                mask_tensor = torch.from_numpy(legal_mask).bool().to(device)
                logits_flat = logits.view(225).masked_fill(~mask_tensor.view(225), LOGIT_MASK_VALUE)

                # Sample with temperature
                probs = torch.softmax(logits_flat, dim=0)
                action = torch.multinomial(probs, 1).item()

            # Execute move
            row, col = idx_to_pos(action)
            outcome = board.Move((row, col))
            move_count += 1

            # Stop if game ends or too many moves
            if outcome.value != 0 or move_count >= 100:
                break

        if game_idx % 5 == 0:
            print(f"Generated {game_idx + 1}/{num_games} games, collected {len(positions)} positions")

    return positions


def analyze_position(model, device: torch.device, obs: np.ndarray, legal_mask: np.ndarray) -> dict:
    """
    Analyze a single position: generate candidates and extract logit statistics.

    Returns:
        dict with candidate logits, non-candidate logits, and statistics
    """
    model.eval()

    with torch.no_grad():
        obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(device)
        logits_grid = model.forward_policy_only(obs_tensor)
        logits = logits_grid.squeeze()

        # Get flat logits for legal moves
        mask_tensor = torch.from_numpy(legal_mask).bool().to(device)
        logits_flat = logits.view(225).masked_fill(~mask_tensor.view(225), LOGIT_MASK_VALUE)

        # Get top-5 candidates
        legal_indices = torch.where(mask_tensor.view(225))[0].cpu().numpy()
        legal_logits = logits_flat[legal_indices].cpu().numpy()
        sorted_indices = np.argsort(legal_logits)[::-1]  # Descending order

        top5_legal_idx = sorted_indices[:5]
        top5_actions = legal_indices[top5_legal_idx]
        top5_logits = legal_logits[top5_legal_idx]

        # Generate random neighbor candidate (c6)
        local_candidates = get_local_candidate_moves(obs, legal_mask, radius=1)
        if len(local_candidates) > 0:
            c6_action = random.choice(local_candidates)
        else:
            # Fallback: random legal move
            c6_action = random.choice(legal_indices.tolist())

        c6_logit = logits_flat[c6_action].cpu().item()

        # All candidates (top-5 + c6)
        all_candidates = set(top5_actions.tolist() + [c6_action])

        # Non-candidate logits
        non_candidate_actions = [a for a in legal_indices if a not in all_candidates]
        if len(non_candidate_actions) > 0:
            non_candidate_logits = logits_flat[non_candidate_actions].cpu().numpy()
        else:
            non_candidate_logits = np.array([])

        return {
            'top5_actions': top5_actions,
            'top5_logits': top5_logits,
            'c6_action': c6_action,
            'c6_logit': c6_logit,
            'non_candidate_logits': non_candidate_logits,
            'num_legal': len(legal_indices),
        }


def compute_statistics(all_results: List[dict]) -> dict:
    """
    Compute aggregate statistics across all positions.
    """
    # Collect pairwise differences within top-4
    delta_12 = []  # L(c1) - L(c2)
    delta_23 = []  # L(c2) - L(c3)
    delta_34 = []  # L(c3) - L(c4)

    # Collect separation margins: L(c4) - L(non_candidate)
    separation_margins = []

    for result in all_results:
        logits = result['top5_logits']

        # Top-4 pairwise differences
        if len(logits) >= 4:
            delta_12.append(logits[0] - logits[1])
            delta_23.append(logits[1] - logits[2])
            delta_34.append(logits[2] - logits[3])

            # Separation: c4 vs all non-candidates
            c4_logit = logits[3]
            non_cand_logits = result['non_candidate_logits']
            if len(non_cand_logits) > 0:
                for nc_logit in non_cand_logits:
                    separation_margins.append(c4_logit - nc_logit)

    delta_12 = np.array(delta_12)
    delta_23 = np.array(delta_23)
    delta_34 = np.array(delta_34)
    separation_margins = np.array(separation_margins)

    stats = {
        'delta_12': {
            'mean': np.mean(delta_12),
            'median': np.median(delta_12),
            'std': np.std(delta_12),
            'min': np.min(delta_12),
            'p10': np.percentile(delta_12, 10),
            'p25': np.percentile(delta_12, 25),
        },
        'delta_23': {
            'mean': np.mean(delta_23),
            'median': np.median(delta_23),
            'std': np.std(delta_23),
            'min': np.min(delta_23),
            'p10': np.percentile(delta_23, 10),
            'p25': np.percentile(delta_23, 25),
        },
        'delta_34': {
            'mean': np.mean(delta_34),
            'median': np.median(delta_34),
            'std': np.std(delta_34),
            'min': np.min(delta_34),
            'p10': np.percentile(delta_34, 10),
            'p25': np.percentile(delta_34, 25),
        },
        'separation': {
            'mean': np.mean(separation_margins),
            'median': np.median(separation_margins),
            'std': np.std(separation_margins),
            'min': np.min(separation_margins),
            'p10': np.percentile(separation_margins, 10),
            'p25': np.percentile(separation_margins, 25),
        },
    }

    return stats


def recommend_margins(stats: dict) -> Tuple[float, float]:
    """
    Recommend m_rank and m_sep based on statistics.

    Strategy:
    - m_rank: Should be smaller than typical gaps to allow model flexibility,
              but large enough to penalize violations. Use ~10-25th percentile.
    - m_sep: Should be smaller than typical c4-vs-non_candidate gaps,
             targeting the lower percentiles to catch close calls.
    """
    # For m_rank, look at the smallest typical gap among (delta_12, delta_23, delta_34)
    # We want a value that's conservative (not too large) but meaningful
    min_p10 = min(
        stats['delta_12']['p10'],
        stats['delta_23']['p10'],
        stats['delta_34']['p10']
    )
    min_p25 = min(
        stats['delta_12']['p25'],
        stats['delta_23']['p25'],
        stats['delta_34']['p25']
    )

    # Recommend m_rank as somewhere between p10 and p25
    m_rank_low = max(0.01, min_p10 * 0.8)
    m_rank_high = max(0.01, min_p25)
    m_rank_suggested = (m_rank_low + m_rank_high) / 2

    # For m_sep, use the separation statistics
    # We want to prevent non-candidates from getting too close to c4
    sep_p10 = stats['separation']['p10']
    sep_p25 = stats['separation']['p25']

    m_sep_low = max(0.01, sep_p10 * 0.5)
    m_sep_high = max(0.01, sep_p25 * 0.8)
    m_sep_suggested = (m_sep_low + m_sep_high) / 2

    return m_rank_suggested, m_sep_suggested


def main():
    print("=" * 80)
    print("Margin Value Testing for Post-Training")
    print("=" * 80)

    # Set device (force CPU)
    device = torch.device('cpu')
    print(f"\nUsing device: {device}")

    # Load checkpoint
    checkpoint_path = 'rl.pt'
    print(f"Loading checkpoint: {checkpoint_path}")

    model = GomokuPolicyNet().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("Checkpoint loaded successfully")

    # Generate test positions
    print("\n" + "=" * 80)
    print("Generating test positions via self-play...")
    print("=" * 80)
    num_games = 100
    positions = generate_test_positions(model, device, num_games=num_games)
    print(f"\nCollected {len(positions)} positions from {num_games} games")

    # Analyze positions
    print("\n" + "=" * 80)
    print("Analyzing candidate generation and logit distributions...")
    print("=" * 80)

    all_results = []
    for i, (obs, legal_mask, _move_count) in enumerate(positions):
        result = analyze_position(model, device, obs, legal_mask)
        all_results.append(result)

        if i % 20 == 0:
            print(f"Analyzed {i + 1}/{len(positions)} positions")

    print(f"Analyzed all {len(positions)} positions")

    # Compute statistics
    print("\n" + "=" * 80)
    print("Computing statistics...")
    print("=" * 80)

    stats = compute_statistics(all_results)

    # Print results
    print("\n" + "=" * 80)
    print("LOGIT GAP STATISTICS")
    print("=" * 80)

    print("\n1. Top candidate pairwise gaps (for m_rank):")
    print("-" * 60)
    for gap_name in ['delta_12', 'delta_23', 'delta_34']:
        s = stats[gap_name]
        label = gap_name.replace('delta_', 'L(c') + ')'
        label = label.replace('12', '1) - L(c2')
        label = label.replace('23', '2) - L(c3')
        label = label.replace('34', '3) - L(c4')

        print(f"\n{label}:")
        print(f"  Mean:   {s['mean']:>8.4f}")
        print(f"  Median: {s['median']:>8.4f}")
        print(f"  Std:    {s['std']:>8.4f}")
        print(f"  Min:    {s['min']:>8.4f}")
        print(f"  10th %: {s['p10']:>8.4f}")
        print(f"  25th %: {s['p25']:>8.4f}")

    print("\n2. Separation gaps: L(c4) - L(non_candidate) (for m_sep):")
    print("-" * 60)
    s = stats['separation']
    print(f"  Mean:   {s['mean']:>8.4f}")
    print(f"  Median: {s['median']:>8.4f}")
    print(f"  Std:    {s['std']:>8.4f}")
    print(f"  Min:    {s['min']:>8.4f}")
    print(f"  10th %: {s['p10']:>8.4f}")
    print(f"  25th %: {s['p25']:>8.4f}")

    # Recommendations
    print("\n" + "=" * 80)
    print("MARGIN RECOMMENDATIONS")
    print("=" * 80)

    m_rank, m_sep = recommend_margins(stats)

    print(f"\nRecommended m_rank: {m_rank:.4f}")
    print("  Rationale: Should be smaller than typical inter-candidate gaps")
    print("             to allow flexibility, but enforce ordering when gaps")
    print("             are small. Targets 10th-25th percentile range.")

    print(f"\nRecommended m_sep:  {m_sep:.4f}")
    print("  Rationale: Should prevent non-candidates from getting close to c4.")
    print("             Targets lower percentile to catch marginal cases.")

    print("\n" + "=" * 80)
    print("INTERPRETATION GUIDE")
    print("=" * 80)
    print("""
The margin values control the ranking loss:

1. m_rank (Ranking-Inside margin):
   - Used in: ReLU(L(c_{i+1}) - L(c_i) + m_rank)
   - Penalizes violations when L(c_{i+1}) > L(c_i) - m_rank
   - Smaller m_rank = stricter ordering requirement
   - Larger m_rank = more tolerance for small inversions

2. m_sep (Separation-Outside margin):
   - Used in: ReLU(L(n) - L(c4) + m_sep)
   - Penalizes when non-candidate n gets within m_sep of c4
   - Smaller m_sep = allows non-candidates closer to boundary
   - Larger m_sep = enforces stronger separation

Typical tuning strategy:
- Start with recommended values
- If training shows many ranking violations, decrease margins
- If training is too rigid and loss plateaus, increase margins
- Monitor policy head gradient norms during training

Note: These are starting points. Fine-tune based on training dynamics.
""")

    print("=" * 80)
    print("Testing complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
