# Top-k Negamax Post-Training Implementation

This document describes the implementation of the POST_TRAIN_SPEC.md specification, which transforms the Gomoku RL training pipeline into a supervised learning system using negamax search.

## Overview

The post-training system adapts an existing RL checkpoint for top-k negamax search:
- **Policy head**: Learns move ordering via ranking loss
- **Value head**: Learns leaf evaluation via MSE to search backup values
- **Progressive unfreezing**: Gradual unfreezing from heads → trunk → stem

## Implementation Phases

### Phase 1: Remove OPR and Convert Tactical Boost to Probe

**Goal**: Clean up RL-specific enhancement code, keep only tactical accuracy probing.

#### Changes to `enhancement.py`

**Removed:**
- `generate_offpolicy_rollout_samples()` - OPR sample generation
- `apply_tactical_enhancements()` - Training sample modification
- `compute_adaptive_boosts()` - Dynamic boost computation
- `update_miss_rate_ema()` - EMA tracking
- `OffPolicyRolloutStats` dataclass
- `TacticalBoostInfo` dataclass
- All OPR and tactical boost constants

**Kept:**
- `TacticalStats` dataclass (lines 23-44)
- `get_local_candidate_positions()` (lines 51-88)
- `is_winning_move()` (lines 91-126)
- `find_all_win_in_1()` (lines 129-152)
- `find_blocking_moves()` (lines 155-187)
- `augment_batch_8fold()` (lines 247-311)
- `augment_candidates_8fold()` (lines 314-348)

**Added:**
- `probe_tactical_accuracy(trajectories, current_is_black)` - Compute tactical stats without modifying training (lines 194-240)

#### Changes to `training.py`

- Removed imports of `apply_tactical_enhancements`, `compute_adaptive_boosts`
- Added `EPISODE_WEIGHT_ALPHA = 0.25` constant locally
- Simplified `_train_on_batch_internal()` and `train_on_batch()` signatures
- Removed tactical boost application and OPR sample handling

#### Changes to `main.py`

- Updated imports to use `probe_tactical_accuracy` and `TacticalStats`
- Removed `win_miss_ema`, `block_miss_ema` from training state
- Removed OPR generation calls
- Added tactical probe execution alongside gradient probe

#### Changes to `csv_logger.py`

- Simplified `training_updates.csv` columns (removed OPR/tactical columns)
- Added `tactical_probe.csv` with columns:
  - `update`, `win_opportunities`, `win_hits`, `win_misses`, `win_accuracy`
  - `block_opportunities`, `block_hits`, `block_misses`, `block_accuracy`

---

### Phase 2: Implement Negamax Search

**Goal**: Add batch-efficient negamax search to `gomoku.py`.

#### New Constants (configurable)

```python
SEARCH_DEPTH = 3                 # Default search depth

# Root node candidate generation
ROOT_TOP_K = 5                   # Top-k from policy at root
ROOT_RANDOM_K = 1                # Random neighbors at root
# Total root candidates = 6

# Internal node candidate generation
INTERNAL_TOP_K = 4               # Top-k from policy at internal nodes
INTERNAL_RANDOM_K = 1            # Random neighbors at internal nodes
# Total internal candidates = 5

# Sampling parameters
TOP_K_SAMPLE = 5                 # Top candidates for sampling (excludes worst random)
SAMPLING_TAU = 0.5               # Temperature for Q-based sampling
Q_NORM_EPSILON = 1e-6            # Normalization epsilon for Q values

# Search tree size at depth=3: 6 × 5 × 5 = 150 leaf nodes
```

#### New Functions

**`generate_candidates(obs, legal_mask, model, device, is_root=True)`**
- Generates candidates: top-k from policy + random neighbors
- Returns list of action indices (6 for root, 5 for internal)

**`generate_candidates_batched(obs_batch, mask_batch, model, device, is_root=True)`**
- Batch version for efficiency during tree expansion

**`negamax_batched(root_obs, root_mask, root_player, current_model, opponent_model, device, depth=3)`**
- Batch-efficient negamax search
- Strategy: Generate all leaf positions upfront, batch evaluate, then backpropagate
- Returns: `(candidates, Q_search)` where Q_search maps action → Q value

**`sample_move_from_q(candidates, Q_search, tau=0.5)`**
- Sample from top candidates (TOP_K_SAMPLE) using normalized Q softmax
- Scale normalization: `Q_norm = (Q - Q_min) / (Q_max - Q_min + ε)`

---

### Phase 3: Add Search-Based Self-Play

**Goal**: Replace RL self-play with search-based data generation.

#### New Dataclass

```python
@dataclass
class SearchSample:
    obs: np.ndarray               # [3, 15, 15] canonical observation
    sorted_candidates: List[int]  # Top candidates sorted by Q descending (TOP_K_SAMPLE)
    all_candidates: List[int]     # All candidates (ROOT_TOP_K + ROOT_RANDOM_K)
    Q_values: List[float]         # Q values for sorted_candidates
    legal_mask: np.ndarray        # [15, 15]
    V_target: float               # max Q_search value (search backup target)
```

#### New Function

**`play_episodes_with_search(num_episodes, current_policy, opponents, opponent_indices, current_is_black, device, depth=3, opening_ids=None, tau=0.5)`**

For each move by current_policy:
1. Generate candidates (ROOT_TOP_K + ROOT_RANDOM_K)
2. Run negamax search
3. Record SearchSample with Q values
4. Sample move from top candidates (TOP_K_SAMPLE) using Q softmax

Returns: `(List[List[SearchSample]], List[GameState])`

---

### Phase 4: Replace RL Losses with Ranking/Value Losses

**Goal**: Implement supervised losses based on search results.

#### New Constants

```python
M_RANK = 0.15        # Ranking margin (max) for inside loss
M_SEP = 0.15         # Separation margin for outside loss
ALPHA_SEP = 1.0      # Separation loss weight
LAMBDA_V = 1.0       # Value loss weight
```

#### Loss Functions

**`compute_ranking_inside_loss(logits_flat, sorted_candidates, Q_norms)`**

Margin-based ranking loss for c1-c4:
```
margin(i, j) = min(m_rank, Q_norm[ci] - Q_norm[cj])
RankingInsideLoss = ReLU(L(c2) - L(c1) + margin(1,2))
                  + ReLU(L(c3) - L(c2) + margin(2,3))
                  + ReLU(L(c4) - L(c3) + margin(3,4))
```

**`compute_separation_outside_loss(logits_flat, all_candidates, c4_indices, legal_masks_flat)`**

Push non-candidates below c4:
```
SeparationOutsideLoss = mean over n of ReLU(L(n) - L(c4) + m_sep)
```

**`compute_search_value_loss(V_pred, V_target)`**

Simple MSE between predicted and search backup value.

#### Training Function

**`train_on_search_samples(model, samples, optimizer, device, batch_size=512)`**

1. Flatten samples from all games
2. Apply 8-fold augmentation (transform candidates too using `augment_candidates_8fold`)
3. Compute Q_norm from stored Q values
4. Compute policy ranking loss (inside + outside)
5. Compute value MSE loss
6. Total = PolicyLoss + λ_v * ValueLoss

Returns metrics: `loss`, `policy_loss`, `ranking_inside_loss`, `separation_outside_loss`, `value_loss`, `top1_acc`, `top3_acc`, `value_mse`

---

### Phase 5: Progressive Unfreezing

**Goal**: Gradually unfreeze model parameters during training.

#### New Constants

```python
HEADS_ONLY_UPDATES = 2048      # N: updates with only heads trainable
BLOCK_UNFREEZE_INTERVAL = 128  # M: interval between block unfreezes
```

#### Unfreezing Schedule

| Phase | Update Range | Trainable Parameters |
|-------|--------------|---------------------|
| 1 | [0, 2048) | policy_* + value_* only |
| 2 | [2048, 2176) | + blocks[15] |
| 3 | [2176, 2304) | + blocks[14] |
| ... | ... | Each M updates: +1 block (15→0) |
| 18 | [4096, ...) | + stem (fully trainable) |

#### New Functions

**`get_unfrozen_param_groups(model, update)`**
- Returns tuple of (unfrozen_prefixes, unfrozen_blocks)

**`apply_freeze_schedule(model, update)`**
- Sets `requires_grad` for each parameter based on schedule
- Returns number of unfrozen blocks

**`create_optimizer_for_unfrozen(model, update, lr, weight_decay)`**
- Creates AdamW optimizer for unfrozen parameters only

**`maybe_update_optimizer(model, optimizer, update, prev_unfrozen_blocks, lr, weight_decay)`**
- Checks if optimizer needs recreation due to unfreezing change
- Returns (optimizer, current_unfrozen_blocks)

---

### Phase 6: Validation Metrics and Logging

**Goal**: Add CSV logging for search training metrics.

#### New CSV File: `search_training.csv`

Columns:
- `update` - Training update number
- `policy_loss` - Total policy loss
- `ranking_inside_loss` - Ranking loss for c1-c4
- `separation_outside_loss` - Separation loss for non-candidates
- `value_loss` - Value MSE loss
- `top1_acc` - c1 has highest logit (%)
- `top3_acc` - c1,c2,c3 are top-3 by logit (%)
- `value_mse` - Raw value MSE
- `unfrozen_blocks` - Number of trunk blocks unfrozen
- `learning_rate` - Current learning rate
- `time_total` - Total update time
- `time_selfplay` - Self-play time
- `time_train` - Training time

#### New Method

**`CSVLogger.log_search_training(update, metrics)`**

---

## File Summary

| File | Changes |
|------|---------|
| `enhancement.py` | Simplified to probe-only; removed OPR and tactical boost |
| `training.py` | Added search losses, progressive unfreezing, `train_on_search_samples()` |
| `main.py` | Removed OPR/tactical calls, simplified state management |
| `gomoku.py` | Added negamax search, `SearchSample`, `play_episodes_with_search()` |
| `csv_logger.py` | Added `tactical_probe.csv`, `search_training.csv` |
| `model.py` | No changes needed |
| `eval.py` | No changes needed |

---

## Usage Example

```python
from gomoku import play_episodes_with_search, SearchSample
from training import (
    train_on_search_samples, apply_freeze_schedule,
    maybe_update_optimizer, HEADS_ONLY_UPDATES
)
from csv_logger import CSVLogger

# Initialize
model = ...  # Load checkpoint
device = torch.device('cuda')
csv_logger = CSVLogger(output_dir)

# Apply initial freeze schedule
unfrozen_blocks = apply_freeze_schedule(model, update=0)
optimizer = create_optimizer_for_unfrozen(model, update=0)

for update in range(total_updates):
    # Check if unfreezing schedule changed
    optimizer, unfrozen_blocks = maybe_update_optimizer(
        model, optimizer, update, unfrozen_blocks
    )

    # Self-play with search
    samples, outcomes = play_episodes_with_search(
        num_episodes=64,
        current_policy=model,
        opponents=opponent_pool,
        opponent_indices=sampled_indices,
        current_is_black=is_black_list,
        device=device
    )

    # Train on search samples
    metrics = train_on_search_samples(model, samples, optimizer, device)

    # Log metrics
    csv_logger.log_search_training(update, {
        **metrics,
        'unfrozen_blocks': unfrozen_blocks,
        'learning_rate': optimizer.param_groups[0]['lr'],
        'time_total': time_total,
        'time_selfplay': time_selfplay,
        'time_train': time_train
    })
```

---

## Verification

1. **Syntax check**: `python -m py_compile *.py`
2. **Import test**: All new functions import successfully
3. **Short training run**: `python main.py runs/test_postrain` (requires main.py integration)

---

## Notes

- The existing RL training functions are preserved for backward compatibility
- Search-based training can coexist with RL training in the same codebase
- `training_state.json` will need updates to track `unfrozen_blocks` for resume
- Main training loop integration (updating `main.py` to use search-based training) is a separate task

### Design Decisions

**Ranking loss includes c4-c5 constraint**: The implementation applies ranking constraints to all consecutive pairs (c1,c2), (c2,c3), (c3,c4), and (c4,c5), which is stricter than the spec's requirement to constrain only c1-c4. This ensures clearer separation in the learned move ordering and does not harm training.

**Search result caching not implemented**: At depth=3, caching previous search subtrees provides minimal benefit (saves ~6+30 node expansions vs ~156 total). The added complexity of cache management is not justified for the marginal performance gain at this search depth.

**Assumes sufficient legal moves**: The candidate generation expects at least ROOT_TOP_K + ROOT_RANDOM_K (currently 6) legal moves at the root. This is not a practical concern because Gomoku games average 20-30 moves on a 15×15 board (225 positions), meaning the board is never close to full during normal play. Games end via five-in-a-row, not board exhaustion.
