# main/ — Training Pipeline (Advanced Model)

## File Responsibilities

- `main.py` — Entry point. Training loop, state save/load, CLI.
- `model.py` — `GomokuPolicyNet` architecture and constants. Rarely changed.
- `training.py` — Loss computation, GAE, gradient accumulation, gradient probe.
- `enhancement.py` — Sample augmentation/generation: 8-fold symmetry, tactical boost, OPR, imitation.
- `gomoku.py` — Game engine, batched self-play, action selection, Renju openings. Rarely changed.
- `eval.py` — Opponent pool management, evaluation, historical mining.
- `csv_logger.py` — CSV logging for all metrics.
- `play_web.py` — Flask server for interactive play against a checkpoint.

## Model Architecture (`model.py`)

### Stem

Multi-branch design — 6 parallel branches on the raw `[3, 15, 15]` input. Branch widths sum to `WIDTH`, so the branches are concatenated directly into the trunk input: concat → GroupNorm → SiLU. The branches:

- `conv_3x3`: standard 3×3
- `conv_full5`, `conv_full7`: standard 5×5 and 7×7
- `conv_directional5_*`: sum of two dilated 3×3 (d=1, d=2) — builds a kernel covering 4 line directions at range 2
- `conv_directional7_*`: sum of three dilated 3×3 (d=1, d=2, d=3) — 4 line directions at range 3
- `conv_1x1`: 1×1

Center taps of d>1 directional convs are zeroed (init + gradient hook) because the d=1 conv already covers the center.

### Trunk

- `shared_blocks`: `N_SHARED_BLOCKS` standard `ResidualBlock` (pre-activation, GroupNorm, optional dilation on conv2)
- `dual_se_blocks`: `N_DUAL_SE_BLOCKS` `DualSEResidualBlock` — shared conv weights, but separate GroupNorm and SE per stream (policy/value). Over multiple blocks, the differential SE gating makes the two streams progressively diverge.
- Dilation schedule for conv2 in each block: `TRUNK_DILATION2_SCHEDULE`

`forward()` runs both streams (training). `forward_policy_only()` (self-play/eval) and `forward_value_only()` (GAE) run a single stream through the dual-SE blocks.

### Policy Head

Dual-attention with conv refinement:
1. Fused 1×1 projection → split into features + Q/K/V
2. First `AttentionCore` (multi-head self-attention + dihedral-symmetric 2D relative positional bias, `out_proj` zero-init)
3. Two-conv residual refinement
4. Second `AttentionCore` (separate Q/K/V projection)
5. Final conv → 1×1 → 225 logits

### Value Head

1. 1×1 conv (`WIDTH` → `VALUE_HEAD_C1`) → grouped 3×3 conv (`VALUE_HEAD_C1` → `VALUE_HEAD_C2`, groups=`VALUE_HEAD_GROUPS`)
2. Per-channel log-mean-exp pooling: `τ * (logsumexp(x/τ) - ln(225))` where τ = softplus(learnable). Interpolates between max-pool (τ→0) and mean-pool (τ→∞).
3. LayerNorm → FC → SiLU → FC → tanh

## Training Loop (`main.py`)

Each update:
1. Sample opponents from pool (`UNIFORM_SAMPLING_FRACTION` uniform, rest difficulty-weighted)
2. Play `EPISODES_PER_UPDATE` games via `play_episodes_batched` (`SEED_PROBABILITY` start from Renju openings)
3. Generate OPR samples from lost games (`enhancement.py`)
4. `train_on_batch`: tactical enhancement → 8-fold augmentation → GAE → loss → gradient probe → optimizer step
5. Every `get_eval_interval(update)` updates: save checkpoint, evaluate vs pool, conditionally add to pool, possibly mine historical exploiters

## Loss Function (`training.py`)

```
loss = policy_loss + value_coeff * value_loss + ENTROPY_BONUS_COEFF * entropy_bonus_scale * entropy_loss
```

- **Policy loss**: `-Σ(weight * advantage * log_prob) / Σ(weight)`. Advantage is a blended `(1-α)*return + α*GAE + tactical_boost`, passed through a leaky ReLU (`NEGATIVE_ADVANTAGE_SLOPE`) that attenuates negative advantages rather than zeroing them.
- **Value loss**: weighted MSE between predicted value and a TD(λ)-return target (negamax convention — consecutive plies alternate player, so the sign flips each step). Terminal step uses the raw return `z_t`; each non-terminal step uses `value_return = -(1 - VALUE_GAE_LAMBDA)·V(next) - VALUE_GAE_LAMBDA·value_return`, a λ-blend between bootstrapping on `-V(next)` (λ=0) and the Monte-Carlo return (λ=1). With `VALUE_GAE_LAMBDA = 13/16` the target is mostly Monte-Carlo with light bootstrapping. Only computed on real samples (not synthetic). Coefficient ramps from `VALUE_LOSS_COEFF_START` to `VALUE_LOSS_COEFF_END` via α.
- **Entropy loss**: `-Σ(weight * entropy) / Σ(weight)`, scaled by `entropy_bonus_scale = entropy_schedule / ema_entropy` (adaptive ratio).

## Schedules and Ramps

- **LR**: tanh decay from `LEARNING_RATE` to `MIN_LR`. Midpoint at `LR_DECAY_MIDPOINT_PERCENTAGE` of training, transition width `LR_DECAY_STEEPNESS`.
- **Entropy target**: sigmoid (tanh-based) decay from `ENTROPY_TARGET_START` to `ENTROPY_TARGET_END`. Midpoint at `ENTROPY_DECAY_MIDPOINT_PERCENTAGE`, width `ENTROPY_DECAY_STEEPNESS`.
- **Baseline α** (GAE ramp): cosine ramp from 0→1 over `[0, BASELINE_RAMP_END]`. At α=0 uses raw returns; at α=1 uses GAE. Also controls value loss coefficient interpolation.
- **Eval interval**: `EVAL_INTERVAL_EARLY` (updates < 128) → `EVAL_INTERVAL_MID` (< 2048) → `EVAL_INTERVAL_LATE`.

## Enhancement Details (`enhancement.py`)

### Tactical Boost

Scans every training sample for win-in-1 and block-win-in-1. If detected:
- **Correct move taken**: adds a boost to the sample's advantage (adaptive, based on miss rate EMA)
- **Correct move missed**: generates synthetic samples with the correct move and a fixed high advantage

### OPR (Off-Policy Rollout)

Starts at `OPR_START_UPDATE`. For lost games, with probability `OPR_TRIGGER_PROB`:
1. Find steps where policy entropy < `entropy_schedule * OPR_ENTROPY_TH_MULTIPLIER` and far from terminal
2. Weighted-sample one such step (weight = threshold − entropy)
3. Try `OPR_NUM_ACTIONS` local alternative actions, each with `OPR_NUM_ROLLOUTS` rollouts
4. Also rollout the original action for fair comparison
5. If best alternative beats original by ≥ `OPR_WIN_MARGIN`, add as synthetic training sample

### Imitation Learning

From `IMITATION_START_UPDATE`: opponent moves that led to opponent winning are added as training samples — but **only when the opponent played White** (`traj.players[step_idx] == Player.WHITE`). Black moves first, and that first-move advantage is large enough that a Black win is unsurprising and carries little signal worth imitating; a White win is the informative case. (Opponent moves not imitated still enter the batch as value-only samples so the negamax GAE recursion sees the full alternating-player value sequence.) Weight = `(1 - win_rate) * (IMITATION_MAX_WEIGHT - IMITATION_MIN_WEIGHT) + IMITATION_MIN_WEIGHT`.

## Gradient Probe (`training.py`)

Every `PROBE_INTERVAL` updates, computes 5 separate gradient vectors (policy_real, policy_synthetic, value_real, entropy_real, entropy_synthetic) by running 5 backward passes. Saves full vectors to `.npz` files (`gradient_probe_NNNNNN.npz`) containing param names, offsets, and all gradient vectors for post-hoc analysis (e.g., per-layer cosine similarity).

## Opponent Pool & KL-Aware Mining (`eval.py`)

The self-play opponent pool is fixed-capacity (`OPPONENT_POOL_SIZE`); new checkpoints are added and old ones evicted, and a periodic scan mines historical checkpoints that the current model struggles against. To keep the pool *behaviorally* diverse — not a cluster of near-identical checkpoints, which would narrow the self-play signal — both eviction and mining are biased by a symmetric-KL distance between checkpoints.

### Fingerprints

A checkpoint's "fingerprint" is its masked log-softmax policy over a fixed set of probe positions (`fingerprint_model`). The probe positions are regenerated at the start of each mining scan from the *current* model's self-play on Renju openings (`generate_fingerprint_positions`, `FINGERPRINT_NUM_TRAJECTORIES` trajectories), and every pool member is re-fingerprinted into `_fp_cache`. `compute_symmetric_kl` is the mean symmetric KL (nats) between two fingerprints — a behavioral distance; `compute_min_kl` gives a checkpoint's distance to its *nearest* neighbor in a set (its redundancy). All KL machinery is gated on fingerprint state existing; before the first scan, eviction and mining fall back to pure win-rate paths.

### KL-aware eviction (`_evict_kl_aware`)

When the pool is full, each member is scored `score = win_rate - (win_rate_range · EVICTION_KL_WEIGHT) · (min_kl / max_min_kl)` and the highest score is evicted. So an opponent that is both *easy* (high win rate against it) and *redundant* (small `min_kl` to its nearest neighbor) is the preferred eviction target; a behaviorally distinctive opponent is protected even when it is easy. The new entrant participates in the distance computation but is never an eviction candidate. Fallback when no fingerprints: `evict_easiest_opponent` (pure win-rate).

### KL-aware mining acceptance

When the scan ranks historical candidates, each gets a per-candidate acceptance threshold instead of the flat `MINING_WIN_RATE_THRESHOLD`:

```
threshold = MINING_BASE_THRESHOLD + (win_rate_ema - MINING_BASE_THRESHOLD) · min(MINING_KL_CAP, min_kl) / MINING_KL_CAP
accept if candidate_win_rate < threshold
```

A candidate distinct from the pool (large `min_kl`, capped at `MINING_KL_CAP`) earns a threshold relaxed toward `win_rate_ema`, so it can be mined even if it is only moderately hard; a redundant candidate (small `min_kl`) must be genuinely hard (win rate near `MINING_BASE_THRESHOLD`) to be accepted. At most `MAX_MINED_OPPONENTS_PER_EVENT` are mined per scan.
