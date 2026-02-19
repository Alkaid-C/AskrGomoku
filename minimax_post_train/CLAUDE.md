# minimax_post_train/ — Post-Training Pipeline (Search Supervision)

Takes a `main/`-trained checkpoint and refines it using negamax tree search with ranking losses and progressive unfreezing. `model.py` must stay in sync with `main/model.py`.

## File Responsibilities

- `main.py` — Entry point. Training loop with search-based self-play, state save/load, CLI.
- `model.py` — Same as `main/model.py` (must stay in sync).
- `training.py` — RL loss (same as main/), search-based losses (ranking, separation, value), progressive unfreezing logic.
- `enhancement.py` — 8-fold augmentation and tactical accuracy probing (monitoring only — no OPR, no imitation, no tactical boost for training).
- `gomoku.py` — Game engine (same as main/) plus negamax search: `play_episodes_with_search`, `SearchSample` dataclass.
- `eval.py` — Opponent pool management, evaluation, historical mining (same as main/).
- `csv_logger.py` — CSV logging. Adds `tactical_probe.csv` and `search_training.csv` on top of the standard files.
- `play_web.py` — Flask server for interactive play.
- `test_margins.py` — Standalone utility: runs self-play games on a checkpoint (`rl.pt` in CWD), analyzes logit gaps, prints recommended `M_RANK`/`M_SEP` values.

## Training Loop (`main.py`)

Each update:
1. Play `EPISODES_PER_UPDATE` games via `play_episodes_with_search` — each game produces `SearchSample` objects instead of raw trajectories
2. `train_on_search_samples`: 8-fold augmentation (including candidate index transforms) → search-based losses → optimizer step
3. Every `PROBE_INTERVAL` updates: run tactical accuracy probe on search samples (monitoring, no training effect)
4. Evaluation, opponent pool, and historical mining work the same as main/

## Negamax Search (`gomoku.py`)

`play_episodes_with_search` runs games where the current policy's moves are selected via negamax tree search:

1. At each position, select `ROOT_TOP_K` top policy candidates + `ROOT_RANDOM_K` random legal moves as root candidates
2. For each root candidate, expand `INTERNAL_TOP_K` + `INTERNAL_RANDOM_K` candidates at each internal node, down to `SEARCH_DEPTH`
3. All leaf positions are generated upfront and batch-evaluated in a single forward pass
4. Q values are backpropagated up the tree (negamax: negate at each level)
5. Move is sampled from top candidates weighted by softmax over Q values with temperature `SAMPLING_TAU`

Produces `SearchSample` per position: `obs`, `sorted_candidates` (top-k by Q, descending), `all_candidates`, `Q_values`, `legal_mask`, `V_target` (max Q).

## Search-Based Loss (`training.py`)

```
loss = (ranking_inside + ALPHA_SEP * separation_outside + LAMBDA_V * value_loss) / num_batches
```

- **Ranking inside** (`compute_ranking_inside_loss`): pairwise margin loss over sorted candidates. For adjacent pairs (c1,c2), (c2,c3), etc.: `ReLU(L(c_lower) - L(c_higher) + margin)` where margin = `min(M_RANK, Q_norm_diff)`.
- **Separation outside** (`compute_separation_outside_loss`): pushes all non-candidate legal moves below c4. For each non-candidate n: `ReLU(L(n) - L(c4) + M_SEP)`. Per-sample mean, then batch mean.
- **Search value loss** (`compute_search_value_loss`): MSE between predicted value and `V_target` (max Q from search).

## Progressive Unfreezing (`training.py`)

Controlled by `apply_freeze_schedule` / `maybe_update_optimizer`:

1. Updates `[0, HEADS_ONLY_UPDATES)`: only policy and value head parameters are trainable
2. After `HEADS_ONLY_UPDATES`: every `BLOCK_UNFREEZE_INTERVAL` updates, unfreeze one more block from the trunk end — `dual_se_blocks[5]` → `[4]` → ... → `[0]` → `shared_blocks[11]` → ... → `[0]`
3. After all blocks unfrozen: unfreeze stem

When the unfreezing boundary changes, the optimizer is recreated with the newly unfrozen parameters (Adam momentum resets).
