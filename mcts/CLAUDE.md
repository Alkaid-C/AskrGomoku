# mcts/ — MCTS Training with RL Warm Start

MCTS self-play training for the Gomoku policy/value network, started from an RL-trained checkpoint to skip the noisy random-init phase. The actual MCTS training is **stage 2**; stages 0 and 1 produce the warm-start checkpoint.

```bash
python3 main.py generate_data <working_dir>   # RL teacher.pt -> stage1_data/*.npz   (warm start)
python3 main.py stage1        <working_dir>   # offline distillation -> stage1_final.pt (warm start)
python3 main.py stage2        <working_dir>   # MCTS self-play training -> final_policy.pt
```

The `<working_dir>` must contain `teacher.pt` (a frozen RL checkpoint) before `generate_data`.

## Why a Warm Start

A randomly initialized network produces near-uniform priors and meaningless leaf values, so early MCTS self-play wastes simulations on uninformative exploration and trains the network on essentially-random visit distributions. The warm start sidesteps this stage entirely: the RL checkpoint already plays competently, so the very first MCTS rollout in stage 2 is informative.

The warm start has two phases because the RL checkpoint is a *policy network*, not an MCTS-shaped target. Stage 0 runs MCTS on the RL teacher to produce search-shaped supervision; stage 1 distills a fresh student to that supervision. The output is a network whose policy/value heads are already aligned with the MCTS distribution shape stage 2 will produce.

## Pipeline Stages

### Stage 0: `generate_data` — teacher MCTS rollouts (`data_generator.py`)

Warm-start data prep. The frozen RL teacher runs pure MCTS self-play and writes every ply as a `(obs [3,15,15] uint8, visit_dist [225] f32, root_Q f32)` tuple into sharded `.npz` files. Two temperatures decouple supervision from coverage:

- **`PRIOR_TEMPERATURE`** (entropy multiplier on the prior): per-position, the prior is rescaled so its entropy = `H(softmax(logits)) * PRIOR_TEMPERATURE`. Softens the teacher's logits inside MCTS so the search explores beyond its top moves.
- **`ACTION_TEMPERATURE`**: applied only at action sampling time (`visits ** (1/T)` then renormalize). The supervision target stored in the shard is the *original* visit distribution — sampling temperature broadens trajectories without distorting targets.

No Dirichlet noise during generation. Resumable: shards already on disk are skipped, and per-shard RNG seeding (`_seed_shard_rngs`) ensures different shards get different content.

### Stage 1: `stage1` — offline distillation (`stage1_trainer.py`)

Warm-start training. A freshly initialized `GomokuPolicyNet` student trains on the static stage-0 shards. No MCTS. Per update:

1. Pull `RAW_BATCH_PER_UPDATE` samples (epoch permutation seeded by `(SEED, epoch)`).
2. 8-fold dihedral augmentation (see "Distribution augmentation" below).
3. Gradient accumulation in micro-batches of `TRAIN_BATCH_SIZE`, then `clip_grad_norm_` + step.
4. EMA-track `KL(target || student)` over the last `STAGE1_KL_EMA_WINDOW` updates.
5. If `STAGE1_KL_EMA_THRESHOLD` is set and `kl_ema < threshold`, stop early.

Loss is plain CE + `VALUE_LOSS_COEFF * MSE` against the **raw** visit distribution (no sharpening). LR follows `CosineAnnealingLR` from `STAGE1_LR` to `STAGE1_MIN_LR` over `epochs * updates_per_epoch`. Final: `stage1/stage1_final.pt` — the warm-start checkpoint consumed by stage 2.

### Stage 2: `stage2` — MCTS self-play training (`stage2_trainer.py`)

The actual MCTS training. The warm-started student plays itself with vanilla AlphaZero MCTS. Per update:

1. Sample opening IDs (`SEED_PROBABILITY` fraction get a Renju opening, rest start empty).
2. `play_mcts_games`: model plays both sides in `STAGE2_EPISODES_PER_UPDATE` games, batched into one `mcts_search_batched` call per ply. Every recorded ply becomes a training sample, plus harvested internal nodes (see "Subtree harvesting").
3. `compute_block_rates`: block-win-in-1 hit-rate diagnostic, split black/white.
4. `train_on_mcts_batch` — targets are the raw visit distributions; harvested samples carry per-sample policy/value weights.
5. Clear the NN eval cache and `torch.cuda.empty_cache()`.

Stage 2 uses `entropy_multiplier=None` (raw masked softmax priors), Dirichlet noise (`STAGE2_DIRICHLET_ALPHA`/`EPSILON`) at the root for exploration, and `action_temperature=1.0`. Loss is CE + MSE — no entropy bonus, no GAE, no tactical boost, no imitation, no OPR. The search itself supplies exploration.

Final: `stage2/final_policy.pt`.

#### Subtree harvesting

Each ply's full search evaluates thousands of leaf positions but, by default, only the played root becomes a training sample. Harvesting recovers more of that compute: the C++ `MCTSNode.harvest(min_visits)` walks the tree and emits every internal node (depth ≥ 1) whose visit statistic `N = Σ child_n` clears a threshold. Each emitted node carries its value target (`Σ child_total / N`, identical formula and sign convention to `root_Q`) and policy target (its own child visit distribution); `mcts_search_batched` reconstructs the obs by replaying the node's action path from the root board.

Two thresholds track target reliability: the value target (a low-order mean) is reliable at lower `N` than the policy distribution (higher-order), so `STAGE2_HARVEST_VALUE_MIN_VISITS` < `STAGE2_HARVEST_POLICY_MIN_VISITS`. A node clearing only the value threshold contributes value loss only (`policy_weight = 0`). Each harvested sample is weighted `min(N / NUM_SIMULATIONS_S2, 1)` — inverse-variance-optimal for a mean estimator — so played roots (effectively full budget, weight 1) dominate automatically. **Within-game dedup**: a board reached as internal nodes of several plies' trees (and possibly as a played root) is kept once at its highest-`N` instance; positions also played are dropped (the played root subsumes them). Scope is one game; cross-game repeats are kept. Each loss term is a per-sample weighted **mean** — the weighted sum divided by that term's own weight total (`Σ policy_weight` for CE, `Σ value_weight` for MSE) — i.e. the inverse-variance-weighted estimator. Normalizing by the weight total rather than the sample count keeps the played-root scale stable as harvest volume varies, drops value-only nodes out of the policy denominator, and reduces exactly to the previous unweighted mean when all weights are 1 (no harvest). Reported `policy_loss`/`value_loss`/`kl_target_student` are means over played roots only, so they stay comparable across runs; `VALUE_LOSS_COEFF` may need retuning since the policy and value terms now carry different weight totals.

## File Responsibilities

- `main.py` — Hyperparameter constants (single source of truth) + CLI dispatcher to the three stage entry points.
- `data_generator.py` — Stage-0 rollouts + sharded `.npz` writer with crash-safe rename.
- `stage1_trainer.py` — Offline distillation loop, KL-EMA early-exit, cosine LR, resumable from `stage1/training_state.json`.
- `stage2_trainer.py` — MCTS self-play training loop, cosine LR, resumable from `stage2/training_state.json`.
- `mcts.py` — PUCT tree search, batched cache-aware leaf evaluation, D4 NN-eval cache. `mcts_search_batched` also reconstructs harvested-node obs when `harvest_min_visits` is set.
- `mcts_ext.cpp` — C++ `MCTSNode` (PUCT `select_child`, `backup`, and `harvest(min_visits)` subtree walk). Rebuild in-place with `python3 setup_ext.py build_ext --inplace` after editing.
- `self_play.py` — `play_mcts_games` (used by both stage 0 and stage 2) + `compute_block_rates` diagnostic. Holds the within-game harvest dedup + per-sample weighting (`MCTSGameRecord.harvested`).
- `training.py` — `train_on_mcts_batch` (per-sample weighted CE + MSE against raw visit distributions; weight-1 played roots + weighted harvested samples) and `augment_mcts_batch_8fold`.
- `entropy_ops.py` — `rescale_to_entropy_np`: per-row softmax temperature solved by bisection so that the rescaled distribution has a requested entropy. Used only by `mcts.py::_evaluate_with_cache` for prior softening.
- `csv_logger.py` — `Stage1CSVLogger` and `Stage2CSVLogger` write `training_updates.csv` per stage with stage-specific columns.
- `gomoku.py` → symlink to `main/gomoku.py`
- `model.py` → symlink to `main/model.py`
- `enhancement.py` → symlink to `main/enhancement.py` (only `find_all_win_in_1` and `find_blocking_moves` are used, for the block-rate diagnostic)

## MCTS Search (`mcts.py`)

PUCT selection: `Q(child) + c_puct * P(child) * sqrt(N_parent) / (1 + N_child)`.

### Value & sign convention

- The model outputs value from the **side-to-move's perspective**.
- Each node stores Q (= `child_total / child_n`) from the **parent's perspective** (how good this action was for the parent).
- `backup(leaf, value, gamma)` applies `v ← -v * gamma` at every level, so each ply flips perspective AND discounts. A terminal `±1` reaches the root with magnitude `gamma^depth` and alternating sign — "lose later" is strictly better than "lose sooner".
- Terminal nodes: the player who just moved won (or drew), so `terminal_value = +1` (or `0`) from the parent's perspective. `backup` is called with `-terminal_value` so the side-to-move-at-leaf perspective convention is preserved.
- Root has no parent. It gets `visit_count = 1` as a virtual visit so PUCT uses priors on the first selection. The extracted `root_Q` is the visit-weighted mean of children's Q (= side-to-move's perspective at the root, suitable as a value training target).

### Batched search

For N concurrent positions, each simulation:
1. PUCT-select to a leaf in each tree (children are materialized lazily via `select_child`).
2. Terminal leaves → backup the cached terminal value, skip eval.
3. Non-terminal leaves → clone the board, replay the action path, collect observations.
4. **Cache-aware** batched forward pass on the unique misses only (see below).
5. Expand (store priors over legal actions; child `MCTSNode`s are still lazy), backup leaf values.

Stat storage on each node uses parallel **Python lists** (not numpy arrays) — `child_priors`, `child_q`, `child_total: list[float]`, `child_n: list[int]`, `child_actions: list[int]`, `child_node: list[Optional[MCTSNode]]`. Native-list indexing in the PUCT inner loop is ~3× faster than numpy scalar indexing.

### D4-canonical NN eval cache

Across one update, many MCTS positions repeat under D4 symmetry. `_evaluate_with_cache`:

1. For each obs, picks the lexicographically smallest of its 8 D4 transforms as a canonical key (`canonicalize_obs`, packs the 2 stone planes into 57 bytes).
2. Misses are forwarded through the model in canonical orientation in fixed-size chunks of `_FIXED_FWD_BATCH`, with short trailing chunks zero-padded to the same shape. Two reasons: (a) uniform chunk shapes prevent allocator fragmentation (variable batch sizes caused a ~14 GB high-water mark in stage 0); (b) per-position throughput peaks at this size for this model's attention-dominated arithmetic and degrades with larger batches. The result (post-mask, post-softmax prior + value) is cached.
3. Hits collapse to a permutation lookup: `priors[i] = canonical_priors[_FORWARD_PERM[s]]`.

Caching the *fully scaled* prior (not raw logits) is sound because `entropy_multiplier` is constant within any window where the cache lives. The cache is an `OrderedDict` with LRU eviction capped at `_NN_EVAL_CACHE_MAX_ENTRIES`. Stage 2 calls `clear_nn_eval_cache` at every optimizer.step() (model weights change). Stage 0 (data generation) runs against a frozen teacher and does **not** clear between shards — entries stay valid and the LRU cap bounds memory.

### Dirichlet noise

Added to root priors only. Support is restricted to **legal moves within Chebyshev distance `_DIRICHLET_NEIGHBORHOOD_RADIUS` (= 4) of any existing stone** — the tactical horizon of a single stone. This keeps `Kα` close to the AlphaZero ~10 sweet spot even though Gomoku has 225 legal moves at the opening, and avoids spending exploration mass on faraway empty corners that can never matter on a ~25-ply game. On an empty board the restriction is skipped and noise covers all legal moves. The ε mass is distributed only over the noise support; positions outside it keep their plain `(1 − ε)·P` mass, and the resulting priors over legal moves still sum to 1. Skipped entirely when `dirichlet_epsilon == 0`.

## Entropy Rescaling (`entropy_ops.py`)

Prior softening is softmax temperature scaling. The naive `p**alpha / Z` form gives an entropy that depends on the input's shape. `rescale_to_entropy_np` instead solves for the per-row temperature that yields a requested target entropy via bisection in log-tau space (24 iters, ~1.4e-6 nat precision). Degenerate near-onehot rows fall back to plain softmax.

Single caller — **prior softening** (`mcts.py::_evaluate_with_cache`): `target_H = min(H_model * entropy_multiplier, ln 225)`. Used during stage 0 with `entropy_multiplier = PRIOR_TEMPERATURE`. Stage 2 passes `entropy_multiplier=None`, which skips the rescale and uses raw masked softmax.

## Training (`training.py`)

### Loss (`train_on_mcts_batch`, stage 2)

```
loss = policy_loss + VALUE_LOSS_COEFF * value_loss
```

- **Policy loss**: per-sample weighted mean of CE against the raw visit distribution — `(policy_w * ce).sum() / Σ policy_w`.
- **Value loss**: per-sample weighted mean of squared error against MCTS root Q — `(value_w * se).sum() / Σ value_w`.
- Played roots have weight 1 on both terms; harvested nodes carry `min(N / NUM_SIMULATIONS_S2, 1)` (0 policy weight = value-only — see "Subtree harvesting"). Both weight totals are global over the full augmented batch, so they are constant across micro-batches and gradient accumulation is exact. When every weight is 1 (no harvest) each term reduces to the previous unweighted mean.
- **`kl_target_student`** = CE(target, student) − H(target). Reported alongside `policy_loss`/`value_loss` as an *unweighted* mean over played-root samples only (`value_weight == 1.0`), so the metrics stay comparable across runs with and without harvesting.

Stage 1 (`stage1_trainer.py`) does not call `train_on_mcts_batch`; it uses a plain unweighted CE + MSE (all weights 1, no harvested samples) and logs the same `kl_target_student`.

Logits are masked with `LOGIT_MASK_VALUE` on illegal squares before softmax.

### Distribution augmentation

`augment_mcts_batch_8fold` applies the 8 dihedral transforms. For obs and masks it uses spatial flips/transposes (matching `augment_batch_8fold` in `enhancement.py`). For the `[225]` visit distribution it precomputes 8 inverse permutation tables from `enhancement.py`'s `new_rows`/`new_cols` coordinate transforms: `new_dist[s, new_idx] = old_dist[inv_perm[s, new_idx]]`. The forward permutation table in `mcts.py::_FORWARD_PERM` and the inverse table in `training.py::DIST_PERM_TABLES` derive from the same coordinate transforms — they must stay in sync.

Gradient accumulation across micro-batches of `TRAIN_BATCH_SIZE`, then `clip_grad_norm_(GRAD_CLIP_NORM)` + `optimizer.step()`.

## Hyperparameters & Schedules

All hyperparameters are module-level constants at the top of `main.py` (no CLI tuning surface).

- **Optimizer**: `AdamW(fused=True)` with `WEIGHT_DECAY` (`1 / 2**24`).
- **LR**: `CosineAnnealingLR` per stage. Stage 1 from `STAGE1_LR` → `STAGE1_MIN_LR` over `epochs * updates_per_epoch`. Stage 2 from `STAGE2_LR` → `STAGE2_MIN_LR` over `STAGE2_TOTAL_UPDATES`.
- **MCTS backup discount**: `DISCOUNT_GAMMA` (shared by all stages).
- **PUCT**: `C_PUCT` (shared).
- **Checkpointing**: every `STAGE{1,2}_CHECKPOINT_INTERVAL` updates as `checkpoint_update_{N}.pt`, paired with `training_state.json` for resume.

## Resume Semantics

Each stage owns its `training_state.json` and is independently resumable from the most recent checkpoint named in that file:

- **Stage 1**: state stores `current_update` and `kl_ema`. Resume re-loads model, optimizer, scheduler. Final write produces both `checkpoint_update_{N}.pt` and `stage1_final.pt`.
- **Stage 2**: state stores `current_update`. Stage 1's `stage1_final.pt` is required only for a fresh start; once stage 2 has its own state file, it resumes independently. Final write produces `final_policy.pt` (also a `checkpoint_update_{N}.pt`).
- **Stage 0**: no state file — uses on-disk shard presence as the resume signal. Per-shard RNG seeding keeps newly generated shards from duplicating skipped ones.
