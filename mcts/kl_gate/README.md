# kl_gate/ — Is a Position's MCTS Improvement Gap Predictable?

Side study, not part of the pipeline. Stage 2 trains on `avg_raw_mcts_kl` =
`KL(visit_dist ‖ raw prior)`, the amount search moves the policy away from the network's raw
prior. `flip_analysis/` showed this is very unevenly spread across positions — in many of them
search barely changes anything. This study asks whether the per-position gap can be **predicted
from the position itself**, without running the search. That is the prerequisite for anything
that gates search effort by position.

```bash
# from mcts/ (paths resolve relative to it)
python3 kl_gate/data_gen.py     # by far the longer of the two; resumable, skips shards on disk
python3 kl_gate/train.py
```

## Task

Regression target `y = log(KL + LOG_EPSILON)`, loss MSE. The epsilon sits at the finite-simulation
resolution floor of a `LABEL_NUM_SIMULATIONS`-sim search, `~(k_eff − 1) / (2·LABEL_NUM_SIMULATIONS)`,
so the log cannot stretch the unresolvable low end into large negative values that would dominate
the gradient. It has to be retuned whenever `LABEL_NUM_SIMULATIONS` changes.

Input: the three observation planes, `log(prior + PRIOR_LOG_EPSILON)` as a plane, and the network's
value broadcast over the board. The prior enters in log space because `KL(visits ‖ prior)` is a
functional of `log prior`; feeding raw probabilities would make the net learn the log itself from a
plane where nearly every entry is ~0. Occupied squares have prior exactly 0 and therefore sit at the
`PRIOR_LOG_EPSILON` floor — a legitimate marker, and channels 0/1 already flag those squares.

## Data (`data_gen.py`)

`POLICY_PATH` self-plays with the deployment search settings — raw masked-softmax priors
(`entropy_multiplier=None`), no Dirichlet noise, `LABEL_NUM_SIMULATIONS` simulations, sampling
moves at `ACTION_TEMPERATURE` — and every played ply is one sample. `mcts.py` and `self_play.py`
are untouched: `mcts_search_batched` already returns `raw_mcts_kl` per root, which *is* the label.

`POLICY_PATH` must be a **stage-2** checkpoint. Note that `mcts/final_policy.pt` is *not* one — it
is a byte-identical copy of the RL `teacher.pt` (`update = 0`). Labelling with it measures a policy
that was never distilled onto MCTS-shaped targets, so `MCTS(raw)` sits far from `raw` by
construction: a one-shard trial gave mean KL 2.15 (median 0.15) with 33% of positions at
max prior > 0.95, against 8% for the stage-2 network.

`play_and_label` is a trimmed copy of `self_play.play_mcts_games`'s loop rather than a call to it,
because that function builds its boards from `opening_ids` and so cannot express the random-3
start family below. Harvesting, visit-distribution recording and root-Q recording are dropped.

**Start mix** (`P_RENJU` / `P_RANDOM3` / `P_EMPTY`): a Renju opening (184 variants × the board
class's own random offset), three distinct random squares played Black/White/Black (White to
move), or the empty board. Without the random-3 and Renju families the shallow plies would collapse
onto a handful of positions, since the labelling search is deterministic.

**`prior` / `value` back-fill.** Both are recomputed after a shard's games finish, through
`mcts._evaluate_with_cache` — *the same function the search used*. That matters: the network is
not exactly D4-equivariant and the search evaluates in canonical orientation, so a plain
`model(obs)` forward would return a slightly different prior from the one the KL label was measured
against. Every recorded position was evaluated as a search root, so the back-fill is served almost
entirely from the warm NN cache.

Shards are `[N, ...]` `.npz` files of `obs uint8`, `raw_mcts_kl f32`, `prior f32[225]` (linear —
the log transform lives in `dataset.py`), `value f32`, `game_id int32`. Written tmp-then-rename,
skipped if present, with per-shard RNG seeding so a resumed run does not replay the skipped shards'
draws. `game_id < TRAIN_GAMES` is the train split, the rest validation — the split is by game, so
no game contributes plies to both sides.

Note that exact duplicate positions (mostly the empty board and its first few successors) do occur
across the split, so val MSE is mildly optimistic. This is deliberate: the duplication rate is the
true frequency of those positions in self-play.

## Network (`net.py`)

Reuses `main/model.py`'s design ideas — line-aware multi-scale dilated stem with zeroed center taps
on the `d>1` branches, pre-activation residual blocks with SE, per-channel temperature-scaled
log-mean-exp pooling — but the classes are **copied**, not imported, so this study picks its own
width and GroupNorm grouping without touching `model.py` (symlinked from `main/`, shared with the
RL pipeline). `train.py` prints the parameter count at startup.

```
stem:  3x3(NET_STEM_3X3_CHANNELS)
     | dir5 d1+d2(NET_STEM_DIRECTIONAL_5X5_CHANNELS)
     | dir7 d1+d2+d3(NET_STEM_DIRECTIONAL_7X7_CHANNELS)     -> NET_WIDTH
GroupNorm -> SiLU -> NET_BLOCKS x ResidualBlock(NET_WIDTH, NET_DILATION_SCHEDULE[i], SE)
GroupNorm -> SiLU -> LSE pool -> LayerNorm -> FC(NET_HEAD_HIDDEN) -> SiLU -> FC -> scalar
```

The stem branch channels must sum to `NET_WIDTH`.

The head mirrors `model.py`'s value head because the output is likewise a single scalar; there is
no `tanh`, since `log(KL + eps)` is unbounded.

## Training (`train.py`)

Every sample is augmented 8-fold (`dataset.augment_8fold`); because the prior travels as a plane
rather than a `[225]` vector, the same spatial flips/transposes cover it and no permutation table
is needed, and the scalar target is D4-invariant. `BATCH_SIZE` is counted in **base** samples, so
one step sees `8 * BATCH_SIZE` rows. `AdamW(fused=True)`, cosine LR from `LR` to `MIN_LR` over the
whole run.

Each epoch ends with validation in identity orientation: MSE (the objective) plus MAE and Pearson
r. The number to beat is printed at startup — the constant predictor that outputs the training-set
mean everywhere. A good MSE alone proves nothing here; `r` is what shows the model separates
positions.
