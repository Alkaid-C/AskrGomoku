# vanilla/ — Training Pipeline (Simple Model)

Every `.py` file here except `model.py` is a symlink to `main/`, so the training
recipe is identical by construction. **See `main/CLAUDE.md` for the training
pipeline** — file responsibilities, training loop, loss function, schedules and
ramps, enhancements (tactical boost, OPR, imitation), and the gradient probe.
Do not duplicate that material here; it would only drift.

This file covers the one thing that is genuinely vanilla-specific: the model.

## Model Architecture (`model.py`)

Same three-method API as `main/model.py`: `forward()` (both heads), `forward_policy_only()` (self-play/eval), `forward_value_only()` (GAE). Block count is read from the `N_BLOCKS` constant — not a constructor argument.

### Stem

Single 3×3 conv (`3` → `WIDTH`) → GroupNorm → SiLU.

### Trunk

`N_BLOCKS` standard `ResidualBlock` (pre-activation, GroupNorm, no dilation, no SE) → GroupNorm → SiLU.

### Policy Head

3× Conv3×3 with GroupNorm + SiLU → Conv1×1 → 225 logits.

### Value Head

Conv1×1 (`WIDTH` → `VALUE_HEAD_CHANNELS`) → SiLU → flatten → FC (`VALUE_HEAD_CHANNELS * 225` → `VALUE_HEAD_HIDDEN`) → SiLU → FC → tanh.
