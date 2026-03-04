# Gomoku AI — Self-Play RL Training & Deployment

Research codebase for training Gomoku (15×15) policy/value networks via self-play reinforcement learning, with browser deployment.

## Directory Structure

- **`main/`** — Primary training pipeline with an advanced model. Entry point: `main.py`.
- **`vanilla/`** — Baseline training pipeline with a simpler model. All `.py` files except `model.py` are symlinks to `main/` to ensure that the training recipe is the same.
- **`deploy/`** — ONNX export (`export_onnx.py`) and browser web app (`web_app/`).

## Running

```bash
# Training (from main/ or vanilla/)
python main.py <output_dir>          # starts or resumes training

# Interactive play against a checkpoint (from main/ or vanilla/)
python play_web.py                   # Flask server at http://localhost:5000

# ONNX export (from deploy/)
python3 export_onnx.py --input checkpoint.pt --output model.onnx

# Lint and type check (run after any code change)
ruff check
pyright
```

## Key Concepts

### Training (shared by `main/` and `vanilla/` unless noted)

- **Self-play RL**: PPO-style policy gradient with no MCTS; the model plays against an opponent pool directly. (`main.py`, `gomoku.py`)
- **GAE**: Generalized Advantage Estimation with negamax value convention; cosine ramp from raw returns to GAE over `BASELINE_RAMP_END` updates. (`training.py`)
- **8-fold augmentation**: All dihedral symmetries (4 rotations × 2 flips) applied on GPU. (`enhancement.py`)
- **Imitation learning**: Opponent winning moves added as training samples, weighted by `(1 - win_rate)`. (`enhancement.py`)
- **Tactical boost**: Detects win-in-1 and block-win-in-1 positions; boosts correct moves and generates synthetic corrective samples. (`enhancement.py`)
- **OPR (Off-Policy Rollout)**: Tests alternative actions near low-entropy lost positions; adds corrective samples when an alternative wins by a margin. (`enhancement.py`, `gomoku.py`)
- **Gradient probe**: Every N updates, computes per-component gradient vectors to detect gradient conflicts. Saves `.npz` files for post-hoc analysis. (`training.py`)
- **Opponent pool & historical mining**: Pool of past checkpoints for self-play evaluation; periodic scanning to mine historically hard opponents. (`eval.py`)
- **Renju openings**: Pre-defined 3-move opening sequences used in `SEED_PROBABILITY` fraction of games for diversity. (`gomoku.py`)

### Model

- **Input**: `[batch, 3, 15, 15]` float32 — channel 0 = current player's stones, channel 1 = opponent's stones, channel 2 = board mask (all 1s). The current/opponent perspective flips each move (the board is always from the side-to-move's point of view).
- **Output**: policy logits `[batch, 225]` (flat 15×15) and scalar value `[batch, 1]` (tanh-bounded, from current player's perspective).
- **vanilla** (`vanilla/model.py`): ResNet-like — single 3×3 stem, 18 plain residual blocks, conv policy head, flatten-FC value head.
- **main** (`main/model.py`): Multi-scale dilated stem (standard + directional convolutions), 12 shared residual blocks + 6 dual-SE blocks (policy/value streams diverge via separate norms and SE), dual-attention policy head with relative positional bias, log-mean-exp pooling value head.

## Other Notes

- Do not hardcode tunable constant values in comments. Values defined as global constants at the top of files (e.g., `WIDTH` in `model.py`) may change; writing comments like `# 96 channels` creates stale documentation. Only mention truly fixed constants (e.g., `# max entropy: ln(225) ≈ 5.416` — this is determined by the board size).
