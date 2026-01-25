# Repository Guidelines

## Project Structure & Module Organization
This repository is a Python-based Gomoku self-play training project. Key files:
- `main.py`: training entry point; manages checkpoints, evaluation cadence, and resume state.
- `training.py`: core learning loop utilities (losses, GAE, batching).
- `model.py`: network architecture and constants.
- `gomoku.py`: game rules, state encoding, and self-play logic.
- `eval.py` and `enhancement.py`: evaluation helpers and tactical/imitative boosts.
- `play_web.py`: Flask-based web UI for interactive play against checkpoints.
- `csv_logger.py`: CSV telemetry for training/eval metrics.
- `old_code/`: legacy scripts; keep for reference only.
- `j124/`, `j1241/`: sample CSV logs; output artifacts are ignored by `.gitignore`.

## Architecture Overview (Intent & Data Flow)
- `main.py` owns the training lifecycle: build or resume state, sample opponents, self-play, train, evaluate, checkpoint, and persist resume metadata; it also drives historical mining of exploiters.
- `gomoku.py` provides the canonical game representation used everywhere else. It keeps absolute colors internally, but exposes observations as `[3, 15, 15]` with channel 0 = current player, channel 1 = opponent, and channel 2 = a board mask for padding awareness. Actions are flat indices in `[0, 224]`.
- Self-play is batched in `gomoku.py` using per-model grouping to avoid redundant forward passes; each game records a `Trajectory` containing observations, actions, legal masks, log-probs, entropies, and the final outcome.
- `training.py` converts trajectories into learning samples, adds imitation samples when enabled, applies enhancements, augments 8-fold symmetries, computes advantages (GAE or raw returns), and performs gradient-accumulated optimization with policy, value, and entropy losses.
- `enhancement.py` injects tactical supervision (win-in-1 / block detection) and CLER counterfactual rollouts that replace overconfident losing moves with stronger alternatives.
- `model.py` defines the policy+value net: multi-branch dilated stem → residual trunk with scheduled dilation/SE → FiLM-based policy head + value head.
- `csv_logger.py` captures training/eval/mining/probe metrics for later analysis.

## Training Loop Details
- Opponents are sampled from a pool; current policy alternates black/white to balance exposure.
- Opening seeding uses Renju openings with random offsets to avoid overfitting to a single center pattern.
- Training uses gradient accumulation across episode chunks to fit VRAM constraints.
- Entropy bonus is adaptive (target entropy decays over training), stabilizing exploration early and sharpening later.
- Evaluation periodically checkpoints the model, plays round-robin matches vs the pool, and conditionally inserts the new model into the pool.
- Historical mining periodically scans older checkpoints for exploiters and rotates them into the pool.

## Enhancements & Sample Shaping
- Tactical detection identifies immediate wins and blocks; correct moves get advantage boosts, and missed tactics spawn synthetic corrective samples.
- CLER (Counterfactual Low-Entropy Rescue) targets low-entropy losing positions, tests local alternative moves via rollout, and adds samples when alternatives outperform the original by a margin.
- Imitation learning (after a start update) reuses opponent-winning moves from games the current policy won, with a dynamic weight based on win rate.

## Model Notes
- The stem uses mixed receptive fields (dense and sparse/dilated branches) to capture local and wider patterns.
- The policy head uses FiLM to inject global context into local policy features; a bypass path preserves spatial detail.
- The value head uses split dilated convs plus a small MLP and `tanh` output.
- Dilated stem center taps are zeroed and frozen to prevent redundant center contributions.

## Logging & Artifacts
- CSV logs: `training_updates.csv`, `eval_summary.csv`, `eval_opponent_details.csv`, `mining_log.csv`, and `gradient_probe.csv`.
- Checkpoints are stored as `checkpoint_update_*.pt`; final weights saved as `final_policy.pt`; resume state in `training_state.json`.
- Large artifacts are intentionally ignored by `.gitignore`.

## Build, Test, and Development Commands
- `python main.py runs/exp1`: start or resume training; writes `checkpoint_update_*.pt`, `training_state.json`, and CSV logs into the output dir.
- `python play_web.py`: launch the local UI at `http://localhost:5000`; it scans `**/*.pt` under the current working directory, so run it from (or above) a folder containing checkpoints.

## Coding Style & Naming Conventions
- Python with 4-space indentation and PEP 8-style spacing.
- Constants live near module tops and use `UPPER_SNAKE_CASE`.
- Functions/variables use `snake_case`; classes use `CamelCase`.
- Prefer explicit typing when extending public helpers or config values.
- Avoid exact line-number references in docs; cite file + symbol name instead (e.g., `gomoku.py` (Trajectory), `training.py` (train_on_batch)).

## Testing Guidelines
There is no formal test suite in this repo. Use lightweight smoke checks:
- `python -m py_compile *.py` to catch syntax errors.
- Run a short training session in a new output directory and confirm checkpoints/logs are emitted.

## Commit & Pull Request Guidelines
- Commit history shows short, single-line summaries (often lowercase). Follow that style unless a longer description is necessary.
- PRs should include: purpose, key changes, and how to validate (e.g., training command or UI steps).
- Do not commit large artifacts (`*.pt`, `*.log`, `*.json` are ignored by `.gitignore`).
