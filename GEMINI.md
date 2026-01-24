# vibe2 - Gomoku Reinforcement Learning

**vibe2** is an advanced reinforcement learning system designed to train a high-performance Artificial Intelligence agent for the game of Gomoku (Five-in-a-Row). It employs a self-play training loop inspired by AlphaZero, augmented with specific tactical enhancements, counterfactual learning, and a sophisticated neural network architecture.

## Project Overview

The project uses a **Policy-Value Network** trained via self-play. Key features include:

*   **Advanced Architecture**: A ResNet-style backbone with Squeeze-and-Excitation (SE) blocks, a complex dilated stem, and Feature-wise Linear Modulation (FiLM) in the policy head.
*   **Tactical Enhancements**: Explicit handling of tactical situations (Win-in-1, Blocking) to prevent simple blunders during training.
*   **CLER (Counterfactual Low-Entropy Rescue)**: A mechanism to actively learn from "overconfident" mistakes by simulating alternative actions in lost games.
*   **Historical Exploiter Mining**: The system periodically scans past checkpoints to find "exploiters" (older models that beat the current one) and re-adds them to the training opponent pool to prevent cyclic forgetting.
*   **Opponent Pool**: Maintains a dynamic pool of opponents with varying difficulty levels to ensure robust training.

## Codebase Structure

### Core Logic
*   **`main.py`**: The entry point. Handles the main training loop, argument parsing, state persistence (`training_state.json`), and orchestrates self-play and updates.
*   **`model.py`**: Defines `GomokuPolicyNet`.
    *   **Stem**: Mixed sparse/dense dilated convolutions.
    *   **Trunk**: 16 Residual blocks with specific Dilation and SE schedules.
    *   **Heads**: FiLM-modulated Policy Head and a split Value Head.
*   **`gomoku.py`**: The game engine. Implements the 15x15 board, Renju opening sequences for diversity, and batched inference logic.
*   **`training.py`**: Implements the learning mathematics.
    *   **GAE**: Generalized Advantage Estimation.
    *   **Losses**: Policy gradient, Value loss (MSE), and Entropy regularization.
    *   **Gradient Probing**: Tools to detect conflicts between policy and value gradients.

### Utilities & Enhancements
*   **`enhancement.py`**:
    *   **Tactical**: Detection of immediate winning/blocking moves.
    *   **Augmentation**: GPU-accelerated 8-fold symmetry (rotation/flipping).
    *   **CLER**: Logic for generating counterfactual training samples.
*   **`eval.py`**: Logic for evaluating the current model against the opponent pool and scanning historical checkpoints.
*   **`csv_logger.py`**: Handles structured logging of training metrics to CSV files.

### Interface
*   **`play_web.py`**: A standalone Flask web server. Allows humans to play against trained checkpoints and visualize the AI's probability heatmap and value estimation.

## Setup & Usage

### Prerequisites
*   Python 3.x
*   PyTorch (CUDA recommended for training)
*   NumPy
*   Flask (for the web UI)

### Training
To start a new training run or resume an existing one:

```bash
python main.py <output_directory>
```

Example:
```bash
python main.py runs/experiment_1
```
*   This will create the directory, initialize a `training_state.json`, and begin the self-play loop.
*   Logs will be written to `training_updates.csv` and `eval_summary.csv` within that directory.

### Web Interface
To play against your trained models:

1.  Ensure you have checkpoints (e.g., `checkpoint_update_100.pt`) in your output directory.
2.  Run the web server:
    ```bash
    python play_web.py
    ```
3.  Open `http://localhost:5000` in your browser.
4.  Select a checkpoint from the dropdown menu.

## Configuration
Hyperparameters (Learning Rate, Batch Size, Network Depth, etc.) are defined as constants at the top of their respective files (primarily `main.py`, `training.py`, and `model.py`). Modify these files directly to tune the training process.
