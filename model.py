"""
Gomoku Policy+Value Network Architecture and Configuration.

Contains the neural network architecture (GomokuPolicyNet), configuration constants,
and inference helper functions used during self-play and evaluation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from typing import List, Tuple


# ============================================================================
# Configuration
# ============================================================================

# --- Training Duration ---
TOTAL_UPDATES = 65536

# --- Model Architecture ---
N_BLOCKS = 16              # Number of residual blocks
WIDTH = 96                 # Residual block width (must equal sum of all stem channels)
STEM_3X3_CHANNELS = 6 * 6      # 3x3 convolution channels in stem
STEM_SPARSE_5X5_CHANNELS = 3 * 6  # Sparse 5x5 (dilated 3x3 sum) channels in stem
STEM_DENSE_5X5_CHANNELS = 2 * 6   # Dense 5x5 convolution channels in stem
STEM_SPARSE_7X7_CHANNELS = 3 * 6  # Sparse 7x7 (dilated 3x3 sum) channels in stem
STEM_DENSE_7X7_CHANNELS = 1 * 6   # Dense 7x7 convolution channels in stem
STEM_1x1_CHANNELS = 1 * 6
GROUPNORM_GROUPS = 16       # Groups for GroupNorm layers (must divide WIDTH evenly)

# Trunk dilation schedule for conv2 in each residual block (length must equal N_BLOCKS)
TRUNK_DILATION2_SCHEDULE = [
    1, 1, 2, 3,
    1, 1, 2, 3,
    1, 1, 2, 3,
    1, 1, 1, 1,
]
# SE schedule: True to enable Squeeze-and-Excitation for that block (length must equal N_BLOCKS)
SE_SCHEDULE = [
    False, False, False, False,
    True, True, False, False,
    True, True, False, False,
    True, True, False, False,
]
POLICY_HEAD_D = 64          # Policy head intermediate channels (d_p)
POLICY_HEAD_MLP_HIDDEN = 64 # Policy head global MLP hidden size (h)
VALUE_HEAD_CHANNELS = 16    # Channels after value head 1x1 conv (d)
VALUE_HEAD_HIDDEN = 96      # Hidden layer size for value head MLP

# --- Optimizer & Learning Rate ---
LEARNING_RATE = 8e-4
MIN_LR = 1e-4
LR_DECAY = (MIN_LR / LEARNING_RATE) ** (1.0 / TOTAL_UPDATES)  # Derived
WEIGHT_DECAY = 1e-8
GRAD_CLIP_NORM = 16.0

# --- Batching & Memory ---
EPISODES_PER_UPDATE = 64    # Episodes to collect before each training update
EPISODES_CHUNK_SIZE = 32    # Chunk size for gradient accumulation (saves VRAM)
BATCH_INFERENCE_SIZE = 64   # Positions processed simultaneously during self-play
TRAIN_BATCH_SIZE = 1024     # Micro-batch size for training

# --- Exploration & Entropy ---
TEMPERATURE_TRAIN = 1.25    # Flattens sampling distribution
ENTROPY_COEFF_START = 1e-3  # Compensated for 1/T gradient scaling
ENTROPY_COEFF_END = 1e-5    # Final entropy coefficient
ENTROPY_DECAY_MIDPOINT_PERCENTAGE = 0.75  # Transition occurs at 75% of training
ENTROPY_DECAY_STEEPNESS = 0.5  # Transition spread over 50% of total training duration

# --- Value Head & Advantage Estimation ---
VALUE_LOSS_COEFF = 0.5      # Weight for value head loss
GAE_LAMBDA = 0.95           # GAE lambda (0=TD(0), 1=MC)
VALUE_BASELINE_START = 1024 # Update at which to start using value baseline

# --- Tactical Enhancements ---
MISS_RATE_EMA_WINDOW = 128  # Effective window for miss rate EMA
WIN_MIN_BOOST = 0.0         # Minimum boost for win-in-1 (when miss rate is 0)
WIN_MAX_BOOST = 1.0         # Maximum boost for win-in-1 (when miss rate is 1)
BLOCK_MIN_BOOST = 0.0       # Minimum boost for blocking (when miss rate is 0)
BLOCK_MAX_BOOST = 0.75      # Maximum boost for blocking (when miss rate is 1)

SYNTHETIC_WIN_BOOST = 2.0   # Signal for missed win-in-1 (synthetic examples)
SYNTHETIC_BLOCKING_BOOST = 1.0  # Signal for missed blocks (synthetic examples)
MAX_SYNTHETIC_WINS = 256    # Max synthetic win-in-1 examples per batch
MAX_SYNTHETIC_BLOCKS = 256  # Max synthetic blocking examples per batch
EPISODE_WEIGHT_ALPHA = 0.5  # 0 => per-step weighting, 1 => per-episode equal mass

# --- Imitation Learning ---
IMITATION_WEIGHT = 0.6      # Weight for learning from opponent's winning moves
IMITATION_START_UPDATE = 2048  # Update at which to enable imitation learning

# --- Self-Play & Evaluation ---
OPPONENT_POOL_SIZE = 16
EVAL_ROUNDS = 32           # Rounds per eval
EVAL_TEMP = 1.0            # Temperature for evaluation
EVAL_INTERVAL_EARLY = 8    # Eval interval for early training phase
EVAL_INTERVAL_MID = 32     # Eval interval for mid training phase
EVAL_INTERVAL_LATE = 128   # Eval interval for late training phase
WIN_RATE_THRESHOLD = 0.625 # Win rate needed to update opponent pool

# --- Historical Exploiter Scanning ---
SCAN_START_UPDATE = 8192         # Start scanning after this update (late phase)
SCAN_PERIOD = 16                 # Scan every N evaluations (after SCAN_START_UPDATE)
NUM_SCAN_BUCKETS = 4             # Number of buckets for round-robin checkpoint coverage
QUICK_SCREEN_ROUNDS = 16         # Games per candidate in quick screening phase
TOP_K_QUICK_SCREEN = 16          # Number of hardest candidates to advance to final screen
FINAL_SCREEN_ROUNDS = 64         # Games per candidate in final screening phase
MAX_MINED_OPPONENTS_PER_EVENT = 2  # Max historical exploiters added per scan event

# --- Dynamic Opponent Sampling ---
UNIFORM_SAMPLING_FRACTION = 0.5    # Fraction of episodes using uniform sampling
DEFAULT_WIN_RATE = 0.5             # Default win rate for opponents without eval stats

# --- Numerical Stability ---
LOG_PROB_MIN = -10.0       # Minimum log probability
LOGIT_MASK_VALUE = -1e9

# --- Logging & Checkpointing ---
PRINT_INTERVAL = 8         # Print stats every N updates
TRAINING_STATE_FILE = "training_state.json"

# ============================================================================
# PyTorch Performance Settings
# ============================================================================

# --- Device ---
DEVICE = torch.device("cuda")
torch.backends.cudnn.conv.fp32_precision = 'tf32'  # New API in PyTorch2.9
torch.backends.cuda.matmul.fp32_precision = 'tf32' # New API in PyTorch2.9


# ============================================================================
# Neural Network Architecture
# ============================================================================

def _zero_center_tap_hook(grad: torch.Tensor) -> torch.Tensor:
    """Backward hook to zero gradients at center tap position (1,1) of 3x3 kernels."""
    grad = grad.clone()
    grad[:, :, 1, 1] = 0
    return grad


class GomokuPolicyNet(nn.Module):
    """
    Policy + Value neural network for Gomoku.

    Architecture:
    - Mixed stem with dilated convolution branches
    - Residual trunk with configurable depth and dilation schedule
    - Policy head: FiLM-based (local conv tower + global FiLM modulation)
    - Value head: 2x 3x3 valid convs (15->11) + 1x1 reduction + 2-layer MLP
    """

    def __init__(self, n_blocks: int = N_BLOCKS):
        super().__init__()

        # === Stem: dilated design ===
        # Input channels: 3 (current player pieces, opponent pieces, board mask)
        self.conv_3x3 = nn.Conv2d(3, STEM_3X3_CHANNELS, kernel_size=3, stride=1, padding=1, dilation=1)

        self.conv_sparse5_d1 = nn.Conv2d(3, STEM_SPARSE_5X5_CHANNELS, kernel_size=3, stride=1, padding=1, dilation=1)
        self.conv_sparse5_d2 = nn.Conv2d(3, STEM_SPARSE_5X5_CHANNELS, kernel_size=3, stride=1, padding=2, dilation=2)

        self.conv_dense_5x5 = nn.Conv2d(3, STEM_DENSE_5X5_CHANNELS, kernel_size=5, stride=1, padding=2, dilation=1)

        self.conv_sparse7_d1 = nn.Conv2d(3, STEM_SPARSE_7X7_CHANNELS, kernel_size=3, stride=1, padding=1, dilation=1)
        self.conv_sparse7_d2 = nn.Conv2d(3, STEM_SPARSE_7X7_CHANNELS, kernel_size=3, stride=1, padding=2, dilation=2)
        self.conv_sparse7_d3 = nn.Conv2d(3, STEM_SPARSE_7X7_CHANNELS, kernel_size=3, stride=1, padding=3, dilation=3)

        self.conv_dense_7x7 = nn.Conv2d(3, STEM_DENSE_7X7_CHANNELS, kernel_size=7, stride=1, padding=3, dilation=1)
        self.conv_1x1 = nn.Conv2d(3, STEM_1x1_CHANNELS, kernel_size=1, bias=False)

        # Register hooks to zero center tap gradients (prevents optimizer from updating them)
        self.conv_sparse5_d2.weight.register_hook(_zero_center_tap_hook)
        self.conv_sparse7_d2.weight.register_hook(_zero_center_tap_hook)
        self.conv_sparse7_d3.weight.register_hook(_zero_center_tap_hook)

        self.stem_norm = nn.GroupNorm(num_groups=GROUPNORM_GROUPS, num_channels=WIDTH)

        self.blocks = nn.ModuleList([
            ResidualBlock(WIDTH, dilation2=TRUNK_DILATION2_SCHEDULE[i], use_se=SE_SCHEDULE[i])
            for i in range(n_blocks)
        ])

        # Policy head: FiLM-based (Feature-wise Linear Modulation)
        # Local feature tower: 1x1 (no reduce) -> 3x3 (reduce) -> 3x3
        self.policy_embed = nn.Conv2d(WIDTH, WIDTH, kernel_size=1, stride=1, padding=0)
        self.policy_norm1 = nn.GroupNorm(num_groups=GROUPNORM_GROUPS, num_channels=WIDTH)
        self.policy_conv1 = nn.Conv2d(WIDTH, POLICY_HEAD_D, kernel_size=3, stride=1, padding=1)
        self.policy_norm2 = nn.GroupNorm(num_groups=GROUPNORM_GROUPS, num_channels=POLICY_HEAD_D)
        self.policy_conv2 = nn.Conv2d(POLICY_HEAD_D, POLICY_HEAD_D, kernel_size=3, stride=1, padding=1)
        self.policy_norm3 = nn.GroupNorm(num_groups=GROUPNORM_GROUPS, num_channels=POLICY_HEAD_D)
        # FiLM parameter generator
        self.policy_film_fc1 = nn.Linear(WIDTH, POLICY_HEAD_MLP_HIDDEN)
        self.policy_film_fc2 = nn.Linear(POLICY_HEAD_MLP_HIDDEN, 2 * POLICY_HEAD_D)

        # FiLM identity initialization: gamma ≈ 1, beta ≈ 0
        # This ensures FiLM(H) ≈ H at training start for stability
        nn.init.zeros_(self.policy_film_fc2.weight)
        with torch.no_grad():
            D = POLICY_HEAD_D
            self.policy_film_fc2.bias[:D].fill_(1.0)   # gamma init -> 1
            self.policy_film_fc2.bias[D:].zero_()      # beta init -> 0

        # Logits output
        self.policy_logits = nn.Conv2d(POLICY_HEAD_D, 1, kernel_size=1, stride=1, padding=0)

        # No-FiLM bypass path (safety mechanism for spatial detail preservation)
        self.policy_logits_bypass = nn.Conv2d(POLICY_HEAD_D, 1, kernel_size=1, bias=False)
        # Use sigmoid to constrain alpha ∈ (0, 1)
        # Initialize alpha_raw to -2.0, so sigmoid(-2.0) ≈ 0.12 (small but non-zero)
        self.policy_bypass_alpha = nn.Parameter(torch.tensor(-2.0))

        # Value head: two 3x3 valid convs (15->13->11), then 1x1 reduction
        self.value_conv1 = nn.Conv2d(WIDTH, WIDTH, kernel_size=3, stride=1, padding=0)
        self.value_norm1 = nn.GroupNorm(num_groups=GROUPNORM_GROUPS, num_channels=WIDTH)
        self.value_conv2 = nn.Conv2d(WIDTH, WIDTH, kernel_size=3, stride=1, padding=0)
        self.value_norm2 = nn.GroupNorm(num_groups=GROUPNORM_GROUPS, num_channels=WIDTH)
        self.value_reduce = nn.Conv2d(WIDTH, VALUE_HEAD_CHANNELS, kernel_size=1, stride=1, padding=0)
        self.value_norm3 = nn.GroupNorm(num_groups=VALUE_HEAD_CHANNELS, num_channels=VALUE_HEAD_CHANNELS)
        self.value_fc1 = nn.Linear(VALUE_HEAD_CHANNELS * 11 * 11, VALUE_HEAD_HIDDEN)
        self.value_fc2 = nn.Linear(VALUE_HEAD_HIDDEN, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input tensor [B, 3, 15, 15] where:
               - Channel 0: Current player's pieces
               - Channel 1: Opponent's pieces
               - Channel 2: Board mask (all 1s within valid board region)

        Returns:
            Tuple of (logits_grid [B, 1, 15, 15], value [B, 1])

        Policy head uses FiLM (Feature-wise Linear Modulation):
            - Local features extracted by 3x conv tower
            - Global state from GAP generates γ, β modulation parameters
            - Modulated features: H' = γ ⊙ H + β
        """
        branch_3x3 = self.conv_3x3(x)
        branch_sparse_5x5 = self.conv_sparse5_d1(x) + self.conv_sparse5_d2(x)
        branch_dense_5x5 = self.conv_dense_5x5(x)
        branch_sparse_7x7 = self.conv_sparse7_d1(x) + self.conv_sparse7_d2(x) + self.conv_sparse7_d3(x)
        branch_dense_7x7 = self.conv_dense_7x7(x)
        branch_1x1 = self.conv_1x1(x)

        x = torch.cat([branch_3x3, branch_dense_5x5, branch_sparse_5x5, branch_dense_7x7, branch_sparse_7x7, branch_1x1], dim=1)
        x = F.silu(x)
        x = self.stem_norm(x)

        for block in self.blocks:
            x = block(x)

        trunk_features = x
        batch_size = trunk_features.size(0)

        # Policy head: FiLM-based (Feature-wise Linear Modulation)
        # 1. Local feature tower
        E0 = F.silu(self.policy_norm1(self.policy_embed(trunk_features)))
        E1 = F.silu(self.policy_norm2(self.policy_conv1(E0)))
        H = F.silu(self.policy_norm3(self.policy_conv2(E1)))  # [B, d_p, 15, 15]

        # 2. Global state extraction
        g = trunk_features.mean(dim=(2, 3))  # [B, WIDTH] - Global Average Pooling

        # 3. FiLM parameter generation
        film_params = F.silu(self.policy_film_fc1(g))  # [B, h]
        film_params = self.policy_film_fc2(film_params)  # [B, 2*d_p]
        gamma, beta = film_params.chunk(2, dim=1)  # [B, d_p], [B, d_p]

        # 4. FiLM modulation: H' = γ ⊙ H + β
        H_modulated = gamma[:, :, None, None] * H + beta[:, :, None, None]  # [B, d_p, 15, 15]

        # 5. Logits output (with bypass path for spatial detail preservation)
        logits_film = self.policy_logits(H_modulated)  # FiLM-modulated path [B, 1, 15, 15]
        logits_bypass = self.policy_logits_bypass(H)   # Pre-FiLM bypass path [B, 1, 15, 15]
        # Use sigmoid to constrain alpha ∈ (0, 1)
        alpha = torch.sigmoid(self.policy_bypass_alpha)
        logits_grid = logits_film + alpha * logits_bypass  # Combined output

        value_x = F.silu(self.value_norm1(self.value_conv1(trunk_features)))
        value_x = F.silu(self.value_norm2(self.value_conv2(value_x)))
        value_x = F.silu(self.value_norm3(self.value_reduce(value_x)))
        value_x = value_x.view(batch_size, -1)
        value_x = F.silu(self.value_fc1(value_x))
        value = torch.tanh(self.value_fc2(value_x))

        return logits_grid, value


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel attention."""

    def __init__(self, channels: int, r: int = 4):
        super().__init__()
        hidden = channels // r
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        y = x.mean(dim=(2, 3))
        y = F.silu(self.fc1(y), inplace=True)
        y = torch.sigmoid(self.fc2(y))
        return x * y.view(b, c, 1, 1)


class ResidualBlock(nn.Module):
    """Pre-activation residual block with configurable dilation and optional SE."""

    def __init__(self, channels: int, dilation2: int = 2, use_se: bool = False):
        super().__init__()
        self.norm1 = nn.GroupNorm(num_groups=GROUPNORM_GROUPS, num_channels=channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=GROUPNORM_GROUPS, num_channels=channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=dilation2, dilation=dilation2)
        self.se = SEBlock(channels) if use_se else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.silu(self.norm1(x), inplace=True)
        out = self.conv1(out)
        out = F.silu(self.norm2(out), inplace=True)
        out = self.conv2(out)
        if self.se is not None:
            out = self.se(out)
        return out + x


def zero_center_taps(model: nn.Module) -> None:
    """
    Zero out center taps in dilated stem convolutions to prevent redundant center contributions.

    Should be called once after model initialization. Gradient hooks registered in the model
    __init__ prevent optimizer updates to these positions, so no need to call after optimizer.step().
    """
    with torch.no_grad():
        model.conv_sparse5_d2.weight[:, :, 1, 1] = 0
        model.conv_sparse7_d2.weight[:, :, 1, 1] = 0
        model.conv_sparse7_d3.weight[:, :, 1, 1] = 0




# ============================================================================
# Inference Helper Functions
# ============================================================================

def obs_batch_to_tensor(obs_list: List[np.ndarray], device: torch.device) -> torch.Tensor:
    """Convert list of observations to batched tensor."""
    return torch.from_numpy(np.stack(obs_list)).float().to(device)


def mask_batch_to_tensor(mask_list: List[np.ndarray], device: torch.device) -> torch.Tensor:
    """Convert list of legal masks to batched tensor."""
    return torch.from_numpy(np.stack(mask_list)).bool().to(device)


def select_action_batch(model: nn.Module, obs_list: List[np.ndarray],
                        mask_list: List[np.ndarray],
                        temperature: float, device: torch.device,
                        deterministic: bool = False) -> Tuple[List[int], List[float]]:
    """
    Select actions for a batch of positions using the policy network.

    Returns:
        Tuple of (actions, log_probs)
    """
    if len(obs_list) == 0:
        return [], []

    with torch.no_grad():
        obs_tensor = obs_batch_to_tensor(obs_list, device)
        mask_tensor = mask_batch_to_tensor(mask_list, device)

        logits_grid, _ = model(obs_tensor)
        logits = logits_grid.squeeze(1)

        logits = logits.masked_fill(~mask_tensor, LOGIT_MASK_VALUE)

        if temperature > 0 and not deterministic:
            logits = logits / temperature

        logits_flat = logits.view(len(obs_list), 225)

        if deterministic or temperature == 0:
            actions = logits_flat.argmax(dim=1).cpu().numpy().tolist()
            dist = Categorical(logits=logits_flat)
            actions_tensor = torch.tensor(actions, dtype=torch.long, device=device)
            log_probs = dist.log_prob(actions_tensor)
            log_probs = torch.clamp(log_probs, min=LOG_PROB_MIN).cpu().numpy().tolist()
        else:
            dist = Categorical(logits=logits_flat)
            actions_tensor = dist.sample()
            actions = actions_tensor.cpu().numpy().tolist()
            log_probs = dist.log_prob(actions_tensor)
            log_probs = torch.clamp(log_probs, min=LOG_PROB_MIN).cpu().numpy().tolist()

    return actions, log_probs


def select_action_batch_eval(model: nn.Module, obs_list: List[np.ndarray],
                              mask_list: List[np.ndarray],
                              temperature: float, device: torch.device,
                              deterministic: bool = False) -> List[int]:
    """
    Select actions for evaluation - no log_prob computation.

    Returns:
        List of action indices
    """
    if len(obs_list) == 0:
        return []

    with torch.no_grad():
        obs_tensor = obs_batch_to_tensor(obs_list, device)
        mask_tensor = mask_batch_to_tensor(mask_list, device)

        logits_grid, _ = model(obs_tensor)
        logits = logits_grid.squeeze(1)
        logits = logits.masked_fill(~mask_tensor, LOGIT_MASK_VALUE)

        if temperature > 0 and not deterministic:
            logits = logits / temperature

        logits_flat = logits.view(len(obs_list), 225)

        if deterministic or temperature == 0:
            actions = logits_flat.argmax(dim=1).cpu().tolist()
        else:
            probs = F.softmax(logits_flat, dim=1)
            actions = torch.multinomial(probs, num_samples=1).squeeze(1).cpu().tolist()

    return actions
