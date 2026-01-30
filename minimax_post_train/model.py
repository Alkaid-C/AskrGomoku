"""
Gomoku Policy+Value Network Architecture.

Contains ONLY the neural network architecture (GomokuPolicyNet) and architecture-related constants.
This module rarely needs changes unless network structure is modified.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


# ============================================================================
# Model Architecture Constants
# ============================================================================

N_BLOCKS = 16                     # Number of residual blocks
WIDTH = 96                        # Residual block width (must equal sum of all stem channels)
STEM_3X3_CHANNELS = 6 * 6         # 3x3 convolution channels in stem
STEM_DIRECTIONAL_5X5_CHANNELS = 3 * 6  # Directional 5x5 (4-line kernel via dilated 3x3 sum) channels in stem
STEM_FULL_5X5_CHANNELS = 2 * 6     # Full 5x5 convolution channels in stem
STEM_DIRECTIONAL_7X7_CHANNELS = 3 * 6  # Directional 7x7 (4-line kernel via dilated 3x3 sum) channels in stem
STEM_FULL_7X7_CHANNELS = 1 * 6     # Full 7x7 convolution channels in stem
STEM_1x1_CHANNELS = 1 * 6
GROUPNORM_GROUPS = 16             # Groups for GroupNorm layers (must divide WIDTH evenly)
SE_REDUCTION = 4                  # Squeeze-and-Excitation channel reduction ratio

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

# Head architecture
POLICY_HEAD_D = 128            # Policy head intermediate channels (d_p)
POLICY_HEAD_MLP_HIDDEN = 256   # Policy head global MLP hidden size (h)
VALUE_HEAD_C1 = 128            # Layer 1: 96 -> 128
VALUE_HEAD_C2_SPLIT = 128      # Layer 2: 128 -> 128(d1) + 128(d2) = 256
VALUE_HEAD_HIDDEN = 256        # FC: 256 -> 256 -> 1


# ============================================================================
# Neural Network Architecture
# ============================================================================

def _zero_center_tap_hook(grad: torch.Tensor) -> torch.Tensor:
    """Zero gradients at center tap (1,1) of dilated 3x3 kernels.

    In the directional branches, the d1 conv already covers the center position.
    Higher-dilation convs (d2, d3) must have their center weight permanently zeroed
    to avoid double-counting. This hook prevents the optimizer from reintroducing it.
    """
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

    def __init__(self, n_blocks: int):
        super().__init__()

        # === Stem: line-aware multi-scale design ===
        # Gomoku threats are strictly along 4 directions (horizontal, vertical,
        # diagonal, anti-diagonal). The "directional" branches use sums of dilated
        # 3x3 convolutions to build kernels that cover exactly these 4 line
        # directions at range 5 or 7 — an efficient substitute for non-standard
        # 1xN / diagonal kernels that PyTorch doesn't natively optimize.
        # The "full" branches use standard NxN kernels for complete spatial coverage.
        # Input channels: 3 (current player pieces, opponent pieces, board mask)

        self.conv_3x3 = nn.Conv2d(3, STEM_3X3_CHANNELS, kernel_size=3, stride=1, padding=1, dilation=1)

        # Directional 5x5: d1 covers inner 3x3, d2 extends to range-2 along 4 directions
        self.conv_directional5_d1 = nn.Conv2d(3, STEM_DIRECTIONAL_5X5_CHANNELS, kernel_size=3, stride=1, padding=1, dilation=1)
        self.conv_directional5_d2 = nn.Conv2d(3, STEM_DIRECTIONAL_5X5_CHANNELS, kernel_size=3, stride=1, padding=2, dilation=2)

        # Full 5x5: standard convolution for complete 5x5 spatial coverage
        self.conv_full5 = nn.Conv2d(3, STEM_FULL_5X5_CHANNELS, kernel_size=5, stride=1, padding=2, dilation=1)

        # Directional 7x7: d1 + d2 + d3 cover 4 directions out to range 3
        self.conv_directional7_d1 = nn.Conv2d(3, STEM_DIRECTIONAL_7X7_CHANNELS, kernel_size=3, stride=1, padding=1, dilation=1)
        self.conv_directional7_d2 = nn.Conv2d(3, STEM_DIRECTIONAL_7X7_CHANNELS, kernel_size=3, stride=1, padding=2, dilation=2)
        self.conv_directional7_d3 = nn.Conv2d(3, STEM_DIRECTIONAL_7X7_CHANNELS, kernel_size=3, stride=1, padding=3, dilation=3)

        # Full 7x7: standard convolution for complete 7x7 spatial coverage
        self.conv_full7 = nn.Conv2d(3, STEM_FULL_7X7_CHANNELS, kernel_size=7, stride=1, padding=3, dilation=1)
        self.conv_1x1 = nn.Conv2d(3, STEM_1x1_CHANNELS, kernel_size=1, bias=False)

        # Zero center tap gradients for d>1 convs: the center position is already
        # covered by d1, so higher dilations must not duplicate it.
        self.conv_directional5_d2.weight.register_hook(_zero_center_tap_hook)
        self.conv_directional7_d2.weight.register_hook(_zero_center_tap_hook)
        self.conv_directional7_d3.weight.register_hook(_zero_center_tap_hook)

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

        # Value head: conv & channel expansion + GAP + FC
        #Conv Layer 1
        self.value_conv1 = nn.Conv2d(WIDTH, VALUE_HEAD_C1, kernel_size=3, stride=1, padding=1, bias=False)
        self.value_norm1 = nn.GroupNorm(num_groups=GROUPNORM_GROUPS, num_channels=VALUE_HEAD_C1)
        # Conv Layer 2
        # Branch A: Dilation 1 (Dense)
        self.value_conv2a = nn.Conv2d(VALUE_HEAD_C1, VALUE_HEAD_C2_SPLIT, kernel_size=3, stride=1, padding=1, dilation=1, bias=False)
        # Branch B: Dilation 2 (Sparse/Wide) - Padding=2 for Dilation=2
        self.value_conv2b = nn.Conv2d(VALUE_HEAD_C1, VALUE_HEAD_C2_SPLIT, kernel_size=3, stride=1, padding=2, dilation=2, bias=False)
        # Norm after concat
        self.value_norm2 = nn.GroupNorm(num_groups=GROUPNORM_GROUPS, num_channels=VALUE_HEAD_C2_SPLIT * 2)
        # FC Layers (Input -> Hidden -> Out 1)
        self.value_fc1 = nn.Linear(VALUE_HEAD_C2_SPLIT * 2, VALUE_HEAD_HIDDEN)
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
        branch_directional_5x5 = self.conv_directional5_d1(x) + self.conv_directional5_d2(x)
        branch_full_5x5 = self.conv_full5(x)
        branch_directional_7x7 = self.conv_directional7_d1(x) + self.conv_directional7_d2(x) + self.conv_directional7_d3(x)
        branch_full_7x7 = self.conv_full7(x)
        branch_1x1 = self.conv_1x1(x)

        x = torch.cat([branch_3x3, branch_full_5x5, branch_directional_5x5, branch_full_7x7, branch_directional_7x7, branch_1x1], dim=1)
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

        # Value Head
        # 1. Layer 1 (WIDTH -> VALUE_HEAD_C1)
        vx = F.silu(self.value_norm1(self.value_conv1(trunk_features)))
        # 2. Layer 2 Split (VALUE_HEAD_C1 -> VALUE_HEAD_C2_SPLIT + VALUE_HEAD_C2_SPLIT)
        v_d1 = self.value_conv2a(vx)
        v_d2 = self.value_conv2b(vx)
        vx = torch.cat([v_d1, v_d2], dim=1) # Concat
        vx = F.silu(self.value_norm2(vx))
        # 3. GAP
        vx = vx.mean(dim=(2, 3))
        # 4. Dense
        vx = F.silu(self.value_fc1(vx))
        value = torch.tanh(self.value_fc2(vx))

        return logits_grid, value


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel attention."""

    def __init__(self, channels: int):
        super().__init__()
        hidden = channels // SE_REDUCTION
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

    def __init__(self, channels: int, dilation2: int, use_se: bool):
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
    Zero out center taps in directional stem convolutions (d>1).

    The center position is already covered by d1; higher-dilation convs must not
    duplicate it. Called once after model init. Gradient hooks in __init__ keep
    these weights at zero during training, so no need to call after optimizer.step().
    """
    with torch.no_grad():
        model.conv_directional5_d2.weight[:, :, 1, 1] = 0
        model.conv_directional7_d2.weight[:, :, 1, 1] = 0
        model.conv_directional7_d3.weight[:, :, 1, 1] = 0
