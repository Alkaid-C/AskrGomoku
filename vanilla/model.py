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

N_BLOCKS = 18                     # Number of residual blocks
WIDTH = 96                        # Residual block width
GROUPNORM_GROUPS = 16             # Groups for GroupNorm layers (must divide WIDTH evenly)

# Head architecture
POLICY_HEAD_CHANNELS = 8          # Policy head intermediate channels
VALUE_HEAD_CHANNELS = 4           # Value head intermediate channels
VALUE_HEAD_HIDDEN = 160           # Value head hidden layer size


# ============================================================================
# Neural Network Architecture
# ============================================================================

class GomokuPolicyNet(nn.Module):
    """
    Policy + Value neural network for Gomoku.

    Architecture:
    - Simple stem with single 3x3 convolution
    - Residual trunk with standard residual blocks (no dilation, no SE)
    - Policy head: Conv1x1 to POLICY_HEAD_CHANNELS → GroupNorm → SiLU → flatten → FC to 225
    - Value head: Conv1x1 to VALUE_HEAD_CHANNELS → GroupNorm → SiLU → flatten → FC to VALUE_HEAD_HIDDEN → SiLU → FC to 1 → tanh
    """

    def __init__(self, n_blocks: int):
        super().__init__()

        # === Stem: simple 3x3 convolution ===
        # Input channels: 3 (current player pieces, opponent pieces, board mask)
        self.stem_conv = nn.Conv2d(3, WIDTH, kernel_size=3, stride=1, padding=1)
        self.stem_norm = nn.GroupNorm(num_groups=GROUPNORM_GROUPS, num_channels=WIDTH)

        # === Trunk: standard residual blocks (no dilation, no SE) ===
        self.blocks = nn.ModuleList([
            ResidualBlock(WIDTH)
            for _ in range(n_blocks)
        ])

        # === Policy head: classic design ===
        # Conv1x1 to POLICY_HEAD_CHANNELS → GroupNorm → SiLU → flatten → FC to 225
        self.policy_conv = nn.Conv2d(WIDTH, POLICY_HEAD_CHANNELS, kernel_size=1)
        self.policy_fc = nn.Linear(POLICY_HEAD_CHANNELS * 15 * 15, 225)

        # === Value head: classic design ===
        # Conv1x1 to VALUE_HEAD_CHANNELS → GroupNorm → SiLU → flatten → FC to VALUE_HEAD_HIDDEN → SiLU → FC to 1 → tanh
        self.value_conv = nn.Conv2d(WIDTH, VALUE_HEAD_CHANNELS, kernel_size=1)
        self.value_fc1 = nn.Linear(VALUE_HEAD_CHANNELS * 15 * 15, VALUE_HEAD_HIDDEN)
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
        """
        # Stem
        x = self.stem_conv(x)
        x = self.stem_norm(x)
        x = F.silu(x)

        # Trunk
        for block in self.blocks:
            x = block(x)

        trunk_features = x
        batch_size = trunk_features.size(0)

        # Policy head: Conv1x1 → GroupNorm → SiLU → flatten → FC → reshape
        policy_features = self.policy_conv(trunk_features)  # [B, POLICY_HEAD_CHANNELS, 15, 15]
        policy_features = F.silu(policy_features)
        policy_features = policy_features.view(batch_size, -1)  # [B, POLICY_HEAD_CHANNELS*15*15]
        logits_flat = self.policy_fc(policy_features)  # [B, 225]
        logits_grid = logits_flat.view(batch_size, 1, 15, 15)  # [B, 1, 15, 15]

        # Value head: Conv1x1 → GroupNorm → SiLU → flatten → FC → SiLU → FC → tanh
        value_features = self.value_conv(trunk_features)  # [B, VALUE_HEAD_CHANNELS, 15, 15]
        value_features = F.silu(value_features)
        value_features = value_features.view(batch_size, -1)  # [B, VALUE_HEAD_CHANNELS*15*15]
        value = F.silu(self.value_fc1(value_features))  # [B, VALUE_HEAD_HIDDEN]
        value = torch.tanh(self.value_fc2(value))  # [B, 1]

        return logits_grid, value


class ResidualBlock(nn.Module):
    """Standard pre-activation residual block."""

    def __init__(self, channels: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(num_groups=GROUPNORM_GROUPS, num_channels=channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=GROUPNORM_GROUPS, num_channels=channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.silu(self.norm1(x), inplace=True)
        out = self.conv1(out)
        out = F.silu(self.norm2(out), inplace=True)
        out = self.conv2(out)
        return out + x
