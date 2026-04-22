"""
Gomoku Policy+Value Network Architecture.

Contains ONLY the neural network architecture (GomokuPolicyNet) and architecture-related constants.
This module rarely needs changes unless network structure is modified.
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================================
# Model Architecture Constants
# ============================================================================

N_BLOCKS = 16                     # Number of residual blocks
WIDTH = 48                        # Residual block width
GROUPNORM_GROUPS = 8              # Groups for GroupNorm layers (must divide WIDTH evenly)

# Head architecture
POLICY_WIDTH = 32                 # Policy head width
VALUE_HEAD_CHANNELS = 2           # Value head intermediate channels
VALUE_HEAD_HIDDEN = 96            # Value head hidden layer size


# ============================================================================
# Neural Network Architecture
# ============================================================================

class GomokuPolicyNet(nn.Module):
    """
    Policy + Value neural network for Gomoku.

    Architecture:
    - Simple stem with single 3x3 convolution
    - Residual trunk with standard residual blocks (no dilation, no SE)
    - Policy head: 3x Conv3x3 with GroupNorm+SiLU → Conv1x1 to 1
    - Value head: Conv1x1 to VALUE_HEAD_CHANNELS → SiLU → flatten → FC to VALUE_HEAD_HIDDEN → SiLU → FC to 1 → tanh
    """

    def __init__(self):
        super().__init__()

        # === Stem: simple 3x3 convolution ===
        # Input channels: 3 (current player pieces, opponent pieces, board mask)
        self.stem_conv = nn.Conv2d(3, WIDTH, kernel_size=3, stride=1, padding=1)
        self.stem_norm = nn.GroupNorm(num_groups=GROUPNORM_GROUPS, num_channels=WIDTH)

        # === Trunk: standard residual blocks (no dilation, no SE) ===
        self.blocks = nn.ModuleList([
            ResidualBlock(WIDTH)
            for _ in range(N_BLOCKS)
        ])
        self.trunk_norm = nn.GroupNorm(num_groups=GROUPNORM_GROUPS, num_channels=WIDTH)

        # === Policy head: 3x Conv3x3 with GroupNorm+SiLU → Conv1x1 ===
        self.policy_conv1 = nn.Conv2d(WIDTH, POLICY_WIDTH, kernel_size=3, padding=1)
        self.policy_norm1 = nn.GroupNorm(num_groups=GROUPNORM_GROUPS, num_channels=POLICY_WIDTH)
        self.policy_conv2 = nn.Conv2d(POLICY_WIDTH, POLICY_WIDTH, kernel_size=3, padding=1)
        self.policy_norm2 = nn.GroupNorm(num_groups=GROUPNORM_GROUPS, num_channels=POLICY_WIDTH)
        self.policy_conv3 = nn.Conv2d(POLICY_WIDTH, POLICY_WIDTH, kernel_size=3, padding=1)
        self.policy_norm3 = nn.GroupNorm(num_groups=GROUPNORM_GROUPS, num_channels=POLICY_WIDTH)
        self.policy_out = nn.Conv2d(POLICY_WIDTH, 1, kernel_size=1)

        # === Value head: classic design ===
        # Conv1x1 to VALUE_HEAD_CHANNELS → SiLU → flatten → FC to VALUE_HEAD_HIDDEN → SiLU → FC to 1 → tanh
        self.value_conv = nn.Conv2d(WIDTH, VALUE_HEAD_CHANNELS, kernel_size=1)
        self.value_fc1 = nn.Linear(VALUE_HEAD_CHANNELS * 15 * 15, VALUE_HEAD_HIDDEN)
        self.value_fc2 = nn.Linear(VALUE_HEAD_HIDDEN, 1)

    def _trunk(self, x: torch.Tensor) -> torch.Tensor:
        """Stem + trunk blocks + norm. Returns trunk features."""
        x = self.stem_conv(x)
        x = self.stem_norm(x)
        x = F.silu(x)
        for block in self.blocks:
            x = block(x)
        return F.silu(self.trunk_norm(x))

    def _policy_head(self, trunk_features: torch.Tensor) -> torch.Tensor:
        """Policy head. Returns logits_grid [B, 1, 15, 15]."""
        p = F.silu(self.policy_norm1(self.policy_conv1(trunk_features)))
        p = F.silu(self.policy_norm2(self.policy_conv2(p)))
        p = F.silu(self.policy_norm3(self.policy_conv3(p)))
        return self.policy_out(p)

    def _value_head(self, trunk_features: torch.Tensor) -> torch.Tensor:
        """Value head. Returns value [B, 1]."""
        v = F.silu(self.value_conv(trunk_features))
        v = v.view(v.size(0), -1)
        v = F.silu(self.value_fc1(v))
        return torch.tanh(self.value_fc2(v))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Both heads. Used for training step, gradient probing.

        Returns:
            Tuple of (logits_grid [B, 1, 15, 15], value [B, 1])
        """
        trunk_features = self._trunk(x)
        return self._policy_head(trunk_features), self._value_head(trunk_features)

    def forward_policy_only(self, x: torch.Tensor) -> torch.Tensor:
        """
        Policy head only. Used for self-play and evaluation inference.

        Returns:
            logits_grid [B, 1, 15, 15]
        """
        return self._policy_head(self._trunk(x))

    def forward_value_only(self, x: torch.Tensor) -> torch.Tensor:
        """
        Value head only. Used for GAE computation.

        Returns:
            value [B, 1]
        """
        return self._value_head(self._trunk(x))

    @staticmethod
    def print_topology() -> None:
        """Print model architecture summary."""
        print(f"  Stem: 3x3 conv -> {WIDTH} channels")
        print(f"  Residual blocks: {N_BLOCKS} x {WIDTH} channels (standard pre-activation, no dilation, no SE)")
        print(f"  Policy head: 3x Conv3x3 -> {POLICY_WIDTH}ch (GroupNorm+SiLU) -> Conv1x1 -> 1")
        print(f"  Value head: Conv1x1 -> {VALUE_HEAD_CHANNELS}ch -> GroupNorm -> SiLU -> FC -> {VALUE_HEAD_HIDDEN} -> SiLU -> FC -> 1 -> tanh")


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
