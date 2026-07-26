"""
KLNet — regresses a position's MCTS improvement gap.

Input `[B, 5, 15, 15]`: the three observation planes, a log-prior plane, and the
network value broadcast over the board. Output is a single unbounded scalar per
position, the regression target `log(KL + LOG_EPSILON)`.

The architecture reuses `main/model.py`'s design ideas — line-aware multi-scale
dilated stem, pre-activation residual blocks with SE, temperature-scaled
log-mean-exp pooling into an FC head — but the classes are copied rather than
imported, so this study can pick its own width and GroupNorm grouping without
touching `model.py` (which is symlinked from `main/` and shared with the RL
pipeline).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import (
    NET_BLOCKS,
    NET_DILATION_SCHEDULE,
    NET_GN_GROUPS,
    NET_HEAD_HIDDEN,
    NET_SE_REDUCTION,
    NET_STEM_3X3_CHANNELS,
    NET_STEM_DIRECTIONAL_5X5_CHANNELS,
    NET_STEM_DIRECTIONAL_7X7_CHANNELS,
    NET_WIDTH,
)

INPUT_CHANNELS = 5
LOG_BOARD_SIZE = math.log(225.0)  # board-fixed: ln(15*15)


def _zero_center_tap_hook(grad: torch.Tensor) -> torch.Tensor:
    """Zero gradients at the center tap (1,1) of dilated 3x3 kernels.

    In the directional branches the d1 conv already covers the center position,
    so higher-dilation convs must keep their center weight at zero to avoid
    double-counting. This hook stops the optimizer reintroducing it.
    """
    grad = grad.clone()
    grad[:, :, 1, 1] = 0
    return grad


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel attention."""

    def __init__(self, channels: int):
        super().__init__()
        hidden = channels // NET_SE_REDUCTION
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
        self.norm1 = nn.GroupNorm(num_groups=NET_GN_GROUPS, num_channels=channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=NET_GN_GROUPS, num_channels=channels)
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


class KLNet(nn.Module):
    """Scalar regressor over a Gomoku position + the policy net's prior/value."""

    def __init__(self):
        super().__init__()

        # === Stem: line-aware multi-scale, scaled down from model.py ===
        self.conv_3x3 = nn.Conv2d(INPUT_CHANNELS, NET_STEM_3X3_CHANNELS, kernel_size=3, padding=1)

        # Directional 5x5: d1 covers the inner 3x3, d2 extends to range 2 along 4 directions.
        self.conv_directional5_d1 = nn.Conv2d(INPUT_CHANNELS, NET_STEM_DIRECTIONAL_5X5_CHANNELS, kernel_size=3, padding=1, dilation=1)
        self.conv_directional5_d2 = nn.Conv2d(INPUT_CHANNELS, NET_STEM_DIRECTIONAL_5X5_CHANNELS, kernel_size=3, padding=2, dilation=2)

        # Directional 7x7: d1 + d2 + d3 cover 4 directions out to range 3.
        self.conv_directional7_d1 = nn.Conv2d(INPUT_CHANNELS, NET_STEM_DIRECTIONAL_7X7_CHANNELS, kernel_size=3, padding=1, dilation=1)
        self.conv_directional7_d2 = nn.Conv2d(INPUT_CHANNELS, NET_STEM_DIRECTIONAL_7X7_CHANNELS, kernel_size=3, padding=2, dilation=2)
        self.conv_directional7_d3 = nn.Conv2d(INPUT_CHANNELS, NET_STEM_DIRECTIONAL_7X7_CHANNELS, kernel_size=3, padding=3, dilation=3)

        self.conv_directional5_d2.weight.register_hook(_zero_center_tap_hook)
        self.conv_directional7_d2.weight.register_hook(_zero_center_tap_hook)
        self.conv_directional7_d3.weight.register_hook(_zero_center_tap_hook)

        self.stem_norm = nn.GroupNorm(num_groups=NET_GN_GROUPS, num_channels=NET_WIDTH)

        self.blocks = nn.ModuleList([
            ResidualBlock(NET_WIDTH, dilation2=NET_DILATION_SCHEDULE[i], use_se=True)
            for i in range(NET_BLOCKS)
        ])
        # The blocks are pre-activation, so the trunk output still needs a
        # norm + activation before it is pooled.
        self.trunk_norm = nn.GroupNorm(num_groups=NET_GN_GROUPS, num_channels=NET_WIDTH)

        # === Head: pool the board to a vector, then FC to a scalar ===
        self.pool_temp_raw = nn.Parameter(torch.zeros(NET_WIDTH))  # softplus -> per-channel temperature
        self.head_norm = nn.LayerNorm(NET_WIDTH, bias=False)
        self.head_fc1 = nn.Linear(NET_WIDTH, NET_HEAD_HIDDEN)
        self.head_fc2 = nn.Linear(NET_HEAD_HIDDEN, 1)

        self._zero_center_taps()
        self.register_load_state_dict_post_hook(lambda m, _: m._zero_center_taps())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, 5, 15, 15] -> [B] predicted log(KL + eps)."""
        branch_3x3 = self.conv_3x3(x)
        branch_directional_5x5 = self.conv_directional5_d1(x) + self.conv_directional5_d2(x)
        branch_directional_7x7 = (
            self.conv_directional7_d1(x) + self.conv_directional7_d2(x) + self.conv_directional7_d3(x)
        )

        h = torch.cat([branch_3x3, branch_directional_5x5, branch_directional_7x7], dim=1)
        h = F.silu(self.stem_norm(h))

        for block in self.blocks:
            h = block(h)
        h = F.silu(self.trunk_norm(h))

        # Per-channel temperature log-mean-exp over the board: tau->0 = max, tau->inf = mean.
        tau = F.softplus(self.pool_temp_raw)
        pooled = tau * (torch.logsumexp(h / tau[None, :, None, None], dim=(2, 3)) - LOG_BOARD_SIZE)

        out = F.silu(self.head_fc1(self.head_norm(pooled)))
        return self.head_fc2(out).squeeze(-1)

    def _zero_center_taps(self) -> None:
        """Zero center taps in the d>1 directional stem convolutions."""
        with torch.no_grad():
            self.conv_directional5_d2.weight[:, :, 1, 1] = 0
            self.conv_directional7_d2.weight[:, :, 1, 1] = 0
            self.conv_directional7_d3.weight[:, :, 1, 1] = 0
