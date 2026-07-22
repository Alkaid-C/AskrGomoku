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

N_SHARED_BLOCKS = 12              # Shared residual blocks (no SE)
N_DUAL_SE_BLOCKS = 6              # Dual-SE residual blocks (policy + value SE streams)
N_BLOCKS = N_SHARED_BLOCKS + N_DUAL_SE_BLOCKS  # Total trunk depth
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
    # Shared blocks
    1, 2, 1, 2, 1, 3,
    1, 2, 1, 2, 1, 3,
    # Dual-SE blocks
    1, 2, 1, 3, 1, 1
]

# Head architecture
POLICY_HEAD_D = 64             # Policy head intermediate channels (d_p)
POLICY_HEAD_GROUPS = 4         # GroupNorm groups for policy head tensors
POLICY_HEAD_NUM_HEADS = 4      # Attention heads in policy head
VALUE_HEAD_C1 = 64             # Layer 1: 1x1 conv, WIDTH -> C1
VALUE_HEAD_C2 = 256            # Layer 2: grouped 3x3 conv, C1 -> C2
VALUE_HEAD_GROUPS = 2          # Groups for grouped conv
VALUE_HEAD_HIDDEN = 256        # FC: C2 -> HIDDEN -> 1


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
    """Policy + Value neural network for Gomoku.

    Architecture is documented in main/CLAUDE.md, "Model Architecture".
    """

    def __init__(self):
        super().__init__()

        # === Stem: line-aware multi-scale design ===
        # The directional branches sum dilated 3x3 convs as an efficient substitute
        # for the 1xN / diagonal kernels PyTorch doesn't natively optimize.

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

        self.shared_blocks = nn.ModuleList([
            ResidualBlock(WIDTH, dilation2=TRUNK_DILATION2_SCHEDULE[i], use_se=False)
            for i in range(N_SHARED_BLOCKS)
        ])
        self.dual_se_blocks = nn.ModuleList([
            DualSEResidualBlock(WIDTH, dilation2=TRUNK_DILATION2_SCHEDULE[N_SHARED_BLOCKS + i])
            for i in range(N_DUAL_SE_BLOCKS)
        ])
        self.trunk_norm_policy = nn.GroupNorm(num_groups=GROUPNORM_GROUPS, num_channels=WIDTH)
        self.trunk_norm_value = nn.GroupNorm(num_groups=GROUPNORM_GROUPS, num_channels=WIDTH)

        # Policy head: dual-attention with conv refinement
        # Stage 1: fused projection (trunk → features + Q/K/V for attention 1)
        self.policy_fused_proj = nn.Conv2d(WIDTH, POLICY_HEAD_D * 4, kernel_size=1, bias=False)
        self.policy_fused_norm = nn.GroupNorm(POLICY_HEAD_GROUPS * 4, POLICY_HEAD_D * 4)

        # Stage 1: first attention
        self.policy_attn1 = AttentionCore(POLICY_HEAD_D, POLICY_HEAD_NUM_HEADS)

        # Stage 2: conv refinement pair (with residual)
        self.policy_refine_conv1 = nn.Conv2d(POLICY_HEAD_D, POLICY_HEAD_D, 3, padding=1, bias=False)
        self.policy_refine_norm1 = nn.GroupNorm(POLICY_HEAD_GROUPS, POLICY_HEAD_D)
        self.policy_refine_conv2 = nn.Conv2d(POLICY_HEAD_D, POLICY_HEAD_D, 3, padding=1, bias=False)
        self.policy_refine_norm2 = nn.GroupNorm(POLICY_HEAD_GROUPS, POLICY_HEAD_D)

        # Stage 3: Q/K/V projection for attention 2
        self.policy_qkv_proj = nn.Conv2d(POLICY_HEAD_D, POLICY_HEAD_D * 3, kernel_size=1, bias=False)
        self.policy_qkv_norm = nn.GroupNorm(POLICY_HEAD_GROUPS * 3, POLICY_HEAD_D * 3)

        # Stage 3: second attention
        self.policy_attn2 = AttentionCore(POLICY_HEAD_D, POLICY_HEAD_NUM_HEADS)

        # Stage 4: final refinement + logits
        self.policy_final_conv = nn.Conv2d(POLICY_HEAD_D, POLICY_HEAD_D, 3, padding=1, bias=False)
        self.policy_final_norm = nn.GroupNorm(POLICY_HEAD_GROUPS, POLICY_HEAD_D)

        # Logits output
        self.policy_logits = nn.Conv2d(POLICY_HEAD_D, 1, kernel_size=1, stride=1, padding=0)

        # Value head: 1x1 projection + grouped 3x3 expansion + LSE/GAP mix + FC
        self.value_conv1 = nn.Conv2d(WIDTH, VALUE_HEAD_C1, kernel_size=1, bias=False)
        self.value_norm1 = nn.GroupNorm(num_groups=VALUE_HEAD_GROUPS, num_channels=VALUE_HEAD_C1)
        self.value_norm2 = nn.GroupNorm(num_groups=VALUE_HEAD_GROUPS, num_channels=VALUE_HEAD_C1)
        self.value_conv2 = nn.Conv2d(VALUE_HEAD_C1, VALUE_HEAD_C2, kernel_size=3, stride=1, padding=1, groups=VALUE_HEAD_GROUPS)
        self.value_pool_temp_raw = nn.Parameter(torch.zeros(VALUE_HEAD_C2))  # softplus → per-channel temperature for log-mean-exp
        self.value_norm3 = nn.LayerNorm(VALUE_HEAD_C2, bias=False)
        self.value_fc1 = nn.Linear(VALUE_HEAD_C2, VALUE_HEAD_HIDDEN)
        self.value_fc2 = nn.Linear(VALUE_HEAD_HIDDEN, 1)

        # Zero center taps for d>1 directional convolutions (d1 already covers center).
        self._zero_center_taps()
        self.register_load_state_dict_post_hook(lambda m, _: m._zero_center_taps())

    def _stem_and_shared_trunk(self, x: torch.Tensor) -> torch.Tensor:
        """Stem + shared trunk (the pre-fork blocks). Returns features for branching."""
        branch_3x3 = self.conv_3x3(x)
        branch_directional_5x5 = self.conv_directional5_d1(x) + self.conv_directional5_d2(x)
        branch_full_5x5 = self.conv_full5(x)
        branch_directional_7x7 = self.conv_directional7_d1(x) + self.conv_directional7_d2(x) + self.conv_directional7_d3(x)
        branch_full_7x7 = self.conv_full7(x)
        branch_1x1 = self.conv_1x1(x)

        x = torch.cat([branch_3x3, branch_full_5x5, branch_directional_5x5, branch_full_7x7, branch_directional_7x7, branch_1x1], dim=1)
        x = self.stem_norm(x)
        x = F.silu(x)

        for block in self.shared_blocks:
            x = block(x)

        return x

    def _policy_head(self, trunk_features: torch.Tensor) -> torch.Tensor:
        """Policy head: dual-attention with conv refinement. Returns logits_grid [B, 1, 15, 15]."""
        # Stage 1: fused projection → split into features + Q/K/V
        fused = self.policy_fused_norm(self.policy_fused_proj(trunk_features))
        x_1_raw, q1, k1, v1 = torch.split(fused, POLICY_HEAD_D, dim=1)
        x_1 = F.silu(x_1_raw)

        # Stage 1: first attention (residual)
        x_2 = x_1 + self.policy_attn1(q1, k1, v1)

        # Stage 2: conv refinement pair (residual)
        x_3 = F.silu(self.policy_refine_norm1(self.policy_refine_conv1(x_2)))
        x_4 = F.silu(self.policy_refine_norm2(self.policy_refine_conv2(x_3)))
        x_5 = x_2 + x_4

        # Stage 3: second attention (residual)
        qkv = self.policy_qkv_norm(self.policy_qkv_proj(x_5))
        q2, k2, v2 = torch.split(qkv, POLICY_HEAD_D, dim=1)
        x_6 = x_5 + self.policy_attn2(q2, k2, v2)

        # Stage 4: final refinement → logits
        x_7 = F.silu(self.policy_final_norm(self.policy_final_conv(x_6)))
        return self.policy_logits(x_7)

    def _value_head(self, trunk_features: torch.Tensor) -> torch.Tensor:
        """Value head: 1x1 + grouped 3x3 + temp-scaled log-mean-exp + FC. Returns value [B, 1]."""
        vx = F.silu(self.value_norm1(self.value_conv1(trunk_features)))
        vx = F.silu(self.value_conv2(self.value_norm2(vx)))
        # Per-channel temperature log-mean-exp: τ→0 = max, τ→∞ = mean
        tau = F.softplus(self.value_pool_temp_raw)                        # [C], positive
        vx = tau * (torch.logsumexp(vx / tau[None, :, None, None], dim=(2, 3)) - 5.4161)  # log(225)=5.4161
        vx = F.silu(self.value_fc1(self.value_norm3(vx)))
        value = torch.tanh(self.value_fc2(vx))
        return value

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Both heads: shared trunk -> fork -> dual-SE -> policy head + value head.
        Used for training step, web play, gradient probing.

        Returns:
            Tuple of (logits_grid [B, 1, 15, 15], value [B, 1])
        """
        x = self._stem_and_shared_trunk(x)

        # Fork into policy and value streams
        x_policy = x
        x_value = x
        for block in self.dual_se_blocks:
            x_policy, x_value = block(x_policy, x_value)

        x_policy = F.silu(self.trunk_norm_policy(x_policy))
        x_value = F.silu(self.trunk_norm_value(x_value))

        logits_grid = self._policy_head(x_policy)
        value = self._value_head(x_value)
        return logits_grid, value

    def forward_policy_only(self, x: torch.Tensor) -> torch.Tensor:
        """
        Policy head only: shared trunk -> policy SE stream -> policy head.
        Used for self-play and evaluation inference.

        Returns:
            logits_grid [B, 1, 15, 15]
        """
        x = self._stem_and_shared_trunk(x)

        for block in self.dual_se_blocks:
            x = block.forward_policy(x)

        x = F.silu(self.trunk_norm_policy(x))
        return self._policy_head(x)

    def forward_value_only(self, x: torch.Tensor) -> torch.Tensor:
        """
        Value head only: shared trunk -> value SE stream -> value head.
        Used for GAE computation.

        Returns:
            value [B, 1]
        """
        x = self._stem_and_shared_trunk(x)

        for block in self.dual_se_blocks:
            x = block.forward_value(x)

        x = F.silu(self.trunk_norm_value(x))
        return self._value_head(x)

    def _zero_center_taps(self) -> None:
        """Zero center taps in d>1 directional stem convolutions."""
        with torch.no_grad():
            self.conv_directional5_d2.weight[:, :, 1, 1] = 0
            self.conv_directional7_d2.weight[:, :, 1, 1] = 0
            self.conv_directional7_d3.weight[:, :, 1, 1] = 0

    @staticmethod
    def print_topology() -> None:
        """Print model architecture summary."""
        print("  Stem (dilated design):")
        print(f"    - 3x3: {STEM_3X3_CHANNELS}ch")
        print(f"    - 5x5 directional (d1+d2): {STEM_DIRECTIONAL_5X5_CHANNELS}ch, 5x5 full: {STEM_FULL_5X5_CHANNELS}ch")
        print(f"    - 7x7 directional (d1+d2+d3): {STEM_DIRECTIONAL_7X7_CHANNELS}ch, 7x7 full: {STEM_FULL_7X7_CHANNELS}ch")
        print(f"    - Total: {WIDTH} channels (center taps zeroed for d>1)")
        print(f"  Residual blocks: {N_BLOCKS} total ({N_SHARED_BLOCKS} shared + {N_DUAL_SE_BLOCKS} dual-SE) x {WIDTH} channels")
        print(f"    - Dilation schedule (conv2): {TRUNK_DILATION2_SCHEDULE}")
        print("    - Shared blocks: no SE | Dual-SE blocks: independent policy/value SE gates")
        print(f"  Policy head: {WIDTH} -> dual-attention ({POLICY_HEAD_D}ch, 2x attn + conv refine) -> 225")
        print(f"  Value head: {WIDTH} -> 1x1 {VALUE_HEAD_C1} -> grouped 3x3 {VALUE_HEAD_C2}(g={VALUE_HEAD_GROUPS}) -> log-mean-exp(τ) -> fc{VALUE_HEAD_HIDDEN} -> 1")


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


class AttentionCore(nn.Module):
    """Multi-head self-attention with dihedral-symmetric 2D relative positional bias.

    Reusable module for the policy head. Does NOT add a residual connection —
    the caller handles that.
    """

    def __init__(self, channels: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5

        self.out_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        nn.init.zeros_(self.out_proj.weight)

        self.relative_bias_table = nn.Parameter(torch.zeros(num_heads, 120))

        # Precompute dihedral-symmetric relative position index buffer
        coords = torch.arange(15)
        y, x = torch.meshgrid(coords, coords, indexing='ij')
        coords_flat = torch.stack([y.flatten(), x.flatten()], dim=-1)   # [225, 2]
        rel = coords_flat[:, None, :] - coords_flat[None, :, :]         # [225, 225, 2]
        abs_rel = rel.abs()
        big = abs_rel.max(dim=-1).values     # [225, 225]
        small = abs_rel.min(dim=-1).values   # [225, 225]
        index = big * (big + 1) // 2 + small  # [225, 225], values in [0, 119]
        self.register_buffer("rel_index", index)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        B, C, H, W = q.shape
        N = H * W  # 225

        # Reshape to [B, heads, N, head_dim]
        q = q.reshape(B, self.num_heads, self.head_dim, N).permute(0, 1, 3, 2)
        k = k.reshape(B, self.num_heads, self.head_dim, N).permute(0, 1, 3, 2)
        v = v.reshape(B, self.num_heads, self.head_dim, N).permute(0, 1, 3, 2)

        # Scaled dot-product attention + relative positional bias
        scores = (q @ k.transpose(-2, -1)) * self.scale   # [B, heads, N, N]
        bias = self.relative_bias_table[:, self.rel_index]  # [heads, 225, 225]
        scores = scores + bias

        attn = torch.softmax(scores, dim=-1)
        out = attn @ v  # [B, heads, N, head_dim]

        # Reshape back to [B, C, H, W]
        out = out.permute(0, 1, 3, 2).reshape(B, C, H, W)
        out = self.out_proj(out)
        return out


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


class DualSEResidualBlock(nn.Module):
    """Pre-activation residual block with dual SE gates for policy/value streams.

    Shared convolution weights process both streams independently, then separate
    SE modules gate channels differently for each head. Over multiple blocks the
    streams progressively diverge through iterative differential gating.
    """

    def __init__(self, channels: int, dilation2: int):
        super().__init__()
        # Norms are separate per stream (cheap, allows divergence);
        # convs are shared (expensive, parameter-efficient).
        self.norm1_policy = nn.GroupNorm(num_groups=GROUPNORM_GROUPS, num_channels=channels)
        self.norm1_value = nn.GroupNorm(num_groups=GROUPNORM_GROUPS, num_channels=channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm2_policy = nn.GroupNorm(num_groups=GROUPNORM_GROUPS, num_channels=channels)
        self.norm2_value = nn.GroupNorm(num_groups=GROUPNORM_GROUPS, num_channels=channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=dilation2, dilation=dilation2)
        self.se_policy = SEBlock(channels)
        self.se_value = SEBlock(channels)

    def _conv_path(self, x: torch.Tensor, norm1: nn.Module, norm2: nn.Module) -> torch.Tensor:
        out = F.silu(norm1(x), inplace=True)
        out = self.conv1(out)
        out = F.silu(norm2(out), inplace=True)
        out = self.conv2(out)
        return out

    def forward(self, x_policy: torch.Tensor, x_value: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Both streams: separate norms + shared convs + independent SE + skip."""
        out_p = self._conv_path(x_policy, self.norm1_policy, self.norm2_policy)
        out_v = self._conv_path(x_value, self.norm1_value, self.norm2_value)
        return self.se_policy(out_p) + x_policy, self.se_value(out_v) + x_value

    def forward_policy(self, x: torch.Tensor) -> torch.Tensor:
        """Policy stream only: policy norms + shared convs + policy SE + skip."""
        out = self._conv_path(x, self.norm1_policy, self.norm2_policy)
        return self.se_policy(out) + x

    def forward_value(self, x: torch.Tensor) -> torch.Tensor:
        """Value stream only: value norms + shared convs + value SE + skip."""
        out = self._conv_path(x, self.norm1_value, self.norm2_value)
        return self.se_value(out) + x
