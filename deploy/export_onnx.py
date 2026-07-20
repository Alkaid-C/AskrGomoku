#!/usr/bin/env python3
"""
ONNX Export Script for Gomoku Model

Converts PyTorch checkpoint (.pt) to ONNX format optimized for web deployment.

Usage:
    python3 export_onnx.py --input runs/exp1/checkpoint_update_1000.pt --output model.onnx

Requirements:
    pip install torch onnx onnxruntime
"""

import argparse
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
from model import GomokuPolicyNet


class DecomposedGroupNorm(nn.Module):
    """GroupNorm re-expressed with primitive tensor ops.

    onnxruntime-web's WebGPU EP has no GroupNormalization kernel; every such
    node forces a GPU->CPU->GPU round trip at runtime, which dominated
    inference latency. Emitting the same computation as
    Reshape/ReduceMean/Sub/Mul/Sqrt/Add keeps the whole graph on the GPU.
    """
    def __init__(self, gn: nn.GroupNorm):
        super().__init__()
        assert gn.affine, "affine-less GroupNorm not handled"
        self.num_groups = gn.num_groups
        self.eps = gn.eps
        self.weight = gn.weight
        self.bias = gn.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.shape
        xg = x.reshape(n, self.num_groups, -1)
        mean = xg.mean(dim=2, keepdim=True)
        var = ((xg - mean) ** 2).mean(dim=2, keepdim=True)
        xg = (xg - mean) / torch.sqrt(var + self.eps)
        x = xg.reshape(n, c, h, w)
        return x * self.weight.view(1, c, 1, 1) + self.bias.view(1, c, 1, 1)


def decompose_groupnorms(module: nn.Module) -> int:
    """Recursively replace every nn.GroupNorm with DecomposedGroupNorm."""
    count = 0
    for name, child in module.named_children():
        if isinstance(child, nn.GroupNorm):
            setattr(module, name, DecomposedGroupNorm(child))
            count += 1
        else:
            count += decompose_groupnorms(child)
    return count


class GomokuModelForExport(nn.Module):
    """
    Wrapper that outputs raw policy logits (no softmax, no temperature).

    Input is [2, 15, 15] (no batch dimension - single inference only).
    Internally adds batch dimension and constant mask channel before feeding to model.

    Outputs:
        - policy_logits: [225] raw logits (flat vector)
        - value: [1] value estimation (scalar)

    Temperature and softmax are handled on the JS side after masking illegal moves,
    avoiding precision issues from softmaxing over illegal positions with large logits.
    """
    def __init__(self, base_model: GomokuPolicyNet):
        super().__init__()
        self.base_model = base_model

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [2, 15, 15] board state (channel 0: current player, channel 1: opponent)

        Returns:
            policy_logits: [225] raw policy logits (flat vector)
            value: [1] win probability estimation
        """
        # Add batch dimension: [2, 15, 15] → [1, 2, 15, 15]
        x = x.unsqueeze(0)

        # Add constant mask channel: [1, 2, 15, 15] → [1, 3, 15, 15]
        batch_size, _, height, width = x.shape
        mask_channel = torch.ones(batch_size, 1, height, width, dtype=x.dtype, device=x.device)
        x_with_mask = torch.cat([x, mask_channel], dim=1)  # [1, 3, 15, 15]

        logits_grid, value = self.base_model(x_with_mask)  # [1, 1, 15, 15], [1, 1]

        # Flatten logits: [1, 1, 15, 15] → [225]
        logits_flat = logits_grid.view(-1)

        # Remove batch dimension from value: [1, 1] → [1]
        value_scalar = value.squeeze(0).squeeze(0)

        return logits_flat, value_scalar


def load_checkpoint(checkpoint_path: str) -> GomokuPolicyNet:
    """Load model from PyTorch checkpoint."""
    print(f"Loading checkpoint: {checkpoint_path}")

    model = GomokuPolicyNet()
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded: {n_params:,} parameters ({n_params/1e6:.2f}M)")

    return model


def export_to_onnx(model: nn.Module, output_path: str):
    """Export model to ONNX with raw logits (no softmax, no temperature)."""
    print("\nExporting to ONNX (opset=21, raw logits)...")

    n_gn = decompose_groupnorms(model)
    print(f"Decomposed {n_gn} GroupNorm layers into primitive ops (WebGPU compatibility)")

    # Wrap model for export (adds batch dim + mask channel)
    wrapped_model = GomokuModelForExport(model)
    wrapped_model.eval()

    # Dummy input: [2, 15, 15] (no batch dimension)
    dummy_input = torch.randn(2, 15, 15)

    # external_data defaults to True, which splits the weights into a
    # <name>.onnx.data sidecar and leaves only relative-path references behind.
    # The browser loads a model as one fetched ArrayBuffer with no filesystem to
    # resolve those paths against, so the export must stay self-contained.
    torch.onnx.export(
        wrapped_model,
        dummy_input,
        output_path,
        opset_version=21,  # Required for correct GroupNormalization
        input_names=["board_state"],
        output_names=["policy_logits", "value"],
        dynamic_axes=None,  # Fixed batch=1
        do_constant_folding=True,
        export_params=True,
        external_data=False,
    )

    print(f"✓ ONNX exported as single file: {output_path}")
    return wrapped_model, dummy_input


def verify_onnx(onnx_path: str, torch_model: nn.Module, dummy_input: torch.Tensor):
    """Verify ONNX output matches PyTorch output."""
    print("\nVerifying ONNX model...")

    # Check ONNX validity
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("✓ ONNX model is valid")

    # Compare outputs
    with torch.no_grad():
        torch_logits, torch_value = torch_model(dummy_input)

    session = ort.InferenceSession(onnx_path)
    onnx_outputs = session.run(None, {"board_state": dummy_input.numpy()})
    onnx_logits, onnx_value = onnx_outputs

    # Compute differences
    logits_diff = np.abs(torch_logits.numpy() - onnx_logits).max()
    value_diff = np.abs(torch_value.numpy() - onnx_value).max()

    print(f"✓ Max difference - Logits: {logits_diff:.2e}, Value: {value_diff:.2e}")

    if logits_diff < 1e-5 and value_diff < 1e-5:
        print("✓ Verification passed (difference < 1e-5)")
    else:
        print("⚠ Warning: Larger than expected difference (may still be acceptable)")


def main():
    parser = argparse.ArgumentParser(
        description="Export Gomoku PyTorch model to ONNX (raw logits, no softmax)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python3 export_onnx.py --input runs/exp1/checkpoint_update_1000.pt --output model.onnx

Temperature and softmax are applied on the JS side after masking illegal moves.
        """
    )
    parser.add_argument("--input", required=True, help="Input PyTorch checkpoint (.pt)")
    parser.add_argument("--output", required=True, help="Output ONNX file (.onnx)")

    args = parser.parse_args()

    # Validate inputs
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {args.input}")
        return

    # Load model
    model = load_checkpoint(args.input)

    # Export to ONNX
    wrapped_model, dummy_input = export_to_onnx(model, args.output)

    # Verify correctness
    verify_onnx(args.output, wrapped_model, dummy_input)

    # Report file size
    output_path = Path(args.output)
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print("\n✓ Export complete!")
    print(f"  Output: {args.output}")
    print(f"  Size: {size_mb:.2f} MB")
    print("  Format: raw logits (apply temperature + softmax on JS side)")
    print("\nReady for deployment with onnxruntime-web (WASM backend)")


if __name__ == "__main__":
    main()
