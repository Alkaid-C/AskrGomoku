#!/usr/bin/env python3
"""
ONNX Export Script for Gomoku Model

Converts PyTorch checkpoint (.pt) to ONNX format optimized for web deployment.

Usage:
    python3 export_onnx.py --input runs/exp1/checkpoint_update_1000.pt --output model.onnx --temp 0.5

Requirements:
    pip install torch onnx onnxruntime
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import onnx
import onnxruntime as ort
import numpy as np
from pathlib import Path

from model import GomokuPolicyNet, N_BLOCKS


class GomokuModelWithSoftmax(nn.Module):
    """
    Wrapper that adds softmax with temperature to the policy output.

    Input is [2, 15, 15] (no batch dimension - single inference only).
    Internally adds batch dimension and constant mask channel before feeding to model.

    Outputs:
        - policy_probs: [15, 15] softmaxed policy probabilities (2D grid)
        - value: [1] value estimation (scalar)
    """
    def __init__(self, base_model: GomokuPolicyNet, temperature: float):
        super().__init__()
        self.base_model = base_model
        self.temperature = temperature

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [2, 15, 15] board state (channel 0: current player, channel 1: opponent)

        Returns:
            policy_probs: [15, 15] probability distribution over moves (2D grid)
            value: [1] win probability estimation
        """
        # Add batch dimension: [2, 15, 15] → [1, 2, 15, 15]
        x = x.unsqueeze(0)

        # Add constant mask channel: [1, 2, 15, 15] → [1, 3, 15, 15]
        batch_size, _, height, width = x.shape
        mask_channel = torch.ones(batch_size, 1, height, width, dtype=x.dtype, device=x.device)
        x_with_mask = torch.cat([x, mask_channel], dim=1)  # [1, 3, 15, 15]

        logits_grid, value = self.base_model(x_with_mask)  # [1, 1, 15, 15], [1, 1]

        # Flatten policy logits for softmax: [1, 1, 15, 15] → [1, 225]
        logits_flat = logits_grid.view(batch_size, -1)

        # Apply temperature and softmax
        policy_probs_flat = F.softmax(logits_flat / self.temperature, dim=1)

        # Reshape back to 2D grid: [1, 225] → [1, 15, 15]
        policy_probs_batched = policy_probs_flat.view(batch_size, height, width)

        # Remove batch dimension: [1, 15, 15] → [15, 15], [1, 1] → [1]
        policy_probs = policy_probs_batched.squeeze(0)
        value_scalar = value.squeeze(0).squeeze(0)

        return policy_probs, value_scalar


def load_checkpoint(checkpoint_path: str) -> GomokuPolicyNet:
    """Load model from PyTorch checkpoint."""
    print(f"Loading checkpoint: {checkpoint_path}")

    model = GomokuPolicyNet(n_blocks=N_BLOCKS)
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded: {n_params:,} parameters ({n_params/1e6:.2f}M)")

    return model


def export_to_onnx(model: nn.Module, output_path: str, temperature: float):
    """Export model to ONNX with softmax and temperature baked in."""
    print(f"\nExporting to ONNX (opset=21, temperature={temperature})...")

    # Wrap model with softmax layer
    wrapped_model = GomokuModelWithSoftmax(model, temperature)
    wrapped_model.eval()

    # Dummy input: [2, 15, 15] (no batch dimension)
    dummy_input = torch.randn(2, 15, 15)

    # Export to ONNX (with all data embedded in single file)
    torch.onnx.export(
        wrapped_model,
        dummy_input,
        output_path,
        opset_version=21,  # Required for correct GroupNormalization
        input_names=["board_state"],
        output_names=["policy_probs", "value"],
        dynamic_axes=None,  # Fixed batch=1
        do_constant_folding=True,
        export_params=True,
    )

    # Load and re-save to merge external data into single file
    # This is critical for web deployment to avoid missing .onnx.data files
    onnx_model = onnx.load(output_path)
    onnx.save(onnx_model, output_path, save_as_external_data=False)

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
        torch_policy, torch_value = torch_model(dummy_input)

    session = ort.InferenceSession(onnx_path)
    onnx_outputs = session.run(None, {"board_state": dummy_input.numpy()})
    onnx_policy, onnx_value = onnx_outputs

    # Compute differences
    policy_diff = np.abs(torch_policy.numpy() - onnx_policy).max()
    value_diff = np.abs(torch_value.numpy() - onnx_value).max()

    print(f"✓ Max difference - Policy: {policy_diff:.2e}, Value: {value_diff:.2e}")

    if policy_diff < 1e-5 and value_diff < 1e-5:
        print("✓ Verification passed (difference < 1e-5)")
    else:
        print("⚠ Warning: Larger than expected difference (may still be acceptable)")


def main():
    parser = argparse.ArgumentParser(
        description="Export Gomoku PyTorch model to ONNX",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python3 export_onnx.py --input runs/exp1/checkpoint_update_1000.pt --output model.onnx --temp 0.5
        """
    )
    parser.add_argument("--input", required=True, help="Input PyTorch checkpoint (.pt)")
    parser.add_argument("--output", required=True, help="Output ONNX file (.onnx)")
    parser.add_argument("--temp", type=float, required=True, help="Softmax temperature (e.g., 0.5)")

    args = parser.parse_args()

    # Validate inputs
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {args.input}")
        return

    if args.temp <= 0:
        print(f"Error: Temperature must be positive, got {args.temp}")
        return

    # Load model
    model = load_checkpoint(args.input)

    # Export to ONNX
    wrapped_model, dummy_input = export_to_onnx(model, args.output, args.temp)

    # Verify correctness
    verify_onnx(args.output, wrapped_model, dummy_input)

    # Report file size
    output_path = Path(args.output)
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"\n✓ Export complete!")
    print(f"  Output: {args.output}")
    print(f"  Size: {size_mb:.2f} MB")
    print(f"  Temperature: {args.temp}")
    print(f"\nReady for deployment with onnxruntime-web (WASM backend)")


if __name__ == "__main__":
    main()
