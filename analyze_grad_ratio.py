"""Analyze value/policy gradient norm ratio on shared (trunk+stem) parameters.

Usage:
    python analyze_grad_ratio.py <directory_containing_npz_files>

Reads gradient_probe_*.npz files, filters to shared parameters (excluding any
param with 'policy' or 'value' in the name), and reports raw V/P gradient norm
ratio statistics for updates > 1024.
"""

import sys
import glob
import os

import numpy as np


def is_shared(name: str) -> bool:
    return "policy" not in name and "value" not in name


def analyze(directory: str) -> None:
    pattern = os.path.join(directory, "gradient_probe_*.npz")
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"No gradient_probe_*.npz files found in {directory}")
        sys.exit(1)

    ratios = []

    for path in files:
        f = np.load(path, allow_pickle=True)
        update = int(f["update"])
        if update <= 1024:
            continue

        names = f["param_names"]
        offsets = f["param_offsets"]
        policy_real = f["policy_real"]
        policy_synthetic = f["policy_synthetic"]
        value_real = f["value_real"]

        policy_total = policy_real + policy_synthetic

        # Build mask for shared parameters
        n_params = len(names)
        total_len = len(policy_total)
        shared_slices = []
        for i in range(n_params):
            if not is_shared(str(names[i])):
                continue
            start = int(offsets[i])
            end = int(offsets[i + 1]) if i + 1 < n_params else total_len
            shared_slices.append((start, end))

        # Extract shared parameter gradients
        policy_parts = [policy_total[s:e] for s, e in shared_slices]
        value_parts = [value_real[s:e] for s, e in shared_slices]

        policy_shared = np.concatenate(policy_parts)
        value_shared = np.concatenate(value_parts)

        policy_norm = float(np.linalg.norm(policy_shared))
        value_norm = float(np.linalg.norm(value_shared))
        ratio = value_norm / max(policy_norm, 1e-12)
        ratios.append(ratio)

    if not ratios:
        print("No probes found with update > 1024")
        sys.exit(1)

    arr = np.array(ratios)
    arr_sorted = np.sort(arr)
    n = len(arr_sorted)

    print(f"Shared-param V/P gradient norm ratio (updates > 1024, n={n})")
    print(f"  Mean:   {arr.mean():.3f}")
    print(f"  Median: {np.median(arr):.3f}")
    print(f"  P10:    {arr_sorted[int(n * 0.1)]:.3f}")
    print(f"  P25:    {arr_sorted[int(n * 0.25)]:.3f}")
    print(f"  P75:    {arr_sorted[int(n * 0.75)]:.3f}")
    print(f"  P90:    {arr_sorted[int(n * 0.9)]:.3f}")
    print(f"  Min:    {arr_sorted[0]:.3f}")
    print(f"  Max:    {arr_sorted[-1]:.3f}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <directory>")
        sys.exit(1)
    analyze(sys.argv[1])
