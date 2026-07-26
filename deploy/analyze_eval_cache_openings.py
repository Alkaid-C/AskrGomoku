#!/usr/bin/env python3
"""Collect shallow exact positions and optionally write a Melody cache file."""

import argparse
import hashlib
import struct
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

MCTS_DIR = Path(__file__).resolve().parent.parent / "mcts"
sys.path.insert(0, str(MCTS_DIR))

from gomoku import GomokuBoard
from model import GomokuPolicyNet

import mcts

C_PUCT = 1.25
DISCOUNT_GAMMA = 63.0 / 64
FPU_MULTIPLIER = 0.95

FORMAT_MAGIC = b"GMKECACH"
FORMAT_VERSION = 1
HEADER_SIZE = 64
BOARD_SIZE = 15
POLICY_SIZE = BOARD_SIZE * BOARD_SIZE
RECORD_SIZE = 964


def exact_position_key(obs: np.ndarray) -> tuple[bytes, int]:
    """Return the browser-cache-equivalent position key with no D4 folding."""
    key = np.packbits(np.ascontiguousarray(obs[:2]).ravel()).tobytes()
    return key, 0


def load_model(checkpoint_path: Path, device: torch.device) -> GomokuPolicyNet:
    model = GomokuPolicyNet()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model.to(device)


def build_roots(grid_min: int, grid_max: int) -> list[GomokuBoard]:
    roots = [GomokuBoard(opening_id=-1)]
    for row in range(grid_min, grid_max + 1):
        for col in range(grid_min, grid_max + 1):
            board = GomokuBoard(opening_id=-1)
            board.Move((row, col))
            roots.append(board)
    return roots


def stone_count_from_key(key: bytes) -> int:
    return sum(byte.bit_count() for byte in key)


def absolute_planes(obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    current_count = int(obs[0].sum())
    opponent_count = int(obs[1].sum())
    if current_count == opponent_count:
        return obs[0], obs[1]
    if opponent_count == current_count + 1:
        return obs[1], obs[0]
    raise ValueError(
        f"unreachable perspective counts: current={current_count}, "
        f"opponent={opponent_count}"
    )


def row_masks(plane: np.ndarray) -> list[int]:
    return [
        sum(int(plane[row, col]) << col for col in range(BOARD_SIZE))
        for row in range(BOARD_SIZE)
    ]


def write_cache_file(
    output_path: Path,
    onnx_path: Path,
    lru_keys: list[bytes],
    positions: dict[bytes, np.ndarray],
) -> None:
    model_bytes = onnx_path.read_bytes()
    model_hash = hashlib.sha256(model_bytes).digest()
    file_bytes = bytearray(HEADER_SIZE + len(lru_keys) * RECORD_SIZE)

    file_bytes[: len(FORMAT_MAGIC)] = FORMAT_MAGIC
    struct.pack_into(
        "<HHHHII",
        file_bytes,
        0x08,
        FORMAT_VERSION,
        BOARD_SIZE,
        POLICY_SIZE,
        len(lru_keys),
        RECORD_SIZE,
        0,
    )
    file_bytes[0x18:0x38] = model_hash

    session = ort.InferenceSession(model_bytes)
    input_name = session.get_inputs()[0].name
    for index, key in enumerate(lru_keys):
        obs = positions[key]
        outputs = session.run(None, {input_name: obs.astype(np.float32, copy=False)})
        logits = np.asarray(outputs[0], dtype=np.float32).reshape(POLICY_SIZE)
        value = float(np.asarray(outputs[1]).reshape(-1)[0])
        if not np.isfinite(logits).all() or not np.isfinite(value):
            raise ValueError(f"non-finite ONNX output for record {index}")
        if value < -1.0 or value > 1.0:
            raise ValueError(f"out-of-range ONNX value for record {index}: {value}")

        black, white = absolute_planes(obs)
        base = HEADER_SIZE + index * RECORD_SIZE
        struct.pack_into("<15H", file_bytes, base, *row_masks(black))
        struct.pack_into("<15H", file_bytes, base + 0x1E, *row_masks(white))
        file_bytes[base + 0x3C:base + 0x3C0] = logits.astype(
            "<f4", copy=False
        ).tobytes()
        struct.pack_into("<f", file_bytes, base + 0x3C0, value)

        if (index + 1) % 256 == 0 or index + 1 == len(lru_keys):
            print(f"ONNX evaluations: {index + 1}/{len(lru_keys)}", flush=True)

    output_path.write_bytes(file_bytes)
    print(f"cache file: {output_path}")
    print(f"cache bytes: {len(file_bytes)}")
    print(f"model SHA-256: {model_hash.hex()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--onnx", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--entries", type=int, default=2048)
    parser.add_argument("--sims", type=int, default=256)
    parser.add_argument("--grid-min", type=int, default=5)
    parser.add_argument("--grid-max", type=int, default=9)
    args = parser.parse_args()

    if args.grid_min > args.grid_max:
        parser.error("--grid-min must not exceed --grid-max")
    if (args.onnx is None) != (args.output is None):
        parser.error("--onnx and --output must be provided together")
    if args.entries < 0:
        parser.error("--entries must be non-negative")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device)
    roots = build_roots(args.grid_min, args.grid_max)

    lookup_counts: Counter[bytes] = Counter()
    positions: dict[bytes, np.ndarray] = {}
    original_evaluate = mcts._evaluate_with_cache

    def recording_evaluate(
        eval_model: torch.nn.Module,
        obs_list: list[np.ndarray],
        entropy_multiplier: float | None,
        eval_device: torch.device,
    ) -> tuple[np.ndarray, np.ndarray]:
        for obs in obs_list:
            key = exact_position_key(obs)[0]
            lookup_counts[key] += 1
            positions.setdefault(key, np.ascontiguousarray(obs[:2]).copy())
        return original_evaluate(
            eval_model, obs_list, entropy_multiplier, eval_device
        )

    mcts.canonicalize_obs = exact_position_key
    mcts._evaluate_with_cache = recording_evaluate
    mcts.clear_nn_eval_cache()

    started = time.perf_counter()
    mcts.mcts_search_batched(
        model=model,
        boards=roots,
        num_simulations=args.sims,
        c_puct=C_PUCT,
        entropy_multiplier=None,
        device=device,
        dirichlet_alpha=0.0,
        dirichlet_epsilon=0.0,
        gamma=DISCOUNT_GAMMA,
        fpu_multiplier=FPU_MULTIPLIER,
    )
    elapsed = time.perf_counter() - started

    hits, misses = mcts.get_nn_eval_cache_stats()
    total_lookups = sum(lookup_counts.values())
    duplicate_lookups = total_lookups - len(lookup_counts)
    depth_counts = Counter(stone_count_from_key(key) for key in lookup_counts)
    repeated_counts = Counter(
        stone_count_from_key(key)
        for key, count in lookup_counts.items()
        for _ in range(count - 1)
    )

    print(f"device: {device}")
    print(f"roots: {len(roots)} (empty + {len(roots) - 1} one-stone)")
    print(f"simulations per root: {args.sims}")
    print(f"eval lookups: {total_lookups}")
    print(f"unique exact positions: {len(lookup_counts)}")
    print(f"duplicate lookups: {duplicate_lookups}")
    print(f"exact-cache stats: hits={hits}, misses={misses}")
    print(f"elapsed seconds: {elapsed:.3f}")
    print("unique positions by stone count:")
    for depth in sorted(depth_counts):
        print(
            f"  stones={depth}: unique={depth_counts[depth]}, "
            f"duplicates={repeated_counts[depth]}"
        )

    if args.output is not None and args.onnx is not None:
        if args.entries > len(positions):
            parser.error(
                f"--entries={args.entries} exceeds {len(positions)} candidates"
            )

        selected_keys = sorted(
            positions,
            key=lambda key: (
                stone_count_from_key(key),
                -lookup_counts[key],
                key,
            ),
        )[: args.entries]
        selected_depths = Counter(stone_count_from_key(key) for key in selected_keys)
        selected_lookups = sum(lookup_counts[key] for key in selected_keys)

        # Records are serialized oldest-to-newest. Shallower positions become
        # unreachable earlier in a game, so they sit at the eviction end.
        lru_keys = sorted(
            selected_keys,
            key=lambda key: (
                stone_count_from_key(key),
                lookup_counts[key],
                key,
            ),
        )

        print("selected positions by stone count:")
        for depth in sorted(selected_depths):
            print(f"  stones={depth}: selected={selected_depths[depth]}")
        print(
            f"selected lookup coverage: {selected_lookups}/{total_lookups} "
            f"({selected_lookups / total_lookups:.2%})"
        )
        write_cache_file(args.output, args.onnx, lru_keys, positions)


if __name__ == "__main__":
    main()
