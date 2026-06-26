from __future__ import annotations

import json
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gomoku import Player
from model import GomokuPolicyNet

import main as train_main


@dataclass(frozen=True)
class SampleShard:
    update: int
    path: Path


def configure_torch() -> None:
    try:
        torch.backends.cudnn.conv.fp32_precision = "tf32"
        torch.backends.cuda.matmul.fp32_precision = "tf32"
    except Exception:
        pass


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def latest_checkpoint_update(stage2_dir: Path) -> int:
    state_path = stage2_dir / "training_state.json"
    if state_path.exists():
        with state_path.open() as f:
            state = json.load(f)
        return int(state["current_update"])

    updates = []
    for path in stage2_dir.glob("checkpoint_update_*.pt"):
        stem = path.stem
        updates.append(int(stem.rsplit("_", 1)[1]))
    if not updates:
        raise FileNotFoundError(f"No checkpoint_update_*.pt files under {stage2_dir}")
    return max(updates)


def load_model(stage2_dir: Path, update: int, device: torch.device) -> GomokuPolicyNet:
    ckpt_path = stage2_dir / f"checkpoint_update_{update}.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = GomokuPolicyNet().to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    return model


def parse_sample_update(path: Path) -> int:
    return int(path.stem.rsplit("_", 1)[1])


def list_sample_shards(stage2_dir: Path) -> list[SampleShard]:
    samples_dir = stage2_dir / "samples"
    shards = [
        SampleShard(parse_sample_update(path), path)
        for path in samples_dir.glob("samples_update_*.npz")
    ]
    shards.sort(key=lambda s: s.update)
    if not shards:
        raise FileNotFoundError(f"No samples_update_*.npz files under {samples_dir}")
    return shards


def sample_shards_by_update(stage2_dir: Path, updates: Iterable[int]) -> list[SampleShard]:
    by_update = {s.update: s for s in list_sample_shards(stage2_dir)}
    out = []
    for update in updates:
        if update not in by_update:
            raise FileNotFoundError(f"Missing samples_update_{update}.npz")
        out.append(by_update[update])
    return out


def default_recent_sample_updates(stage2_dir: Path, rounds: int) -> list[int]:
    shards = list_sample_shards(stage2_dir)
    return [s.update for s in shards[-rounds:]]


def player_to_move_from_obs(obs: np.ndarray) -> Player:
    stones = int(obs[0].sum() + obs[1].sum())
    return Player.BLACK if stones % 2 == 0 else Player.WHITE


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, data: object) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump(data, f, indent=2, sort_keys=True)
    tmp.replace(path)


def percentile_summary(values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        raise ValueError("Cannot summarize an empty array")
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(arr.max()),
    }


def constants_snapshot() -> dict[str, float | int]:
    return {
        "NUM_SIMULATIONS_S2": train_main.NUM_SIMULATIONS_S2,
        "C_PUCT": train_main.C_PUCT,
        "STAGE2_DIRICHLET_ALPHA": train_main.STAGE2_DIRICHLET_ALPHA,
        "STAGE2_DIRICHLET_EPSILON": train_main.STAGE2_DIRICHLET_EPSILON,
        "STAGE2_REPLAY_BUFFER_ROUNDS": train_main.STAGE2_REPLAY_BUFFER_ROUNDS,
        "STAGE2_SAMPLE_RATIO": train_main.STAGE2_SAMPLE_RATIO,
        "STAGE2_DECAY_RATIO": train_main.STAGE2_DECAY_RATIO,
        "VALUE_LOSS_COEFF": train_main.VALUE_LOSS_COEFF,
        "DISCOUNT_GAMMA": train_main.DISCOUNT_GAMMA,
        "FPU_MULTIPLIER": train_main.FPU_MULTIPLIER,
    }
