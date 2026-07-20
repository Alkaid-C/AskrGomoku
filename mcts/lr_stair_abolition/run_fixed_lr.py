"""Resume update 120 while holding the pre-stair LR through update 152."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import stage2_trainer as trainer

ROOT = Path(__file__).resolve().parent
CONFIG = json.loads((ROOT / "provenance.json").read_text())
SOURCE_UPDATE = int(CONFIG["source_update"])
END_UPDATE = int(CONFIG["end_update"])
FIXED_STAIR = int(CONFIG["fixed_stairs_descended"])
FIXED_LR = float(CONFIG["fixed_lr"])


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_inputs() -> int:
    for relative_path, expected in CONFIG["sha256"].items():
        path = ROOT / relative_path
        actual = _sha256(path)
        if actual != expected:
            raise RuntimeError(f"SHA-256 mismatch for {path}: {actual} != {expected}")

    state = json.loads((ROOT / "stage2" / "training_state.json").read_text())
    current_update = int(state["current_update"])
    if not SOURCE_UPDATE <= current_update <= END_UPDATE:
        raise RuntimeError(
            f"Resume update {current_update} is outside [{SOURCE_UPDATE}, {END_UPDATE}]"
        )

    checkpoint = trainer.torch.load(
        ROOT / "stage2" / f"checkpoint_update_{current_update}.pt",
        map_location="cpu",
        weights_only=False,
    )
    if int(checkpoint["update"]) != current_update:
        raise RuntimeError("training_state.json and checkpoint update do not match")
    if len(checkpoint["lr_controller_state_dict"]["kl_history"]) != current_update:
        raise RuntimeError("Checkpoint update and KL history length do not match")
    buffer_path = ROOT / "stage2" / f"replay_buffer_update_{current_update}.pkl"
    if not buffer_path.is_file():
        raise RuntimeError(f"Matching replay buffer is missing: {buffer_path}")
    required_rng = {"python", "numpy", "torch", "torch_cuda"}
    if set(checkpoint.get("rng_state", {})) != required_rng:
        raise RuntimeError("Checkpoint does not contain the complete RNG state")
    if current_update == SOURCE_UPDATE:
        controller_state = checkpoint["lr_controller_state_dict"]
        if int(controller_state["stairs_descended"]) != FIXED_STAIR + 1:
            raise RuntimeError("Source checkpoint is not the expected post-stair snapshot")
    else:
        controller_state = checkpoint["lr_controller_state_dict"]
        if int(controller_state["stairs_descended"]) != FIXED_STAIR:
            raise RuntimeError("Ablation checkpoint is not on the fixed stair")
    return current_update


def _truncate_csv_to_checkpoint(current_update: int) -> None:
    path = ROOT / "stage2" / "training_updates.csv"
    with path.open(newline="") as handle:
        rows = list(csv.reader(handle))
    header, data_rows = rows[0], rows[1:]
    if header != trainer.Stage2CSVLogger.COLUMNS:
        raise RuntimeError("training_updates.csv has an unexpected header")
    by_update = {
        int(row[0]): row for row in data_rows if int(row[0]) <= current_update
    }
    expected = set(range(1, current_update + 1))
    if set(by_update) != expected:
        raise RuntimeError("training_updates.csv is missing committed updates")
    canonical = [header] + [by_update[update] for update in sorted(by_update)]
    if rows == canonical:
        return
    tmp = path.with_suffix(".csv.tmp")
    with tmp.open("w", newline="") as handle:
        csv.writer(handle).writerows(canonical)
    tmp.replace(path)


class FixedLRController(trainer.StaircaseLRController):
    """Keep the counterfactual stair fixed and stop at the experiment horizon."""

    def load_state_dict(self, state: dict) -> None:
        history_len = len(state["kl_history"])
        if not SOURCE_UPDATE <= history_len <= END_UPDATE:
            raise RuntimeError(f"Unexpected KL history length: {history_len}")
        if history_len > SOURCE_UPDATE and int(state["stairs_descended"]) != FIXED_STAIR:
            raise RuntimeError("Ablation checkpoint resumed with the wrong stair")

        super().load_state_dict(state)
        self.stairs_descended = FIXED_STAIR
        self.next_check_step = END_UPDATE + 1
        self.finished = history_len == END_UPDATE
        self._set_lr(FIXED_LR)
        actual_lr = float(self.optimizer.param_groups[0]["lr"])
        if not math.isclose(actual_lr, FIXED_LR, rel_tol=0.0, abs_tol=0.0):
            raise RuntimeError(f"Failed to set fixed LR: {actual_lr}")

    def record(self, step: int, kl: float) -> None:
        expected_step = len(self.kl_history) + 1
        if step != expected_step or step > END_UPDATE:
            raise RuntimeError(f"Unexpected record step {step}; expected {expected_step}")
        self.kl_history.append(float(kl))
        self.last_check_step = -1
        self.last_slope = 0.0
        self.last_threshold = 0.0
        self.finished = step == END_UPDATE


def main() -> None:
    current_update = _verify_inputs()
    if sys.argv[1:] == ["--verify-only"]:
        print("Ablation snapshot and source hashes verified.")
        return
    if sys.argv[1:]:
        raise SystemExit(f"Usage: {Path(sys.argv[0]).name} [--verify-only]")

    _truncate_csv_to_checkpoint(current_update)
    trainer.StaircaseLRController = FixedLRController
    import main as training_main

    print(
        f"LR-stair ablation: resume {SOURCE_UPDATE}, fixed LR {FIXED_LR}, "
        f"stop after update {END_UPDATE}"
    )
    sys.argv = [sys.argv[0], "stage2", str(ROOT)]
    training_main.main()


if __name__ == "__main__":
    main()
