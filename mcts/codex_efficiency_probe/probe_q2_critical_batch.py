#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gc
import json
import sys
from dataclasses import dataclass
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from gomoku import LOGIT_MASK_VALUE
from probe_common import (
    configure_torch,
    constants_snapshot,
    ensure_output_dir,
    list_sample_shards,
    load_model,
    percentile_summary,
    resolve_device,
    seed_everything,
    train_main,
    write_json,
)
from training import TRAIN_BATCH_SIZE, augment_mcts_batch_8fold


@dataclass
class RoundData:
    update: int
    obs: np.ndarray
    dist: np.ndarray
    value: np.ndarray
    policy_weight: np.ndarray
    value_weight: np.ndarray
    reference_weight: float = 1.0

    @property
    def n(self) -> int:
        return int(self.value.shape[0])


@dataclass
class SelectedRound:
    data: RoundData
    indices: np.ndarray | None
    external_weight: float

    @property
    def n(self) -> int:
        return self.data.n if self.indices is None else int(self.indices.shape[0])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure gradient direction redundancy for Stage 2 replay batches."
    )
    parser.add_argument(
        "--stage2-dir",
        type=Path,
        default=Path("codex_efficiency_probe/data/stage2"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("codex_efficiency_probe/results/q2_critical_batch"),
    )
    parser.add_argument("--probe-updates", type=int, nargs="*")
    parser.add_argument("--sample-ratios", type=float, nargs="*")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--cos-threshold", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=train_main.SEED)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--raw-chunk-size", type=int, default=512)
    parser.add_argument("--micro-batch-size", type=int, default=TRAIN_BATCH_SIZE)
    parser.add_argument("--max-samples-per-round", type=int, default=0)
    parser.add_argument(
        "--round-weight-mode",
        choices=["exact_sampler", "decay_only"],
        default="exact_sampler",
    )
    return parser.parse_args()


def default_probe_updates(stage2_dir: Path) -> list[int]:
    shards = list_sample_shards(stage2_dir)
    latest_sample = shards[-1].update
    return [latest_sample]


def default_sample_ratios() -> list[float]:
    current = train_main.STAGE2_SAMPLE_RATIO
    ratios = {1.0, 0.5, current}
    for k in range(1, 5):
        ratios.add(current / (2 ** k))
    return sorted((r for r in ratios if r > 0), reverse=True)


def load_round(path: Path, update: int, rng: np.random.Generator, max_samples: int) -> RoundData:
    with np.load(path) as data:
        n = int(data["value"].shape[0])
        if max_samples > 0 and n > max_samples:
            idx = np.sort(rng.choice(n, size=max_samples, replace=False))
        else:
            idx = slice(None)
        return RoundData(
            update=update,
            obs=np.ascontiguousarray(data["obs"][idx]),
            dist=np.ascontiguousarray(data["dist"][idx]),
            value=np.ascontiguousarray(data["value"][idx]),
            policy_weight=np.ascontiguousarray(data["policy_weight"][idx]),
            value_weight=np.ascontiguousarray(data["value_weight"][idx]),
        )


def load_replay_rounds(
    stage2_dir: Path,
    probe_update: int,
    rng: np.random.Generator,
    max_samples_per_round: int,
) -> list[RoundData]:
    by_update = {s.update: s.path for s in list_sample_shards(stage2_dir)}
    rounds = train_main.STAGE2_REPLAY_BUFFER_ROUNDS
    start = probe_update - rounds + 1
    if start < 0:
        raise ValueError(
            f"Probe update {probe_update} does not have a full replay window; "
            f"need at least {rounds} sample shards."
        )
    missing = [u for u in range(start, probe_update + 1) if u not in by_update]
    if missing:
        raise FileNotFoundError(f"Missing sample shards for updates: {missing}")

    ordered_newest_first = []
    for update in range(probe_update, start - 1, -1):
        ordered_newest_first.append(
            load_round(by_update[update], update, rng, max_samples_per_round)
        )
    return ordered_newest_first


def attach_reference_weights(rounds: list[RoundData], mode: str) -> None:
    newest_n = rounds[0].n
    decay = train_main.STAGE2_DECAY_RATIO
    current_ratio = train_main.STAGE2_SAMPLE_RATIO
    base_k = max(1, min(newest_n, round(current_ratio * newest_n)))
    base_density = base_k / max(newest_n, 1)
    for i, rd in enumerate(rounds):
        if mode == "decay_only":
            rd.reference_weight = float(decay ** i)
        else:
            k_i = max(0, min(rd.n, round(current_ratio * newest_n * (decay ** i))))
            density_i = k_i / max(rd.n, 1)
            rd.reference_weight = float(density_i / base_density)


def make_reference_selection(rounds: list[RoundData]) -> list[SelectedRound]:
    return [
        SelectedRound(data=rd, indices=None, external_weight=rd.reference_weight)
        for rd in rounds
    ]


def make_sampled_selection(
    rounds: list[RoundData],
    sample_ratio: float,
    rng: np.random.Generator,
) -> tuple[list[SelectedRound], list[int]]:
    newest_n = rounds[0].n
    decay = train_main.STAGE2_DECAY_RATIO
    selected = []
    counts = []
    for i, rd in enumerate(rounds):
        k = round(sample_ratio * newest_n * (decay ** i))
        k = max(0, min(rd.n, k))
        counts.append(k)
        if k == 0:
            indices = np.empty(0, dtype=np.int64)
        else:
            indices = np.sort(rng.choice(rd.n, size=k, replace=False))
        selected.append(SelectedRound(data=rd, indices=indices, external_weight=1.0))
    return selected, counts


def iter_selected_arrays(
    block: SelectedRound,
    raw_chunk_size: int,
):
    rd = block.data
    if block.indices is None:
        total = rd.n
        for start in range(0, total, raw_chunk_size):
            idx = slice(start, min(start + raw_chunk_size, total))
            yield (
                rd.obs[idx],
                rd.dist[idx],
                rd.value[idx],
                rd.policy_weight[idx] * block.external_weight,
                rd.value_weight[idx] * block.external_weight,
            )
    else:
        total = block.indices.shape[0]
        for start in range(0, total, raw_chunk_size):
            take = block.indices[start : start + raw_chunk_size]
            if take.size == 0:
                continue
            yield (
                rd.obs[take],
                rd.dist[take],
                rd.value[take],
                rd.policy_weight[take] * block.external_weight,
                rd.value_weight[take] * block.external_weight,
            )


def selected_weight_totals(blocks: list[SelectedRound]) -> tuple[float, float]:
    policy_total = 0.0
    value_total = 0.0
    for block in blocks:
        rd = block.data
        if block.indices is None:
            policy_total += float(rd.policy_weight.sum()) * block.external_weight
            value_total += float(rd.value_weight.sum()) * block.external_weight
        elif block.indices.size > 0:
            policy_total += float(rd.policy_weight[block.indices].sum()) * block.external_weight
            value_total += float(rd.value_weight[block.indices].sum()) * block.external_weight
    return max(1.0, 8.0 * policy_total), max(1.0, 8.0 * value_total)


def selected_sample_count(blocks: list[SelectedRound]) -> int:
    return sum(block.n for block in blocks)


def compute_gradient_vector(
    model: torch.nn.Module,
    blocks: list[SelectedRound],
    *,
    device: torch.device,
    raw_chunk_size: int,
    micro_batch_size: int,
) -> torch.Tensor:
    policy_w_total, value_w_total = selected_weight_totals(blocks)
    policy_w_total_t = torch.tensor(policy_w_total, dtype=torch.float32, device=device)
    value_w_total_t = torch.tensor(value_w_total, dtype=torch.float32, device=device)

    model.zero_grad(set_to_none=True)
    model.train()

    for block in blocks:
        for obs_np, dist_np, value_np, policy_w_np, value_w_np in iter_selected_arrays(
            block, raw_chunk_size
        ):
            mask_np = ((obs_np[:, 0] | obs_np[:, 1]) == 0)
            obs_t = torch.from_numpy(obs_np).float().to(device)
            dist_t = torch.from_numpy(dist_np).float().to(device)
            mask_t = torch.from_numpy(mask_np).bool().to(device)
            value_t = torch.from_numpy(value_np).float().to(device)
            policy_w_t = torch.from_numpy(policy_w_np).float().to(device)
            value_w_t = torch.from_numpy(value_w_np).float().to(device)

            obs_t, dist_t, mask_t, value_t, policy_w_t, value_w_t = augment_mcts_batch_8fold(
                obs_t, dist_t, mask_t, value_t, policy_w_t, value_w_t
            )

            n_aug = obs_t.shape[0]
            for start in range(0, n_aug, micro_batch_size):
                end = min(start + micro_batch_size, n_aug)
                mb_obs = obs_t[start:end]
                mb_dist = dist_t[start:end]
                mb_mask = mask_t[start:end]
                mb_value = value_t[start:end]
                mb_policy_w = policy_w_t[start:end]
                mb_value_w = value_w_t[start:end]
                mb_size = end - start

                logits, pred_values = model(mb_obs)
                logits = logits.squeeze(1).view(mb_size, 225)
                pred_values = pred_values.squeeze(-1)
                logits = logits.masked_fill(~mb_mask.view(mb_size, 225), LOGIT_MASK_VALUE)

                log_probs = F.log_softmax(logits, dim=-1)
                ce = -(mb_dist * log_probs).sum(dim=-1)
                se = (pred_values - mb_value) ** 2
                loss = (
                    (mb_policy_w * ce).sum() / policy_w_total_t
                    + train_main.VALUE_LOSS_COEFF
                    * (mb_value_w * se).sum()
                    / value_w_total_t
                )
                loss.backward()

    parts = []
    for param in model.parameters():
        if param.grad is None:
            parts.append(torch.zeros(param.numel(), dtype=torch.float32))
        else:
            parts.append(param.grad.detach().reshape(-1).float().cpu())
    return torch.cat(parts)


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    denom = torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b)
    if float(denom) == 0.0:
        return float("nan")
    return float(torch.dot(a, b) / denom)


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_probe_update(
    *,
    stage2_dir: Path,
    output_dir: Path,
    probe_update: int,
    sample_ratios: list[float],
    repeats: int,
    threshold: float,
    seed: int,
    device: torch.device,
    raw_chunk_size: int,
    micro_batch_size: int,
    max_samples_per_round: int,
    round_weight_mode: str,
) -> dict[str, object]:
    rng = np.random.default_rng(seed + probe_update)
    rounds = load_replay_rounds(stage2_dir, probe_update, rng, max_samples_per_round)
    attach_reference_weights(rounds, round_weight_mode)

    model = load_model(stage2_dir, probe_update, device)
    reference_blocks = make_reference_selection(rounds)
    print(
        f"computing reference gradient: checkpoint={probe_update} "
        f"raw_samples={selected_sample_count(reference_blocks)}",
        flush=True,
    )
    g_ref = compute_gradient_vector(
        model,
        reference_blocks,
        device=device,
        raw_chunk_size=raw_chunk_size,
        micro_batch_size=micro_batch_size,
    )
    ref_norm = float(torch.linalg.vector_norm(g_ref))

    rows: list[dict[str, object]] = []
    ratio_summaries: dict[str, object] = {}
    for ratio in sample_ratios:
        cosines = []
        raw_counts = []
        per_round_counts_seen = []
        for repeat in range(repeats):
            repeat_rng = np.random.default_rng(seed + probe_update * 100_000 + repeat * 1009 + int(ratio * 1e9))
            blocks, counts = make_sampled_selection(rounds, ratio, repeat_rng)
            raw_n = selected_sample_count(blocks)
            print(
                f"computing subset gradient: checkpoint={probe_update} "
                f"sample_ratio={ratio:g} repeat={repeat} raw_samples={raw_n}",
                flush=True,
            )
            g = compute_gradient_vector(
                model,
                blocks,
                device=device,
                raw_chunk_size=raw_chunk_size,
                micro_batch_size=micro_batch_size,
            )
            c = cosine(g, g_ref)
            cosines.append(c)
            raw_counts.append(raw_n)
            per_round_counts_seen.append(counts)
            rows.append({
                "probe_update": probe_update,
                "sample_ratio": ratio,
                "repeat": repeat,
                "cosine": c,
                "raw_samples": raw_n,
                "augmented_samples": raw_n * 8,
                "per_round_counts": json.dumps(counts),
            })
            del g
            gc.collect()

        cos_arr = np.asarray(cosines, dtype=np.float64)
        ratio_summaries[str(ratio)] = {
            "sample_ratio": ratio,
            "raw_samples_mean": float(np.mean(raw_counts)),
            "augmented_samples_mean": float(np.mean(raw_counts) * 8),
            "cosine": percentile_summary(cos_arr),
            "cosine_std": float(cos_arr.std(ddof=1)) if len(cos_arr) > 1 else 0.0,
            "per_round_counts_first_repeat": per_round_counts_seen[0],
        }

    passing = [
        float(r)
        for r, item in ratio_summaries.items()
        if item["cosine"]["mean"] >= threshold
    ]
    recommended = min(passing) if passing else None

    round_meta = []
    for i, rd in enumerate(rounds):
        round_meta.append({
            "age_index": i,
            "sample_update": rd.update,
            "num_samples": rd.n,
            "reference_weight": rd.reference_weight,
        })

    summary = {
        "probe_update": probe_update,
        "checkpoint": str(stage2_dir / f"checkpoint_update_{probe_update}.pt"),
        "reference": {
            "raw_samples": selected_sample_count(reference_blocks),
            "augmented_samples": selected_sample_count(reference_blocks) * 8,
            "gradient_norm": ref_norm,
            "round_weight_mode": round_weight_mode,
            "rounds_newest_first": round_meta,
        },
        "sample_ratios": ratio_summaries,
        "cos_threshold": threshold,
        "recommended_sample_ratio": recommended,
        "current_sample_ratio": train_main.STAGE2_SAMPLE_RATIO,
        "constants": constants_snapshot(),
        "max_samples_per_round": max_samples_per_round,
    }

    prefix = output_dir / f"q2_update_{probe_update}"
    write_rows(prefix.with_suffix(".cosines.csv"), rows)
    write_json(prefix.with_suffix(".summary.json"), summary)

    del g_ref
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return summary


def main() -> None:
    args = parse_args()
    configure_torch()
    seed_everything(args.seed)
    ensure_output_dir(args.output_dir)
    device = resolve_device(args.device)

    probe_updates = args.probe_updates or default_probe_updates(args.stage2_dir)
    sample_ratios = args.sample_ratios or default_sample_ratios()
    sample_ratios = sorted({r for r in sample_ratios if r > 0}, reverse=True)

    summaries = []
    for probe_update in probe_updates:
        summaries.append(
            run_probe_update(
                stage2_dir=args.stage2_dir,
                output_dir=args.output_dir,
                probe_update=probe_update,
                sample_ratios=sample_ratios,
                repeats=args.repeats,
                threshold=args.cos_threshold,
                seed=args.seed,
                device=device,
                raw_chunk_size=args.raw_chunk_size,
                micro_batch_size=args.micro_batch_size,
                max_samples_per_round=args.max_samples_per_round,
                round_weight_mode=args.round_weight_mode,
            )
        )

    write_json(
        args.output_dir / "q2_summary.json",
        {
            "probe_updates": probe_updates,
            "sample_ratios": sample_ratios,
            "summaries": summaries,
        },
    )
    print(f"wrote {args.output_dir / 'q2_summary.json'}", flush=True)


if __name__ == "__main__":
    main()
