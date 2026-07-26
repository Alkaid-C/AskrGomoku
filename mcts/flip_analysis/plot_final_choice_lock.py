"""Plot when the tracker commits to each position's final choice.

For positions with at least one significant flip, the lock simulation is the
last recorded flip: after that point the committed winner never changes again.
Positions with no flip are treated as locked at simulation zero.
"""

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/flip-analysis-matplotlib")

import matplotlib.pyplot as plt
import numpy as np


DATA_PATH = Path(__file__).with_name("flip_data.jsonl")
BIN_WIDTH = 64
TOTAL_SIMS = 4096


def load_lock_sims() -> np.ndarray:
    lock_sims = []
    with DATA_PATH.open() as src:
        for line in src:
            row = json.loads(line)
            lock_sims.append(row["last_flip_sim"] or 0)
    return np.asarray(lock_sims, dtype=np.int32)


def configure_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 180,
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.7,
        }
    )


def plot_newly_locked(lock_sims: np.ndarray) -> None:
    changed = lock_sims[lock_sims > 0]
    edges = np.arange(0, TOTAL_SIMS + BIN_WIDTH, BIN_WIDTH)
    counts, _ = np.histogram(changed, bins=edges)
    centers = edges[:-1] + BIN_WIDTH / 2
    no_flip = int((lock_sims == 0).sum())

    fig, ax = plt.subplots(figsize=(11, 5.6))
    ax.bar(
        centers,
        counts,
        width=BIN_WIDTH * 0.88,
        color="#3B82F6",
        edgecolor="none",
    )
    ax.axvline(2048, color="#DC2626", linestyle="--", linewidth=1.4)
    ax.text(
        2048 + 35,
        ax.get_ylim()[1] * 0.92,
        "action budget = 2048",
        color="#B91C1C",
        va="top",
    )
    ax.text(
        0.985,
        0.96,
        f"Locked at sim 0 (no significant flip): {no_flip:,}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": "#CBD5E1"},
    )
    ax.set_title(
        "Final choice lock events by simulation\n"
        "Newly locked positions in 64-simulation bins; "
        "sim-0 positions are annotated separately"
    )
    ax.set_xlabel("Simulation")
    ax.set_ylabel("Final choice locked count")
    ax.set_xlim(0, TOTAL_SIMS)
    ax.set_xticks(np.arange(0, TOTAL_SIMS + 1, 512))
    fig.tight_layout()
    fig.savefig(Path(__file__).with_name("final_choice_locked_count.png"))
    plt.close(fig)


def plot_cumulative(lock_sims: np.ndarray) -> None:
    endpoints = np.arange(0, TOTAL_SIMS + BIN_WIDTH, BIN_WIDTH)
    cumulative = np.asarray([(lock_sims <= t).sum() for t in endpoints])
    total = lock_sims.size
    at_action = int((lock_sims <= 2048).sum())

    fig, ax = plt.subplots(figsize=(11, 5.6))
    ax.step(
        endpoints,
        cumulative,
        where="post",
        color="#0F766E",
        linewidth=2.2,
    )
    ax.fill_between(
        endpoints,
        cumulative,
        step="post",
        color="#14B8A6",
        alpha=0.15,
    )
    ax.axvline(2048, color="#DC2626", linestyle="--", linewidth=1.4)
    ax.scatter([0, 2048, TOTAL_SIMS], [cumulative[0], at_action, total],
               color=["#0F766E", "#DC2626", "#0F766E"], zorder=3)
    ax.annotate(
        f"sim 0: {cumulative[0]:,} ({cumulative[0] / total:.1%})",
        (0, cumulative[0]),
        xytext=(160, -34),
        textcoords="offset points",
        arrowprops={"arrowstyle": "-", "color": "#64748B"},
    )
    ax.annotate(
        f"sim 2048: {at_action:,} ({at_action / total:.1%})",
        (2048, at_action),
        xytext=(20, -42),
        textcoords="offset points",
        color="#B91C1C",
        arrowprops={"arrowstyle": "-", "color": "#DC2626"},
    )
    ax.annotate(
        f"sim 4096: {total:,} (100%)",
        (TOTAL_SIMS, total),
        xytext=(-155, -36),
        textcoords="offset points",
        arrowprops={"arrowstyle": "-", "color": "#64748B"},
    )
    ax.set_title(
        "Cumulative final choice locked count\n"
        "A position enters the count after its last significant flip"
    )
    ax.set_xlabel("Simulation")
    ax.set_ylabel("Cumulative final choice locked count")
    ax.set_xlim(0, TOTAL_SIMS)
    ax.set_ylim(0, total * 1.06)
    ax.set_xticks(np.arange(0, TOTAL_SIMS + 1, 512))
    fig.tight_layout()
    fig.savefig(Path(__file__).with_name("cumulative_final_choice_locked_count.png"))
    plt.close(fig)


def main() -> None:
    configure_style()
    lock_sims = load_lock_sims()
    plot_newly_locked(lock_sims)
    plot_cumulative(lock_sims)


if __name__ == "__main__":
    main()
