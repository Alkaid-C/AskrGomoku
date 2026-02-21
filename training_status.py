#!/usr/bin/env python3
"""Quick training status summary from CSV logs."""

import argparse
import csv
import os
import sys


def read_csv_tail(path, n):
    """Read last n rows of a CSV file. Returns (headers, rows)."""
    if not os.path.exists(path):
        return None, []
    with open(path) as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames
        rows = list(reader)
    return headers, rows[-n:]


def read_csv_all(path):
    """Read all rows of a CSV file. Returns (headers, rows)."""
    if not os.path.exists(path):
        return None, []
    with open(path) as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames
        rows = list(reader)
    return headers, rows


def fmt(val, decimals=3):
    if val is None:
        return "—"
    return f"{val:.{decimals}f}"


def mean(values):
    if not values:
        return None
    return sum(values) / len(values)


def safe_floats(rows, key):
    out = []
    for r in rows:
        v = r.get(key, "")
        if v != "":
            try:
                out.append(float(v))
            except ValueError:
                pass
    return out


def main():
    parser = argparse.ArgumentParser(description="Training status summary")
    parser.add_argument("output_dir", help="Training output directory")
    args = parser.parse_args()

    d = args.output_dir

    # --- training_updates.csv ---
    tu_path = os.path.join(d, "training_updates.csv")
    _, tu_rows = read_csv_tail(tu_path, 100)
    if not tu_rows:
        print(f"No data in {tu_path}")
        sys.exit(1)

    current_update = int(tu_rows[-1]["update"])
    n = len(tu_rows)

    wr = mean(safe_floats(tu_rows, "win_rate"))
    wr_b = mean(safe_floats(tu_rows, "win_rate_black"))
    wr_w = mean(safe_floats(tu_rows, "win_rate_white"))
    ent = mean(safe_floats(tu_rows, "entropy"))
    gl = mean(safe_floats(tu_rows, "avg_game_length"))

    print(f"Update: {current_update}")
    print(f"Win rate (last {n}): {fmt(wr)}  (black {fmt(wr_b)}, white {fmt(wr_w)})")
    print(f"Entropy: {fmt(ent)}  Game length: {fmt(gl, 1)}")

    # --- eval_summary.csv ---
    es_path = os.path.join(d, "eval_summary.csv")
    _, es_rows = read_csv_tail(es_path, 100)
    if not es_rows:
        print(f"(No eval data)")
        return

    n_eval = len(es_rows)
    added_count = sum(
        1 for r in es_rows
        if r.get("checkpoint_added", "").lower() not in ("", "false", "0", "none")
    )
    print(f"Addition frequency: {added_count}/{n_eval} evals")

    # --- opponent list from eval_opponent_details.csv ---
    ed_path = os.path.join(d, "eval_opponent_details.csv")
    _, ed_rows = read_csv_all(ed_path)
    if not ed_rows:
        print("(No opponent detail data)")
        return

    last_eval_update = ed_rows[-1]["update"]
    opponents = [
        (int(r["opponent_id"]), float(r["win_rate"]))
        for r in ed_rows if r["update"] == last_eval_update
    ]
    opponents.sort(key=lambda x: x[1])

    print(f"Opponents (eval #{last_eval_update}, win rate vs each):")
    for oid, owr in opponents:
        print(f"  #{oid:<8} {owr:.3f}")


if __name__ == "__main__":
    main()
