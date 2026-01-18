#!/usr/bin/env python3
"""Extract training updates and evaluation results from pv4-4.log into CSV files."""

import re
import csv
from pathlib import Path


def parse_time_to_hours(time_str: str) -> float:
    """Convert time string like '66h53m', '8m07s', '554h02m' to hours."""
    hours = 0.0
    minutes = 0.0
    seconds = 0.0

    h_match = re.search(r'(\d+)h', time_str)
    m_match = re.search(r'(\d+)m', time_str)
    s_match = re.search(r'(\d+)s', time_str)

    if h_match:
        hours = float(h_match.group(1))
    if m_match:
        minutes = float(m_match.group(1))
    if s_match:
        seconds = float(s_match.group(1))

    total_hours = hours + minutes / 60 + seconds / 3600
    return round(total_hours, 2)


def extract_updates(log_content: str) -> list[dict]:
    """Extract all update records from log content."""
    updates = []

    # Pattern for main update line
    # Update    16/65536 | Loss: +6.6462 | WinRate: 52%(B55%-W50%) | AvgLen: 107.3 | Elapsed: 8m07s | ETA: 554h02m
    update_pattern = re.compile(
        r'Update\s+(\d+)/\d+\s*\|\s*Loss:\s*([+-]?\d+\.?\d*)\s*\|\s*'
        r'WinRate:\s*(\d+)%\(B(\d+)%-W(\d+)%\)\s*\|\s*'
        r'AvgLen:\s*([\d.]+)\s*\|\s*'
        r'Elapsed:\s*(\S+)\s*\|\s*'
        r'ETA:\s*(\S+)'
    )

    # Pattern for phase/tactics line
    # Phase: RAW | Tactics: W(39√512+) B(5√696+) | Imitate: 0(B0+W0)
    phase_pattern = re.compile(
        r'Phase:\s*(\S+)\s*\|\s*'
        r'Tactics:\s*W\((\d+)√(\d+)\+\)\s*B\((\d+)√(\d+)\+\)\s*\|\s*'
        r'Imitate:\s*(\d+)\(B(\d+)\+W(\d+)\)'
    )

    # Pattern for entropy/loss line
    # Entropy: 5.096 | V_loss: 0.0116 | Raw_MSE: 0.0116 | Time/Update: 30.33s (33%/67%)
    entropy_pattern = re.compile(
        r'Entropy:\s*([\d.]+)\s*\|\s*'
        r'V_loss:\s*([\d.]+)\s*\|\s*'
        r'Raw_MSE:\s*([\d.]+)\s*\|\s*'
        r'Time/Update:\s*([\d.]+)s'
    )

    lines = log_content.split('\n')
    i = 0
    while i < len(lines):
        line = lines[i]
        update_match = update_pattern.search(line)

        if update_match:
            record = {
                'update': int(update_match.group(1)),
                'loss': float(update_match.group(2)),
                'win_rate': int(update_match.group(3)),
                'win_rate_black': int(update_match.group(4)),
                'win_rate_white': int(update_match.group(5)),
                'avg_len': float(update_match.group(6)),
                'elapsed_h': parse_time_to_hours(update_match.group(7)),
                'eta_h': parse_time_to_hours(update_match.group(8)),
            }

            # Look for phase line (next line)
            if i + 1 < len(lines):
                phase_match = phase_pattern.search(lines[i + 1])
                if phase_match:
                    record['phase'] = phase_match.group(1)
                    record['win_in_1_hit'] = int(phase_match.group(2))
                    record['win_in_1_miss'] = int(phase_match.group(3))
                    record['must_block_hit'] = int(phase_match.group(4))
                    record['must_block_miss'] = int(phase_match.group(5))
                    record['imitate_total'] = int(phase_match.group(6))
                    record['imitate_black'] = int(phase_match.group(7))
                    record['imitate_white'] = int(phase_match.group(8))

            # Look for entropy line (next line after phase)
            if i + 2 < len(lines):
                entropy_match = entropy_pattern.search(lines[i + 2])
                if entropy_match:
                    record['entropy'] = float(entropy_match.group(1))
                    record['v_loss'] = float(entropy_match.group(2))
                    record['raw_mse'] = float(entropy_match.group(3))
                    record['time_per_update_s'] = float(entropy_match.group(4))

            updates.append(record)
            i += 3  # Skip the lines we just processed
        else:
            i += 1

    return updates


def extract_evals(log_content: str) -> list[dict]:
    """Extract all evaluation records from log content."""
    evals = []

    # Pattern for eval header
    # --- Evaluation at update 576 ---
    eval_header_pattern = re.compile(r'--- Evaluation at update (\d+) ---')

    # Pattern for win rate line
    # Win rate against pool: 0.748 (480 games) | Eval time: 34.8s
    winrate_pattern = re.compile(
        r'Win rate against pool:\s*([\d.]+)\s*\((\d+) games\)\s*\|\s*Eval time:\s*([\d.]+)s'
    )

    # Pattern for pool update status
    # Win rate 0.748 >= 0.625, updating opponent pool
    # Win rate 0.579 < 0.625, not updating pool
    pool_update_pattern = re.compile(r'Win rate [\d.]+ ([<>=]+) 0\.625')

    # Pattern for next eval
    # Next eval at update 640 (interval: 64)
    next_eval_pattern = re.compile(r'Next eval at update \d+ \(interval: (\d+)\)')

    lines = log_content.split('\n')
    i = 0
    while i < len(lines):
        line = lines[i]
        header_match = eval_header_pattern.search(line)

        if header_match:
            record = {
                'update': int(header_match.group(1)),
                'win_rate_pool': None,
                'games': None,
                'eval_time_s': None,
                'pool_updated': None,
                'interval': None,
            }

            # Search next few lines for the other patterns
            for j in range(i + 1, min(i + 8, len(lines))):
                check_line = lines[j]

                winrate_match = winrate_pattern.search(check_line)
                if winrate_match:
                    record['win_rate_pool'] = float(winrate_match.group(1))
                    record['games'] = int(winrate_match.group(2))
                    record['eval_time_s'] = float(winrate_match.group(3))

                pool_match = pool_update_pattern.search(check_line)
                if pool_match:
                    record['pool_updated'] = '>=' in pool_match.group(1)

                next_match = next_eval_pattern.search(check_line)
                if next_match:
                    record['interval'] = int(next_match.group(1))

            evals.append(record)
            i += 1
        else:
            i += 1

    return evals


def main():
    log_path = Path(__file__).parent / 'pv4-4.log'

    print(f"Reading log file: {log_path}")
    with open(log_path, 'r') as f:
        log_content = f.read()

    # Extract updates
    print("Extracting update records...")
    updates = extract_updates(log_content)
    print(f"  Found {len(updates)} update records")

    # Extract evals
    print("Extracting evaluation records...")
    evals = extract_evals(log_content)
    print(f"  Found {len(evals)} evaluation records")

    # Write updates CSV
    updates_csv = Path(__file__).parent / 'updates.csv'
    update_columns = [
        'update', 'loss', 'win_rate', 'win_rate_black', 'win_rate_white',
        'avg_len', 'elapsed_h', 'eta_h', 'phase',
        'win_in_1_hit', 'win_in_1_miss', 'must_block_hit', 'must_block_miss',
        'imitate_total', 'imitate_black', 'imitate_white',
        'entropy', 'v_loss', 'raw_mse', 'time_per_update_s'
    ]

    with open(updates_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=update_columns)
        writer.writeheader()
        writer.writerows(updates)
    print(f"Written: {updates_csv}")

    # Write evals CSV
    evals_csv = Path(__file__).parent / 'evals.csv'
    eval_columns = ['update', 'win_rate_pool', 'games', 'eval_time_s', 'pool_updated', 'interval']

    with open(evals_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=eval_columns)
        writer.writeheader()
        writer.writerows(evals)
    print(f"Written: {evals_csv}")

    # Print sample records
    print("\n--- Sample Update Record ---")
    if updates:
        for k, v in updates[0].items():
            print(f"  {k}: {v}")

    print("\n--- Sample Eval Record ---")
    if evals:
        for k, v in evals[0].items():
            print(f"  {k}: {v}")


if __name__ == '__main__':
    main()
