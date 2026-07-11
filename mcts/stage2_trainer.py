"""
Stage 2 trainer: vanilla AlphaZero MCTS self-play.

The student loaded from stage 1 plays itself with raw priors (no entropy
multiplier, no sharpening); Dirichlet noise at the root is the sole
exploration source. Targets are the raw MCTS visit distributions.
"""

import json
import os
import pickle
import random
import time
from collections import deque
from typing import Optional, Tuple

import numpy as np
import torch
from csv_logger import Stage2CSVLogger
from gomoku import RENJU_OPENING_SEQUENCES, GameState
from model import GomokuPolicyNet
from self_play import compute_block_rates, play_mcts_games
from training import train_on_mcts_batch

from mcts import clear_nn_eval_cache, get_nn_eval_cache_stats

TRAINING_STATE_FILE = "training_state.json"
FINAL_NAME = "final_policy.pt"
REPLAY_BUFFER_FILE_TEMPLATE = "replay_buffer_update_{update}.pkl"


class StaircaseLRController:
    """Plateau-driven staircase LR schedule for stage 2.

    Holds a constant LR until ``avg_raw_mcts_kl`` plateaus, then multiplies the LR by
    ``stair_factor``; after ``total_stairs`` descents the next plateau ends the stage
    (``finished`` flips True). A plateau is a trailing-window linear regression whose
    decline rate ``-slope`` falls below ``min_improve_speed * stair_factor**stairs_descended``.

    ``record`` is called once per completed update (1-indexed ``step``). Plateau checks
    fire only on the cadence grid: first at ``regression_range``, then every
    ``regression_interval`` updates; a stair-down resets the next check to
    ``step + regression_range`` so each regression window holds only post-drop data.

    The final stair (once ``stairs_descended == total_stairs``, whose plateau ends the
    stage irreversibly) uses a longer window ``regression_range * stop_regression_range_multiplier``
    for extra margin on that one-way decision; earlier stairs use ``regression_range``.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        base_lr: float,
        stair_factor: float,
        total_stairs: int,
        regression_range: int,
        stop_regression_range_multiplier: int,
        regression_interval: int,
        min_improve_speed: float,
    ) -> None:
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.stair_factor = stair_factor
        self.total_stairs = total_stairs
        self.regression_range = regression_range
        self.stop_regression_range_multiplier = stop_regression_range_multiplier
        self.regression_interval = regression_interval
        self.min_improve_speed = min_improve_speed
        self.kl_history: list[float] = []
        self.stairs_descended = 0
        self.next_check_step = self._current_range()
        self.finished = False
        # Diagnostics from the most recent plateau check (for logging; not persisted).
        self.last_check_step = -1
        self.last_slope = 0.0
        self.last_threshold = 0.0
        self._set_lr(base_lr)

    def _current_range(self) -> int:
        if self.stairs_descended >= self.total_stairs:
            return self.regression_range * self.stop_regression_range_multiplier
        return self.regression_range

    def _set_lr(self, lr: float) -> None:
        for group in self.optimizer.param_groups:
            group['lr'] = lr

    def record(self, step: int, kl: float) -> None:
        self.kl_history.append(kl)
        if step != self.next_check_step:
            return
        rng = self._current_range()
        window = self.kl_history[-rng:]
        slope = float(np.polyfit(np.arange(rng), window, 1)[0])
        improve_speed = -slope
        threshold = self.min_improve_speed * self.stair_factor ** self.stairs_descended
        self.last_check_step = step
        self.last_slope = slope
        self.last_threshold = threshold
        if improve_speed < threshold:  # plateau
            if self.stairs_descended >= self.total_stairs:
                self.finished = True
                return
            self.stairs_descended += 1
            self._set_lr(self.base_lr * self.stair_factor ** self.stairs_descended)
            self.next_check_step = step + self._current_range()
        else:
            self.next_check_step = step + self.regression_interval

    def state_dict(self) -> dict:
        return {
            'kl_history': self.kl_history,
            'stairs_descended': self.stairs_descended,
            'next_check_step': self.next_check_step,
            'finished': self.finished,
        }

    def load_state_dict(self, state: dict) -> None:
        self.kl_history = list(state['kl_history'])
        self.stairs_descended = state['stairs_descended']
        self.next_check_step = state['next_check_step']
        self.finished = state['finished']
        # Re-apply the LR deterministically from the stair count rather than trusting
        # the reloaded optimizer's saved LR.
        self._set_lr(self.base_lr * self.stair_factor ** self.stairs_descended)


def _save_checkpoint(
    path: str,
    update: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    controller: StaircaseLRController,
) -> None:
    torch.save({
        'update': update,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'lr_controller_state_dict': controller.state_dict(),
        # Global RNG states so resume continues the same stream rather than
        # restarting from seed_everything(SEED) (which main() re-applies on every
        # process start). All stage-2 randomness flows through these globals:
        # python `random` (openings, replay sampling) and `np.random` (Dirichlet
        # noise, action sampling); torch/cuda are near-static here but cheap to
        # carry. Not bit-exact overall (GPU FP nondeterminism diverges the
        # trajectory), but avoids repeating the seed-42 stream across resumes.
        'rng_state': {
            'python': random.getstate(),
            'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'torch_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        },
    }, path)


def _save_state(output_dir: str, update: int) -> None:
    state = {'current_update': update}
    path = os.path.join(output_dir, TRAINING_STATE_FILE)
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, path)


def _buffer_path(output_dir: str, update: int) -> str:
    return os.path.join(output_dir, REPLAY_BUFFER_FILE_TEMPLATE.format(update=update))


def _save_buffer(output_dir: str, update: int, replay_buffer: "deque[list]") -> None:
    """Persist the replay buffer next to the matching checkpoint.

    The file is keyed by update so a crash between buffer write and
    training_state.json update cannot make an older checkpoint resume with a
    newer replay buffer.
    """
    path = _buffer_path(output_dir, update)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump({
            "update": update,
            "rounds": list(replay_buffer),
        }, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


def _load_buffer(output_dir: str, update: int) -> Optional[list]:
    """Reload the replay buffer sidecar for `update`.

    Returns None if the matching file is absent, e.g. for checkpoints predating
    this feature. The caller then falls back to an empty buffer.
    """
    path = _buffer_path(output_dir, update)
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        payload = pickle.load(f)
    if not isinstance(payload, dict) or payload.get("update") != update:
        raise RuntimeError(f"Replay buffer sidecar does not match update {update}: {path}")
    return payload["rounds"]


def _dump_samples(samples_dir: str, update: int, plies: list[tuple]) -> None:
    """Persist one update's flattened per-ply samples to a .npz shard for
    offline analysis. Side-channel only — does not affect training. Each ply is
    (obs uint8[3,15,15], dist f32[225], value f32, policy_weight, value_weight);
    columns are stacked into parallel arrays. Crash-safe via tmp + rename."""
    obs = np.stack([p[0] for p in plies]).astype(np.uint8)
    dist = np.stack([p[1] for p in plies]).astype(np.float32)
    value = np.array([p[2] for p in plies], dtype=np.float32)
    policy_weight = np.array([p[3] for p in plies], dtype=np.float32)
    value_weight = np.array([p[4] for p in plies], dtype=np.float32)
    path = os.path.join(samples_dir, f"samples_update_{update}.npz")
    tmp = path + ".tmp.npz"  # keep .npz suffix so np.savez writes this exact name
    np.savez_compressed(
        tmp, obs=obs, dist=dist, value=value,
        policy_weight=policy_weight, value_weight=value_weight,
    )
    os.replace(tmp, path)


def _fmt_time(s: float) -> str:
    h = int(s // 3600)
    m = int((s % 3600) // 60)
    if h > 0:
        return f"{h}h{m:02d}m"
    return f"{m}m{int(s % 60):02d}s"


def _try_resume(
    output_dir: str,
    learning_rate: float,
    stair_factor: float,
    total_stairs: int,
    regression_range: int,
    stop_regression_range_multiplier: int,
    regression_interval: int,
    min_improve_speed: float,
    weight_decay: float,
    device: torch.device,
) -> Optional[Tuple[
    torch.nn.Module,
    torch.optim.Optimizer,
    StaircaseLRController,
    int,
    Optional[list],
]]:
    path = os.path.join(output_dir, TRAINING_STATE_FILE)
    if not os.path.exists(path):
        return None

    with open(path) as f:
        state = json.load(f)
    current_update = state['current_update']
    print(f"Resuming stage 2 from update {current_update}")

    ckpt_path = os.path.join(output_dir, f"checkpoint_update_{current_update}.pt")
    if not os.path.exists(ckpt_path):
        raise RuntimeError(f"Checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = GomokuPolicyNet().to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay, fused=True
    )
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    controller = StaircaseLRController(
        optimizer, learning_rate, stair_factor, total_stairs,
        regression_range, stop_regression_range_multiplier,
        regression_interval, min_improve_speed,
    )
    controller.load_state_dict(ckpt['lr_controller_state_dict'])
    model.train()

    # Restore global RNG states. Must run *after* GomokuPolicyNet() construction
    # above (weight init consumes the torch RNG stream); it also overrides the
    # seed_everything(SEED) that main() applied before dispatching here. Absent
    # on checkpoints predating this feature -> keep the seeded stream.
    rng = ckpt.get('rng_state')
    if rng is not None:
        random.setstate(rng['python'])
        np.random.set_state(rng['numpy'])
        torch.set_rng_state(rng['torch'])
        cuda_states = rng.get('torch_cuda', [])
        if cuda_states and torch.cuda.is_available() \
                and len(cuda_states) == torch.cuda.device_count():
            torch.cuda.set_rng_state_all(cuda_states)
        print("  Restored RNG state (resume continues the same random stream)")
    else:
        print("  No RNG state in checkpoint; continuing from the seeded stream")

    buffer_rounds = _load_buffer(output_dir, current_update)
    if buffer_rounds is None:
        print("  No replay buffer sidecar found; resuming with an empty buffer "
              "(warmup will re-engage for the next few updates)")
    else:
        n_plies = sum(len(rd) for rd in buffer_rounds)
        print(f"  Restored replay buffer: {len(buffer_rounds)} rounds, {n_plies} plies")
    return model, optimizer, controller, current_update, buffer_rounds


def run_stage2(
    stage1_checkpoint: str,
    output_dir: str,
    *,
    episodes_per_update: int,
    num_simulations: int,
    c_puct: float,
    dirichlet_alpha: float,
    dirichlet_epsilon: float,
    action_temperature: float,
    seed_probability: float,
    gamma: float,
    fpu_multiplier: float,
    learning_rate: float,
    stair_factor: float,
    total_stairs: int,
    regression_range: int,
    stop_regression_range_multiplier: int,
    regression_interval: int,
    min_improve_speed: float,
    weight_decay: float,
    value_loss_coeff: float,
    optimize_steps_per_update: int,
    replay_buffer_rounds: int,
    sample_ratio: float,
    decay_ratio: float,
    sample_dump_updates: int,
    checkpoint_interval: int,
    harvest_value_min_visits: int,
    harvest_policy_min_visits: int,
    device: torch.device,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    samples_dir = os.path.join(output_dir, "samples")
    if sample_dump_updates > 0:
        os.makedirs(samples_dir, exist_ok=True)
    csv_logger = Stage2CSVLogger(output_dir)

    print(f"Output: {output_dir}")
    print(f"MCTS: {num_simulations} sims, c_puct={c_puct}, gamma={gamma}")
    print(f"Dirichlet: alpha={dirichlet_alpha}, epsilon={dirichlet_epsilon}")
    print(f"Training: {episodes_per_update} games/update, replay buffer={replay_buffer_rounds} rounds (runs until the staircase finishes)")
    print(f"Replay sampling: k_0=sample_ratio*len(round_0)={sample_ratio}, decay={decay_ratio:.4f}")
    print(f"LR: {learning_rate} (staircase, x{stair_factor} per plateau, up to {total_stairs} stairs)")
    print()

    resumed = _try_resume(
        output_dir, learning_rate, stair_factor, total_stairs,
        regression_range, stop_regression_range_multiplier,
        regression_interval, min_improve_speed, weight_decay, device,
    )
    if resumed is not None:
        model, optimizer, controller, start_update, resumed_buffer = resumed
    else:
        resumed_buffer = None
        if not os.path.exists(stage1_checkpoint):
            raise RuntimeError(
                f"Missing stage 1 final: {stage1_checkpoint} "
                f"(required for fresh start; resume needs {output_dir}/{TRAINING_STATE_FILE})"
            )
        print(f"Loading stage 1 checkpoint: {stage1_checkpoint}")
        ckpt = torch.load(stage1_checkpoint, map_location=device, weights_only=False)
        model = GomokuPolicyNet().to(device)
        model.load_state_dict(ckpt['model_state_dict'])
        model.train()
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay, fused=True
        )
        controller = StaircaseLRController(
            optimizer, learning_rate, stair_factor, total_stairs,
            regression_range, stop_regression_range_multiplier,
            regression_interval, min_improve_speed,
        )
        start_update = 0

    training_start = time.time()
    num_openings = len(RENJU_OPENING_SEQUENCES)
    replay_buffer: deque[list] = deque(maxlen=replay_buffer_rounds)
    if resumed_buffer is not None:
        # deque(maxlen) keeps only the last `replay_buffer_rounds` if the saved
        # buffer is longer (e.g. after a config change); normally it matches.
        replay_buffer.extend(resumed_buffer)
    geom_sum = sum(decay_ratio ** i for i in range(replay_buffer_rounds))

    # No fixed horizon: the staircase controller ends the stage when it flags `finished`
    # (KL decline is finite and bounded below by 0). `update` is 0-indexed; if the loop
    # never runs (resumed already-finished) last_update stays at start_update.
    update = start_update - 1
    while not controller.finished:
        update += 1
        t_start = time.time()

        opening_ids: list[int] = []
        for _ in range(episodes_per_update):
            if random.random() < seed_probability:
                opening_ids.append(random.randint(0, num_openings - 1))
            else:
                opening_ids.append(-1)

        t0 = time.time()
        model.eval()
        records = play_mcts_games(
            model=model,
            num_games=episodes_per_update,
            num_simulations=num_simulations,
            c_puct=c_puct,
            entropy_multiplier=None,
            device=device,
            opening_ids=opening_ids,
            dirichlet_alpha=dirichlet_alpha,
            dirichlet_epsilon=dirichlet_epsilon,
            gamma=gamma,
            fpu_multiplier=fpu_multiplier,
            action_temperature=action_temperature,
            harvest_min_visits=harvest_value_min_visits,
            harvest_policy_min_visits=harvest_policy_min_visits,
        )
        block_stats = compute_block_rates(records, model, device)
        model.train()
        t_selfplay = time.time() - t0

        game_lengths = []
        black_wins = 0
        draws = 0
        sum_raw_H = 0.0
        sum_mcts_H = 0.0
        sum_raw_mcts_kl = 0.0
        n_plies = 0
        harvest_total = 0
        harvest_value_only = 0
        harvest_weight_sum = 0.0
        for record in records:
            game_lengths.append(len(record.observations))
            assert record.outcome is not None, "Game did not terminate"
            if record.outcome == GameState.DRAW:
                draws += 1
            elif record.outcome == GameState.BLACK_WIN:
                black_wins += 1
            sum_raw_H += sum(record.raw_entropy)
            sum_raw_mcts_kl += sum(record.raw_mcts_kl)
            for vd in record.visit_distributions:
                sum_mcts_H += float(-(vd * np.log(vd + 1e-30)).sum())
            n_plies += len(record.observations)
            for harvested_sample in record.harvested:
                policy_w, value_w = harvested_sample[3], harvested_sample[4]
                harvest_total += 1
                harvest_weight_sum += float(value_w)
                if float(policy_w) == 0.0:
                    harvest_value_only += 1

        harvest_value_only_frac = harvest_value_only / harvest_total if harvest_total > 0 else 0.0
        harvest_mean_weight = harvest_weight_sum / harvest_total if harvest_total > 0 else 0.0

        black_win_rate = black_wins / episodes_per_update
        draw_rate = draws / episodes_per_update
        avg_game_length = float(np.mean(game_lengths)) if game_lengths else 0.0
        avg_raw_entropy = sum_raw_H / n_plies if n_plies > 0 else 0.0
        avg_mcts_entropy = sum_mcts_H / n_plies if n_plies > 0 else 0.0
        avg_raw_mcts_kl = sum_raw_mcts_kl / n_plies if n_plies > 0 else 0.0

        cache_hits, cache_misses = get_nn_eval_cache_stats()
        cache_total = cache_hits + cache_misses
        cache_hit_rate = cache_hits / cache_total if cache_total > 0 else 0.0
        current_lr = optimizer.param_groups[0]['lr']

        # Drop the self-play NN eval cache before training so the PyTorch
        # allocator can reuse those blocks for training activations.
        clear_nn_eval_cache()

        # Flatten records to per-ply samples for the replay buffer. Each sample
        # is (obs, dist, value, policy_weight, value_weight); played roots use
        # (1.0, 1.0), harvested nodes carry their own weights. Ply-level
        # sampling decorrelates the batch (consecutive plies of the same game
        # are highly correlated) compared to sampling whole records.
        plies: list[tuple] = []
        for record in records:
            for obs, dist, val in zip(
                record.observations, record.visit_distributions, record.root_values
            ):
                plies.append((obs, dist, float(val), 1.0, 1.0))
            for h_obs, h_policy, h_value, h_policy_w, h_value_w in record.harvested:
                plies.append((h_obs, h_policy, float(h_value), float(h_policy_w), float(h_value_w)))

        # Side-channel: dump the first N updates' samples for offline analysis.
        # Pure side effect on `plies` (already built) — training is untouched.
        if update < sample_dump_updates:
            _dump_samples(samples_dir, update, plies)

        replay_buffer.append(plies)
        # Warmup (buffer not yet full): draw from the most recent round only, at
        # a ratio scaled by the geometric sum so the per-step sample total matches
        # the post-warmup budget (Σ_i sample_ratio * decay_ratio**i).
        # Post-warmup: per-round decaying budget walking newest -> oldest:
        #   k_0 = round(sample_ratio * len(round_0))   # plies, not games
        #   k_i = round(k_0 * decay_ratio ** i), clamped to [0, len(round_i)]; skip if 0.
        if len(replay_buffer) < replay_buffer_rounds:
            ordered_rounds: list[list] = [plies]
            per_round_k = [min(len(plies), round(sample_ratio * geom_sum * len(plies)))]
        else:
            ordered_rounds = list(reversed(replay_buffer))
            k_0 = round(sample_ratio * len(ordered_rounds[0]))
            per_round_k = [
                max(0, min(len(rd), round(k_0 * decay_ratio ** i)))
                for i, rd in enumerate(ordered_rounds)
            ]

        t0 = time.time()
        sum_policy_loss = 0.0
        sum_value_loss = 0.0
        for _ in range(optimize_steps_per_update):
            step_samples: list = []
            for rd, k_i in zip(ordered_rounds, per_round_k):
                if k_i > 0:
                    step_samples.extend(random.sample(rd, k_i))
            step_results = train_on_mcts_batch(
                model, step_samples, optimizer, device, value_loss_coeff=value_loss_coeff,
            )
            sum_policy_loss += step_results['policy_loss']
            sum_value_loss += step_results['value_loss']
        train_results = {
            'policy_loss': sum_policy_loss / optimize_steps_per_update,
            'value_loss': sum_value_loss / optimize_steps_per_update,
        }
        t_train = time.time() - t0
        # Return freed blocks to CUDA after training (training produces the
        # largest transient allocations).
        torch.cuda.empty_cache()

        # Update the staircase LR from this round's KL; may halve the LR or, after the
        # final stair, flag the stage finished (handled after logging/checkpointing).
        controller.record(update + 1, avg_raw_mcts_kl)

        elapsed = time.time() - training_start
        print(
            f"Update {update+1:4d} (stair {controller.stairs_descended}/{total_stairs}) | "
            f"BlackWR: {black_win_rate:.0%} D: {draw_rate:.0%} | "
            f"Len: {avg_game_length:.1f} | "
            f"H: r{avg_raw_entropy:.2f}/m{avg_mcts_entropy:.2f} | "
            f"rmKL: {avg_raw_mcts_kl:.3f} | "
            f"PLoss: {train_results['policy_loss']:.4f} VLoss: {train_results['value_loss']:.4f} | "
            f"{(time.time() - t_start):.1f}s | "
            f"{_fmt_time(elapsed)}"
        )

        if controller.last_check_step == update + 1:
            plateau = -controller.last_slope < controller.last_threshold
            print(
                f"  Plateau check: KL slope {controller.last_slope:+.4f} "
                f"(improve {-controller.last_slope:+.4f} vs thr {controller.last_threshold:.4f}) "
                f"-> {'PLATEAU' if plateau else 'continue'}"
            )

        csv_logger.log(update + 1, {
            'policy_loss': train_results['policy_loss'],
            'value_loss': train_results['value_loss'],
            'lr': current_lr,
            'avg_game_length': avg_game_length,
            'avg_raw_entropy': avg_raw_entropy,
            'avg_mcts_entropy': avg_mcts_entropy,
            'avg_raw_mcts_kl': avg_raw_mcts_kl,
            'black_win_rate': black_win_rate,
            'draw_rate': draw_rate,
            'time_selfplay': t_selfplay,
            'time_train': t_train,
            'cache_hit_rate': cache_hit_rate,
            'cache_hits': cache_hits,
            'cache_misses': cache_misses,
            'harvest_samples': harvest_total,
            'harvest_value_only_frac': harvest_value_only_frac,
            'harvest_mean_weight': harvest_mean_weight,
            **block_stats,
        })

        if (update + 1) % checkpoint_interval == 0:
            ckpt_id = update + 1
            ckpt_path = os.path.join(output_dir, f"checkpoint_update_{ckpt_id}.pt")
            _save_checkpoint(ckpt_path, ckpt_id, model, optimizer, controller)
            # Order matters: .pt, then buffer, then state last (its atomic write
            # is the commit point that guarantees both prior files are complete).
            _save_buffer(output_dir, ckpt_id, replay_buffer)
            _save_state(output_dir, ckpt_id)
            print(f"  Saved: {os.path.basename(ckpt_path)}")

        if controller.finished:  # while-loop exits after this iteration
            print(f"  Final plateau reached after {controller.stairs_descended} stairs; ending stage 2.")

    last_update = update + 1
    final_ckpt = os.path.join(output_dir, f"checkpoint_update_{last_update}.pt")
    _save_checkpoint(final_ckpt, last_update, model, optimizer, controller)
    final_path = os.path.join(output_dir, FINAL_NAME)
    torch.save({'model_state_dict': model.state_dict(), 'update': last_update}, final_path)
    _save_buffer(output_dir, last_update, replay_buffer)
    _save_state(output_dir, last_update)
    print(f"\nStage 2 complete! Final model: {final_path}")
