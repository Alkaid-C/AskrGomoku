#!/usr/bin/env python3
"""
Piskvork-protocol adapter for our Gomoku network.

Wraps a trained policy/value checkpoint as a Gomocup-compatible engine that
speaks the Piskvork text protocol (https://plastovicka.github.io/protocl2en.htm)
over stdin/stdout, so it can be driven by Gomocup managers or our own arena_web.py
and pitted against external engines such as Rapfi.

Move selection mirrors deployment behaviour:
  * MCTS_BUDGET > 0 -> AlphaZero PUCT search via mcts/mcts.py (raw masked-softmax
    priors, no Dirichlet noise = deterministic competitive play). Requires the
    pre-compiled mcts_ext.so; if the import fails we warn to stderr and fall back
    to raw policy.
  * MCTS_BUDGET == 0 -> raw policy-head argmax (single forward, no search).

All model loading and inference mirror mcts/play_web.py and main/gomoku.py.
"""

import os
import sys

# ============================================================================
# Configuration (edit these)
# ============================================================================

# Paths are resolved relative to THIS file (pbrain.py runs as a subprocess with
# an arbitrary cwd), so they must not depend on the launch directory.
CHECKPOINT_PATH = "../mcts/test2/stage2/checkpoint_update_26.pt"
MODEL_DIR = "../main"   # directory containing model.py / gomoku.py
MCTS_BUDGET = 2048       # 0 = raw policy argmax; >0 = MCTS simulations per move

# MCTS search knobs (used only when MCTS_BUDGET > 0). Match mcts/play_web.py /
# stage-2 deployment: raw priors, deterministic (no root noise).
C_PUCT = 1.25
GAMMA = 63.0 / 64.0
FPU_MULTIPLIER = 0.95
DIRICHLET_ALPHA = 0.125
DIRICHLET_EPSILON = 0.0  # 0 = no noise = deterministic

BOARD_SIZE = 15

# ============================================================================
# Import resolution
# ============================================================================

_HERE = os.path.dirname(os.path.abspath(__file__))
_MODEL_DIR_ABS = os.path.normpath(os.path.join(_HERE, MODEL_DIR))
_MCTS_DIR_ABS = os.path.normpath(os.path.join(_HERE, "..", "mcts"))
_CHECKPOINT_ABS = os.path.normpath(os.path.join(_HERE, CHECKPOINT_PATH))

# MODEL_DIR first so model.py / gomoku.py resolve there; mcts dir for mcts.py,
# mcts_ext, entropy_ops. main/ and mcts/ share identical gomoku.py / model.py
# (symlinks), so ordering is safe.
sys.path.insert(0, _MCTS_DIR_ABS)
sys.path.insert(0, _MODEL_DIR_ABS)

import numpy as np
import torch
from gomoku import (
    Player,
    board_from_observation,
    encode_observation,
    idx_to_pos,
    select_action_batch_eval,
)
from model import GomokuPolicyNet

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# MCTS is optional: it needs the compiled mcts_ext.so. If anything fails to
# import we fall back to raw policy and warn on stderr.
_MCTS_AVAILABLE = False
mcts_search_batched = None
if MCTS_BUDGET > 0:
    try:
        from mcts import mcts_search_batched as _mcts_search

        mcts_search_batched = _mcts_search
        _MCTS_AVAILABLE = True
    except Exception as e:  # pragma: no cover - depends on build env
        print(
            f"MESSAGE pbrain: MCTS unavailable ({e}); falling back to raw policy",
            file=sys.stderr,
            flush=True,
        )


def log(msg: str) -> None:
    """Diagnostic to stderr (never pollutes the protocol stdout channel)."""
    print(f"pbrain: {msg}", file=sys.stderr, flush=True)


# ============================================================================
# Model
# ============================================================================


def load_model() -> GomokuPolicyNet:
    checkpoint = torch.load(_CHECKPOINT_ABS, map_location="cpu", weights_only=False)
    model = GomokuPolicyNet()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    model.to(DEVICE)
    return model


# ============================================================================
# Engine state
# ============================================================================


class Engine:
    """Holds the board (from this engine's own perspective) and picks moves."""

    def __init__(self, model: GomokuPolicyNet):
        self.model = model
        self.reset()

    def reset(self) -> None:
        # mine = this engine's stones, theirs = opponent's. Perspective-based,
        # matching the network's canonical input (channel 0 = side-to-move).
        self.mine = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.uint8)
        self.theirs = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.uint8)

    def set_stone(self, x: int, y: int, is_mine: bool) -> None:
        # Piskvork coords: x = col, y = row.
        if is_mine:
            self.mine[y, x] = 1
        else:
            self.theirs[y, x] = 1

    def _legal_mask(self) -> np.ndarray:
        return ((self.mine == 0) & (self.theirs == 0)).astype(np.uint8)

    def choose_move(self) -> tuple[int, int]:
        """Return (x, y) for the engine's move from the current position."""
        obs = encode_observation(self.mine, self.theirs)
        legal_mask = self._legal_mask()

        if _MCTS_AVAILABLE and mcts_search_batched is not None:
            board = board_from_observation(obs, Player.BLACK)
            visit_dists = mcts_search_batched(
                self.model,
                [board],
                num_simulations=MCTS_BUDGET,
                c_puct=C_PUCT,
                entropy_multiplier=None,
                device=DEVICE,
                dirichlet_alpha=DIRICHLET_ALPHA,
                dirichlet_epsilon=DIRICHLET_EPSILON,
                gamma=GAMMA,
                fpu_multiplier=FPU_MULTIPLIER,
                harvest_min_visits=None,
            )[0]
            idx = int(visit_dists[0].argmax())
        else:
            idx = select_action_batch_eval(
                self.model, [obs], [legal_mask], 0.0, DEVICE, deterministic=True
            )[0]

        row, col = idx_to_pos(idx)
        # Both paths mask illegal squares before choosing, so an occupied result
        # means the search or the mask is broken. Fail loudly rather than
        # substituting a legal square, which would hide the defect behind a
        # plausible-looking game.
        if self.mine[row, col] or self.theirs[row, col]:
            raise RuntimeError(
                f"engine chose occupied square ({col},{row}) "
                f"[flat {idx}]; search or legal mask is broken"
            )
        return col, row  # (x, y)

    def play_own(self, x: int, y: int) -> None:
        self.mine[y, x] = 1


# ============================================================================
# Protocol loop
# ============================================================================


def respond_with_move(engine: Engine) -> None:
    x, y = engine.choose_move()
    engine.play_own(x, y)
    print(f"{x},{y}", flush=True)


def handle_board(engine: Engine, stdin) -> None:
    """Read `x,y,who` lines until DONE, then move. who: 1=mine, else opponent."""
    engine.reset()
    for line in stdin:
        line = line.strip()
        if not line:
            continue
        if line.upper() == "DONE":
            break
        parts = line.split(",")
        if len(parts) < 3:
            continue
        try:
            x, y, who = int(parts[0]), int(parts[1]), int(parts[2])
        except ValueError:
            continue
        engine.set_stone(x, y, is_mine=(who == 1))
    respond_with_move(engine)


def main() -> None:
    log(f"device={DEVICE}, mcts={'on' if _MCTS_AVAILABLE else 'off'}, budget={MCTS_BUDGET}")
    model = load_model()
    engine = Engine(model)
    log(f"loaded {_CHECKPOINT_ABS}")

    stdin = sys.stdin
    for raw in stdin:
        line = raw.strip()
        if not line:
            continue
        upper = line.upper()
        cmd = upper.split()[0] if upper.split() else ""

        if cmd == "START":
            parts = line.split()
            size = int(parts[1]) if len(parts) > 1 else BOARD_SIZE
            if size != BOARD_SIZE:
                print(f"ERROR unsupported board size {size}", flush=True)
            else:
                engine.reset()
                print("OK", flush=True)
        elif cmd == "RECTSTART":
            print("ERROR rectangular board is not supported", flush=True)
        elif cmd == "RESTART":
            engine.reset()
            print("OK", flush=True)
        elif cmd == "BEGIN":
            respond_with_move(engine)
        elif cmd == "TURN":
            coords = line.split(None, 1)[1] if len(line.split(None, 1)) > 1 else ""
            x_str, y_str = coords.split(",")
            engine.set_stone(int(x_str), int(y_str), is_mine=False)
            respond_with_move(engine)
        elif cmd == "BOARD":
            handle_board(engine, stdin)
        elif cmd == "INFO":
            pass  # parameters (timeout, memory, ...) ignored
        elif cmd == "ABOUT":
            print(
                'name="AskrGomoku", version="1.0", author="Alkaid-C", '
                'country="-", www="https://github.com/Alkaid-C/AskrGomoku"',
                flush=True,
            )
        elif cmd == "END":
            break
        else:
            print(f"UNKNOWN command: {line}", flush=True)


if __name__ == "__main__":
    main()
