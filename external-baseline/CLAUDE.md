# External Baseline — Piskvork Adapter, Visual Arena & Rapfi

Lets our trained Gomoku engine play against external engines over the Gomocup
**Piskvork** protocol (https://plastovicka.github.io/protocl2en.htm), with a web
UI to watch and control matches. Built primarily to benchmark our network against
**Rapfi**, the strongest open-source Gomoku/Renju engine.

## Files

- **`pbrain.py`** — wraps one of our checkpoints as a Piskvork-protocol engine
  (stdin/stdout). Drop-in opponent for any Gomocup manager or for `arena_web.py`.
- **`arena_web.py`** — Flask web arena (port 5001) that launches two
  Piskvork engines as subprocesses, relays moves between them, and visualizes the
  game step by step. Embedded-HTML, same pattern as `main/play_web.py`.
- **`rapfi-engine/`** — the compiled Rapfi binary (`pbrain-rapfi`) staged together
  with its `config.toml` + NNUE/classical weight files (Rapfi auto-loads files
  from the directory containing the executable).
- **`rapfi/`** — Rapfi source checkout (cloned `--recursive`); build tree under
  `rapfi/Rapfi/build-avx2/`.
- **`RAPFI_MOVE_EMISSION_ISSUE.md`** — post-mortem of a subprocess-I/O bug we hit
  (kept as a worked example; see "Subprocess I/O" below).

## Running

```bash
# 1. Our engine as a Piskvork engine (from external-baseline/)
echo -e "START 15\nBEGIN\nEND" | python3 pbrain.py     # -> OK, a move, exit

# 2. The arena (from external-baseline/)
python3 arena_web.py                                    # http://localhost:5001

# 3. Rapfi as a Piskvork engine (must run from its own dir for weight discovery)
cd rapfi-engine && printf "START 15\nBEGIN\nEND\n" | ./pbrain-rapfi

# In the arena UI, an engine command for Rapfi is:  cd rapfi-engine && ./pbrain-rapfi

# Lint / type check after any change
ruff check && pyright
```

## `pbrain.py` — design

- **Three top constants:** `CHECKPOINT_PATH`, `MODEL_DIR` (dir with `model.py` /
  `gomoku.py`, e.g. `../main`), `MCTS_BUDGET` (0 = raw policy argmax; >0 = MCTS
  simulations per move). Defaults: `../main/release/final_policy.pt`, `../main`, 400.
- **Path resolution is relative to `__file__`, not the cwd** — `pbrain.py` runs as
  a subprocess with an arbitrary working directory, so all paths are made absolute
  at import time. `MODEL_DIR` and `../mcts` are both put on `sys.path` (`main/` and
  `mcts/` share identical `gomoku.py`/`model.py` via symlinks, so order is safe).
- **MCTS is optional.** `mcts_search_batched` (from `mcts/mcts.py`) needs the
  compiled `mcts_ext.so`; if the import fails we warn to stderr and fall back to
  raw policy. Search uses the deployment settings (`entropy_multiplier=None`,
  `dirichlet_epsilon=0` → deterministic), mirroring `mcts/play_web.py`. Move =
  `argmax` of the root visit distribution. Raw-policy path reuses
  `gomoku.select_action_batch_eval(..., temperature=0, deterministic=True)`.
- **Reuses existing code, no game logic reinvented:** model load follows
  `mcts/play_web.py`; board/obs via `gomoku.encode_observation` /
  `board_from_observation`.
- **Coordinate / perspective conventions** (shared with the arena):
  Piskvork `x,y` = `col,row`, 0-indexed; flat idx = `y*15 + x`. The model is
  perspective-based, so `pbrain` keeps two planes — `mine` / `theirs` — and builds
  the canonical obs from them directly (channel 0 = side-to-move).

## `arena_web.py` — design

- **Engine-agnostic & does its own win/draw detection** — it only speaks the text
  protocol and never trusts an engine's claims (`check_win` is a standalone
  5-in-a-row scan).
- **Subprocess I/O — the important gotcha.** Each engine has a dedicated **reader
  thread** doing a blocking `for line in proc.stdout:` loop that pushes lines onto
  a `queue.Queue`; consumers `get(timeout=…)`. Do **NOT** mix `select.select()`
  with buffered text-mode `readline()`: `TextIOWrapper` reads ahead from the OS
  pipe into its own buffer, so `select()` reports "nothing readable" while complete
  lines (e.g. the engine's move) sit in Python's buffer → spurious timeouts. This
  cost a long debugging session (`RAPFI_MOVE_EMISSION_ISSUE.md`); keep the
  reader-thread pattern.
- **Move relaying:** an engine's **first** move is seeded with `BEGIN` (empty
  board) or a full `BOARD … DONE` dump from that engine's perspective (own stones
  tagged `1`, opponent `2`) — this also seeds custom openings. Every **subsequent**
  move is relayed with `TURN x,y` (the opponent's last move).
- **Game loop** runs on a background thread driven by `threading.Event`s
  (`_running` / `_stop` / `_step`): continuous play, single-step while paused, and
  stop. Engine I/O happens **without** holding the state lock so `/api/state` stays
  responsive. The frontend polls `/api/state` (~500 ms).
- **Setup mode:** click the board while idle to place/remove stones for a custom
  opening (auto-alternating B/W/B…). Side-to-move at game start is derived from the
  stone counts (black-first parity), not assumed.
- **Controls / routes:** `/api/config` (engine commands, swap colors, per-move time
  limit → sent as `INFO timeout_turn <ms>`), `/api/setup_stone`, `/api/clear`,
  `/api/start`, `/api/pause`, `/api/resume`, `/api/step`, `/api/stop`,
  `/api/state`. Handshake/move timeouts are generous because `pbrain` loads its
  model lazily before answering `START` and the first search includes CUDA warm-up.

## Rapfi build

Source: `github.com/dhbloo/rapfi` (GPL-3.0), weights from its `Networks`
submodule (CC0). No clang here, so built with gcc/AVX2 (presets are clang-only):

```bash
cd rapfi/Rapfi
cmake -B build-avx2 -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ \
      -DCMAKE_BUILD_TYPE=Release -DUSE_SSE=ON -DUSE_AVX2=ON \
      -DUSE_AVX512=OFF -DUSE_BMI2=ON -DUSE_VNNI=OFF
cmake --build build-avx2 -j
```

The binary is `pbrain-rapfi`. It is staged in `rapfi-engine/` next to `config.toml`
+ `model210901.bin` + the `mix9svq*.bin.lz4` weights referenced by that config, so
it must be launched from `rapfi-engine/` (hence `cd rapfi-engine && ./pbrain-rapfi`).
