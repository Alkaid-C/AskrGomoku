#!/usr/bin/env python3
"""
Gomoku Arena - watch two Piskvork-protocol engines play, step by step.

A Flask web server (port 5001) that launches two Gomocup/Piskvork engines as
subprocesses, relays moves between them, and visualizes the game in the browser.

Design notes (engine-agnostic operation, own win/draw detection, protocol
relaying) are in external-baseline/CLAUDE.md, "`arena_web.py` — design".

All HTML/CSS/JS is embedded for portability (same pattern as mcts/play_web.py).
"""

import contextlib
import os
import queue
import re
import subprocess
import sys
import threading
import time

from flask import Flask, jsonify, request

app = Flask(__name__)

BOARD_SIZE = 15
_HERE = os.path.dirname(os.path.abspath(__file__))

# A move line is exactly "x,y" (possibly with surrounding spaces). Anything else
# an engine prints (MESSAGE/DEBUG/ERROR/UNKNOWN/SUGGEST/OK/...) is informational.
_MOVE_RE = re.compile(r"^\s*(\d+)\s*,\s*(\d+)\s*$")

# Deliberately generous: model load + CUDA warm-up happen inside START and the
# first search.
HANDSHAKE_TIMEOUT = 60.0
MOVE_TIMEOUT_MARGIN = 180.0  # seconds added on top of the configured per-move limit


# ============================================================================
# Engine subprocess wrapper
# ============================================================================


class EngineError(Exception):
    pass


_EOF = object()  # sentinel pushed onto the line queue when stdout closes


class Engine:
    def __init__(self, name: str, cmd: str, cwd: str):
        self.name = name
        self.cmd = cmd
        self.cwd = cwd
        self.proc: subprocess.Popen[str] | None = None
        self.first_move = True
        # A dedicated reader thread feeds decoded stdout lines here. Do NOT mix
        # select() with buffered text-mode readline() — see CLAUDE.md,
        # "Subprocess I/O".
        self._lines: queue.Queue[object] = queue.Queue()

    def start(self) -> None:
        self.proc = subprocess.Popen(
            self.cmd,
            shell=True,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            cwd=self.cwd,
        )
        threading.Thread(target=self._read_stdout, daemon=True).start()
        # Drain stderr in the background so a chatty engine can't deadlock on a
        # full pipe; surface lines on our own stderr for debugging.
        threading.Thread(target=self._drain_stderr, daemon=True).start()

    def _read_stdout(self) -> None:
        proc = self.proc
        if proc is None or proc.stdout is None:
            return
        for line in proc.stdout:  # blocking readline loop; no select()
            self._lines.put(line.rstrip("\n"))
        self._lines.put(_EOF)

    def _drain_stderr(self) -> None:
        proc = self.proc
        if proc is None or proc.stderr is None:
            return
        for line in proc.stderr:
            print(f"[{self.name} stderr] {line.rstrip()}", file=sys.stderr, flush=True)

    def send(self, line: str) -> None:
        if self.proc is None or self.proc.stdin is None:
            raise EngineError("engine not started")
        try:
            self.proc.stdin.write(line + "\n")
            self.proc.stdin.flush()
        except (BrokenPipeError, OSError) as e:
            raise EngineError(f"failed to write to engine: {e}") from e

    def _next_line(self, timeout: float) -> "str | None":
        """Next stdout line, or None on timeout. Raises on EOF (engine exit)."""
        try:
            item = self._lines.get(timeout=max(timeout, 0.0))
        except queue.Empty:
            return None
        if item is _EOF:
            raise EngineError("engine closed its output (crashed?)")
        return item  # type: ignore[return-value]

    def read_move(self, timeout: float) -> "tuple[int, int]":
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise EngineError("timed out waiting for a move")
            line = self._next_line(remaining)
            if line is None:
                raise EngineError("timed out waiting for a move")
            line = line.strip()
            if not line:
                continue
            m = _MOVE_RE.match(line)
            if m:
                return int(m.group(1)), int(m.group(2))
            # otherwise: informational line, ignore and keep reading

    def wait_ok(self, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise EngineError("timed out waiting for OK")
            line = self._next_line(remaining)
            if line is None:
                raise EngineError("timed out waiting for OK")
            line = line.strip()
            if not line:
                continue
            if line.upper() == "OK":
                return
            if line.upper().startswith("ERROR"):
                raise EngineError(f"engine refused START: {line}")
            # otherwise: informational line, ignore

    def handshake(self, time_limit_ms: int) -> None:
        self.send(f"START {BOARD_SIZE}")
        self.wait_ok(HANDSHAKE_TIMEOUT)
        self.send("INFO timeout_match 0")
        self.send(f"INFO timeout_turn {time_limit_ms}")

    def stop(self) -> None:
        if self.proc is None:
            return
        with contextlib.suppress(Exception):
            self.send("END")
        try:
            self.proc.terminate()
            self.proc.wait(timeout=2)
        except Exception:
            with contextlib.suppress(Exception):
                self.proc.kill()
        self.proc = None


# ============================================================================
# Win / draw detection (standalone - the arena does not trust engines)
# ============================================================================


def check_win(board: "list[list[int]]", x: int, y: int, color_val: int) -> bool:
    """True if the stone just played at (x=col, y=row) completes 5-in-a-row."""
    for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
        count = 1
        r, c = y + dr, x + dc
        while 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and board[r][c] == color_val:
            count += 1
            r += dr
            c += dc
        r, c = y - dr, x - dc
        while 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and board[r][c] == color_val:
            count += 1
            r -= dr
            c -= dc
        if count >= 5:
            return True
    return False


def board_full(board: "list[list[int]]") -> bool:
    return all(v != 0 for row in board for v in row)


# ============================================================================
# Shared arena state
# ============================================================================

_lock = threading.Lock()
_running = threading.Event()  # set => play continuously
_stop = threading.Event()
_step = threading.Event()  # set => make exactly one move while paused
_game_thread: "threading.Thread | None" = None

engines: "dict[str, Engine]" = {}


def _empty_board() -> "list[list[int]]":
    return [[0] * BOARD_SIZE for _ in range(BOARD_SIZE)]


# board values: 0 empty, 1 black, 2 white. black always moves first.
GAME: dict = {
    "status": "idle",  # idle | setup | starting | running | paused | finished
    "board": _empty_board(),
    "history": [],  # list of {x,y,color,engine,t}
    "last_move": None,  # [x,y]
    "to_move": "black",
    "result": None,  # human-readable result string, or None while playing
    "cmd_a": "python3 pbrain.py",
    "cmd_b": "cd rapfi-engine && ./pbrain-rapfi",
    "black_engine": "A",  # which engine ('A'/'B') plays black
    "time_limit_ms": 5000,
}


def _other(key: str) -> str:
    return "B" if key == "A" else "A"


def _other_color(color: str) -> str:
    return "white" if color == "black" else "black"


def _mover_key(to_move: str) -> str:
    return GAME["black_engine"] if to_move == "black" else _other(GAME["black_engine"])


# ============================================================================
# Game loop (runs on a background thread)
# ============================================================================


def _send_board(engine: Engine, mover_color: str) -> None:
    """Dump the full position to `engine` (1 = its own stones, 2 = opponent)."""
    engine.send("BOARD")
    board = GAME["board"]
    for y in range(BOARD_SIZE):
        for x in range(BOARD_SIZE):
            v = board[y][x]
            if v == 0:
                continue
            color = "black" if v == 1 else "white"
            who = 1 if color == mover_color else 2
            engine.send(f"{x},{y},{who}")
    engine.send("DONE")


def _finish_locked(winner: "str | None", reason: str) -> None:
    if winner is None:
        GAME["result"] = f"Draw ({reason})"
    else:
        key = _mover_key(winner)
        GAME["result"] = f"{winner.capitalize()} ({key}) wins - {reason}"
    GAME["status"] = "finished"


def _do_one_move() -> None:
    with _lock:
        to_move = GAME["to_move"]
        eng_key = _mover_key(to_move)
        engine = engines[eng_key]
        board_empty = not any(v != 0 for row in GAME["board"] for v in row)
        last = GAME["last_move"]
        first = engine.first_move
        timeout = GAME["time_limit_ms"] / 1000.0 + MOVE_TIMEOUT_MARGIN
        mover_color = to_move

    # Engine I/O happens WITHOUT the lock so /api/state stays responsive.
    try:
        t0 = time.monotonic()
        if first:
            if board_empty:
                engine.send("BEGIN")
            else:
                _send_board(engine, mover_color)
            engine.first_move = False
        else:
            engine.send(f"TURN {last[0]},{last[1]}")
        x, y = engine.read_move(timeout)
        elapsed = time.monotonic() - t0
    except EngineError as e:
        with _lock:
            _finish_locked(_other_color(to_move), f"{eng_key} error: {e}")
        return

    with _lock:
        board = GAME["board"]
        if not (0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE) or board[y][x] != 0:
            _finish_locked(_other_color(to_move), f"{eng_key} played illegal move {x},{y}")
            return
        color_val = 1 if to_move == "black" else 2
        board[y][x] = color_val
        GAME["last_move"] = [x, y]
        GAME["history"].append(
            {"x": x, "y": y, "color": to_move, "engine": eng_key, "t": elapsed}
        )
        if check_win(board, x, y, color_val):
            _finish_locked(to_move, "five in a row")
            return
        if board_full(board):
            _finish_locked(None, "board full")
            return
        GAME["to_move"] = _other_color(to_move)


def _game_loop() -> None:
    while not _stop.is_set():
        with _lock:
            done = GAME["result"] is not None
        if done:
            break
        # Wait for permission to move: continuous run, or a single step.
        while not _stop.is_set():
            if _running.is_set():
                break
            if _step.is_set():
                _step.clear()
                break
            time.sleep(0.05)
        if _stop.is_set():
            break
        _do_one_move()


def _shutdown_engines() -> None:
    for eng in engines.values():
        eng.stop()


def _run_match() -> None:
    try:
        with _lock:
            GAME["status"] = "starting"
            time_limit_ms = GAME["time_limit_ms"]
            cmd_a, cmd_b = GAME["cmd_a"], GAME["cmd_b"]
        engines["A"] = Engine("A", cmd_a, _HERE)
        engines["B"] = Engine("B", cmd_b, _HERE)
        for eng in engines.values():
            eng.start()
        for eng in engines.values():
            eng.handshake(time_limit_ms)
        with _lock:
            GAME["status"] = "running"
        _running.set()
        _game_loop()
    except EngineError as e:
        with _lock:
            GAME["result"] = f"Error: {e}"
            GAME["status"] = "finished"
    finally:
        _shutdown_engines()
        with _lock:
            if GAME["result"] is None:
                GAME["result"] = "Stopped"
            GAME["status"] = "finished"


# ============================================================================
# Flask routes
# ============================================================================


@app.route("/")
def index():
    return HTML_TEMPLATE


@app.route("/api/state")
def api_state():
    with _lock:
        return jsonify(
            {
                "status": GAME["status"],
                "board": GAME["board"],
                "history": GAME["history"],
                "last_move": GAME["last_move"],
                "to_move": GAME["to_move"],
                "result": GAME["result"],
                "cmd_a": GAME["cmd_a"],
                "cmd_b": GAME["cmd_b"],
                "black_engine": GAME["black_engine"],
                "time_limit_ms": GAME["time_limit_ms"],
            }
        )


@app.route("/api/config", methods=["POST"])
def api_config():
    data = request.json or {}
    with _lock:
        if GAME["status"] in ("starting", "running", "paused"):
            return jsonify({"error": "cannot reconfigure while a game is active"}), 409
        if "cmd_a" in data:
            GAME["cmd_a"] = str(data["cmd_a"])
        if "cmd_b" in data:
            GAME["cmd_b"] = str(data["cmd_b"])
        if "black_engine" in data and data["black_engine"] in ("A", "B"):
            GAME["black_engine"] = data["black_engine"]
        if "time_limit_ms" in data:
            with contextlib.suppress(TypeError, ValueError):
                GAME["time_limit_ms"] = max(0, int(data["time_limit_ms"]))
    return jsonify({"ok": True})


@app.route("/api/setup_stone", methods=["POST"])
def api_setup_stone():
    data = request.json or {}
    try:
        x, y = int(data["x"]), int(data["y"])
    except (KeyError, TypeError, ValueError):
        return jsonify({"error": "x,y required"}), 400
    if not (0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE):
        return jsonify({"error": "out of bounds"}), 400
    with _lock:
        if GAME["status"] not in ("idle", "setup", "finished"):
            return jsonify({"error": "cannot edit board while a game is active"}), 409
        if GAME["status"] == "finished":
            _reset_locked()
        board = GAME["board"]
        if board[y][x] != 0:
            board[y][x] = 0  # toggle off
        else:
            # Place the color that keeps black-first alternation.
            blacks = sum(1 for row in board for v in row if v == 1)
            whites = sum(1 for row in board for v in row if v == 2)
            board[y][x] = 1 if blacks <= whites else 2
        GAME["status"] = "setup"
    return jsonify({"ok": True})


def _reset_locked() -> None:
    GAME["board"] = _empty_board()
    GAME["history"] = []
    GAME["last_move"] = None
    GAME["to_move"] = "black"
    GAME["result"] = None
    GAME["status"] = "idle"


@app.route("/api/clear", methods=["POST"])
def api_clear():
    with _lock:
        if GAME["status"] in ("starting", "running", "paused"):
            return jsonify({"error": "stop the game first"}), 409
        _reset_locked()
    return jsonify({"ok": True})


@app.route("/api/start", methods=["POST"])
def api_start():
    global _game_thread
    with _lock:
        if GAME["status"] in ("starting", "running", "paused"):
            return jsonify({"error": "game already active"}), 409
        # Reset transient game data but keep the setup board.
        GAME["history"] = []
        GAME["last_move"] = None
        GAME["result"] = None
        board = GAME["board"]
        blacks = sum(1 for row in board for v in row if v == 1)
        whites = sum(1 for row in board for v in row if v == 2)
        GAME["to_move"] = "black" if blacks <= whites else "white"
    _stop.clear()
    _step.clear()
    _running.clear()
    _game_thread = threading.Thread(target=_run_match, daemon=True)
    _game_thread.start()
    return jsonify({"ok": True})


@app.route("/api/pause", methods=["POST"])
def api_pause():
    _running.clear()
    with _lock:
        if GAME["status"] == "running":
            GAME["status"] = "paused"
    return jsonify({"ok": True})


@app.route("/api/resume", methods=["POST"])
def api_resume():
    with _lock:
        active = GAME["status"] in ("paused", "running")
        if GAME["status"] == "paused":
            GAME["status"] = "running"
    if active:
        _running.set()
    return jsonify({"ok": True})


@app.route("/api/step", methods=["POST"])
def api_step():
    # Single-step only makes sense while paused.
    _running.clear()
    with _lock:
        if GAME["status"] == "running":
            GAME["status"] = "paused"
    _step.set()
    return jsonify({"ok": True})


@app.route("/api/stop", methods=["POST"])
def api_stop():
    global _game_thread
    _stop.set()
    _running.set()  # unblock the loop so it can observe the stop flag
    _step.set()
    thread = _game_thread
    if thread is not None:
        thread.join(timeout=10)
    _game_thread = None
    return jsonify({"ok": True})


# ============================================================================
# Embedded frontend
# ============================================================================

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Gomoku Arena</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: 'Segoe UI', Tahoma, sans-serif; background: linear-gradient(135deg,#2c3e50,#4ca1af); min-height: 100vh; padding: 20px; color: #222; }
.container { max-width: 1200px; margin: 0 auto; }
h1 { text-align: center; color: #fff; margin-bottom: 18px; text-shadow: 2px 2px 4px rgba(0,0,0,.3); }
.panel { background: #fff; border-radius: 10px; padding: 16px; margin-bottom: 16px; box-shadow: 0 4px 6px rgba(0,0,0,.1); }
.row { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; margin-bottom: 8px; }
.row label { font-weight: bold; min-width: 90px; }
.row input[type=text] { flex: 1; min-width: 220px; padding: 7px; border: 2px solid #4ca1af; border-radius: 5px; font-family: monospace; }
.row input[type=number] { width: 110px; padding: 7px; border: 2px solid #4ca1af; border-radius: 5px; }
button { padding: 8px 16px; background: #4ca1af; color: #fff; border: none; border-radius: 5px; cursor: pointer; font-weight: bold; }
button:hover { background: #3b8b98; }
button:disabled { background: #bbb; cursor: not-allowed; }
button.primary { background: #27ae60; } button.primary:hover { background: #219150; }
button.danger { background: #c0392b; } button.danger:hover { background: #a93226; }
.game { display: grid; grid-template-columns: auto 320px; gap: 16px; align-items: start; }
.board { display: inline-grid; grid-template-columns: repeat(15, 36px); grid-template-rows: repeat(15, 36px); background: #deb887; border: 3px solid #8b4513; padding: 8px; }
.cell { width: 36px; height: 36px; position: relative; cursor: pointer; background: #deb887; }
.cell::before { content: ''; position: absolute; top: 50%; left: 0; right: 0; height: 1px; background: #8b4513; transform: translateY(-50%); }
.cell::after { content: ''; position: absolute; left: 50%; top: 0; bottom: 0; width: 1px; background: #8b4513; transform: translateX(-50%); }
.cell.el::before { left: 50%; } .cell.er::before { right: 50%; left: auto; width: 50%; }
.cell.et::after { top: 50%; } .cell.eb::after { bottom: 50%; top: auto; height: 50%; }
.stone { position: absolute; z-index: 1; top: 50%; left: 50%; transform: translate(-50%,-50%); width: 30px; height: 30px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 13px; font-weight: bold; font-family: 'Segoe UI', Tahoma, sans-serif; }
.stone.black { background: #000; color: #fff; }
.stone.white { background: #fff; color: #000; border: 1px solid #999; }
.cell.last .stone { box-shadow: 0 0 0 3px #e74c3c; }
.status-line { font-size: 18px; font-weight: bold; margin-bottom: 10px; }
.result { font-size: 20px; font-weight: bold; color: #c0392b; min-height: 26px; }
.sidebar h3 { color: #4ca1af; border-bottom: 2px solid #4ca1af; padding-bottom: 4px; margin-bottom: 8px; }
.timing { font-family: monospace; font-size: 13px; margin-bottom: 14px; }
.timing div { padding: 2px 4px; display: flex; justify-content: space-between; }
.timing .hdr { font-weight: bold; color: #4ca1af; border-bottom: 1px solid #ddd; }
.history { max-height: 420px; overflow-y: auto; font-family: monospace; font-size: 13px; }
.history div { padding: 2px 4px; }
.history div:nth-child(odd) { background: #f3f3f3; }
.tag { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 13px; font-weight: bold; }
.tag.black { background: #333; color: #fff; } .tag.white { background: #eee; color: #333; border: 1px solid #999; }
.controls button { margin-right: 6px; margin-bottom: 6px; }
.hint { font-size: 13px; color: #666; }
</style>
</head>
<body>
<div class="container">
  <h1>Gomoku Arena</h1>

  <div class="panel">
    <div class="row"><label>Engine A</label><input type="text" id="cmd-a" value="python3 pbrain.py"> <span id="color-a" class="tag black">Black</span></div>
    <div class="row"><label>Engine B</label><input type="text" id="cmd-b" value="cd rapfi-engine && ./pbrain-rapfi"> <span id="color-b" class="tag white">White</span></div>
    <div class="row">
      <label>Time/move</label><input type="number" id="time-limit" value="5000" step="500" min="0"> <span class="hint">ms (sent as INFO timeout_turn)</span>
      <button onclick="swapColors()">Swap colors</button>
    </div>
    <div class="row hint">Setup: while idle, click the board to place/remove stones for a custom opening (auto-alternating colors).</div>
    <div class="row controls">
      <button class="primary" id="btn-start" onclick="apiStart()">Start</button>
      <button id="btn-pause" onclick="apiPause()">Pause</button>
      <button id="btn-resume" onclick="apiResume()">Resume</button>
      <button id="btn-step" onclick="apiStep()">Step</button>
      <button class="danger" id="btn-stop" onclick="apiStop()">Stop</button>
      <button onclick="apiClear()">Clear board</button>
    </div>
  </div>

  <div class="game">
    <div class="panel">
      <div class="status-line">Status: <span id="status">idle</span> &nbsp; | &nbsp; To move: <span id="to-move">black</span></div>
      <div id="board" class="board"></div>
      <div class="result" id="result"></div>
    </div>
    <div class="panel sidebar">
      <h3>Timing</h3>
      <div class="timing" id="timing"></div>
      <h3>Move history</h3>
      <div class="history" id="history"></div>
    </div>
  </div>
</div>

<script>
let lastStatus = null;

function initBoard() {
  const board = document.getElementById('board');
  board.innerHTML = '';
  for (let y = 0; y < 15; y++) {
    for (let x = 0; x < 15; x++) {
      const cell = document.createElement('div');
      cell.className = 'cell';
      if (x === 0) cell.classList.add('el');
      if (x === 14) cell.classList.add('er');
      if (y === 0) cell.classList.add('et');
      if (y === 14) cell.classList.add('eb');
      cell.dataset.x = x; cell.dataset.y = y;
      cell.onclick = () => onCellClick(x, y);
      board.appendChild(cell);
    }
  }
}

function getCell(x, y) { return document.querySelector(`.cell[data-x="${x}"][data-y="${y}"]`); }

async function onCellClick(x, y) {
  await fetch('/api/setup_stone', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({x, y})});
  refresh();
}

function render(state) {
  document.getElementById('status').textContent = state.status;
  document.getElementById('to-move').textContent = state.to_move;
  document.getElementById('result').textContent = state.result || '';

  const blackEng = state.black_engine;
  blackEngine = blackEng;
  document.getElementById('color-a').textContent = blackEng === 'A' ? 'Black' : 'White';
  document.getElementById('color-a').className = 'tag ' + (blackEng === 'A' ? 'black' : 'white');
  document.getElementById('color-b').textContent = blackEng === 'B' ? 'Black' : 'White';
  document.getElementById('color-b').className = 'tag ' + (blackEng === 'B' ? 'black' : 'white');

  const moveNum = {};
  state.history.forEach((m, i) => { moveNum[m.y * 15 + m.x] = i + 1; });

  for (let y = 0; y < 15; y++) {
    for (let x = 0; x < 15; x++) {
      const cell = getCell(x, y);
      cell.classList.remove('last');
      const v = state.board[y][x];
      const existing = cell.querySelector('.stone');
      if (existing) existing.remove();
      if (v !== 0) {
        const s = document.createElement('div');
        s.className = 'stone ' + (v === 1 ? 'black' : 'white');
        const n = moveNum[y * 15 + x];
        if (n) s.textContent = n;
        cell.appendChild(s);
      }
    }
  }
  if (state.last_move) {
    const c = getCell(state.last_move[0], state.last_move[1]);
    if (c) c.classList.add('last');
  }

  const stats = {A: {total: 0, n: 0}, B: {total: 0, n: 0}};
  state.history.forEach(m => {
    const s = stats[m.engine];
    if (s) { s.total += (m.t || 0); s.n += 1; }
  });
  const fmt = s => `${s.total.toFixed(1)}s total / ${s.n ? (s.total / s.n).toFixed(2) : '0.00'}s avg (${s.n})`;
  document.getElementById('timing').innerHTML =
    `<div class="hdr"><span>Engine</span><span>total / avg per step</span></div>` +
    `<div><span>A (${blackEng === 'A' ? 'Black' : 'White'})</span><span>${fmt(stats.A)}</span></div>` +
    `<div><span>B (${blackEng === 'B' ? 'Black' : 'White'})</span><span>${fmt(stats.B)}</span></div>`;

  const hist = document.getElementById('history');
  hist.innerHTML = state.history.map((m, i) =>
    `<div>${i + 1}. ${m.color} (${m.engine}) @ ${m.x},${m.y} &mdash; ${(m.t || 0).toFixed(2)}s</div>`).join('');
  if (lastStatus !== state.status || state.history.length) hist.scrollTop = hist.scrollHeight;
  lastStatus = state.status;

  const active = ['starting','running','paused'].includes(state.status);
  document.getElementById('btn-start').disabled = active;
  document.getElementById('btn-pause').disabled = state.status !== 'running';
  document.getElementById('btn-resume').disabled = state.status !== 'paused';
  document.getElementById('btn-step').disabled = !(state.status === 'paused' || state.status === 'running');
  document.getElementById('btn-stop').disabled = !active;
  document.getElementById('cmd-a').disabled = active;
  document.getElementById('cmd-b').disabled = active;
  document.getElementById('time-limit').disabled = active;
}

async function refresh() {
  const r = await fetch('/api/state');
  render(await r.json());
}

async function pushConfig() {
  await fetch('/api/config', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({
    cmd_a: document.getElementById('cmd-a').value,
    cmd_b: document.getElementById('cmd-b').value,
    time_limit_ms: parseInt(document.getElementById('time-limit').value) || 0,
  })});
}

let blackEngine = 'A';
async function swapColors() {
  blackEngine = blackEngine === 'A' ? 'B' : 'A';
  await fetch('/api/config', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({black_engine: blackEngine})});
  refresh();
}

async function apiStart() { await pushConfig(); await fetch('/api/start', {method:'POST'}); refresh(); }
async function apiPause() { await fetch('/api/pause', {method:'POST'}); refresh(); }
async function apiResume() { await fetch('/api/resume', {method:'POST'}); refresh(); }
async function apiStep() { await fetch('/api/step', {method:'POST'}); refresh(); }
async function apiStop() { await fetch('/api/stop', {method:'POST'}); refresh(); }
async function apiClear() { await fetch('/api/clear', {method:'POST'}); refresh(); }

initBoard();
refresh();
setInterval(refresh, 500);
</script>
</body>
</html>
"""


if __name__ == "__main__":
    print("Starting Gomoku Arena at http://localhost:5001")
    app.run(debug=False, host="0.0.0.0", port=5001, threaded=True)
