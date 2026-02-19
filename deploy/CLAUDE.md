# deploy/ — ONNX Export & Browser Web App

The frontend (HTML/CSS/JS) is fully AI-generated with no human review. The user has no frontend expertise. You must be able to independently write, debug, and verify all frontend changes — keep code straightforward enough to reason about correctly.

## File Responsibilities

- `export_onnx.py` — PyTorch checkpoint → ONNX conversion with verification.
- `web_app/index.html` — Development HTML (external CSS/JS references).
- `web_app/styles.css` — Responsive styling.
- `web_app/js/i18n.js` — Chinese/English localization. `t(key)` lookup, language stored in `localStorage.gomoku-lang`.
- `web_app/js/gomoku-board.js` — Board state, move execution, win detection, legal move mask.
- `web_app/js/model-manager.js` — Three difficulty levels, each mapping to an ONNX model file and temperature.
- `web_app/js/onnx-ai-player.js` — ONNX inference (policy sampling + negamax search).
- `web_app/js/game-controller.js` — Main orchestrator: UI drawing, game flow, event handling, animations.
- `web_app/build.py` — Inlines CSS and JS into `gomoku-standalone.html` (single-file deployment).
- `web_app/gomoku-standalone.html` — Built output (do not edit directly).

## ONNX Export (`export_onnx.py`)

`GomokuModelForExport` wraps the base model to simplify the JS-side interface:

- **Input**: `[2, 15, 15]` float32 (no batch dimension). Channel 0 = current player, channel 1 = opponent.
- The wrapper adds the batch dimension and the constant all-ones mask channel internally: `[2, 15, 15]` → `[1, 3, 15, 15]`.
- **Output**: raw policy logits `[225]` (no softmax, no temperature) + scalar value `[1]`.
- Temperature and softmax are applied on the JS side after masking illegal moves, avoiding precision issues from softmaxing over illegal positions with large negative logits.
- Opset 21 is required for GroupNormalization support.
- After export, the ONNX file is re-saved with `save_as_external_data=False` to ensure a single file (no `.onnx.data` sidecar).

## Board Perspective Flip (`gomoku-board.js`)

The board internally stores `blackPieces` and `whitePieces` as separate 15x15 arrays. `GetBoardState()` returns `[c0, c1]` where c0 = current player's pieces and c1 = opponent's pieces, swapping based on `whoToPlay`. This matches the model's expected input convention — the model always sees the board from the side-to-move's perspective.

All AI inference paths (`getMove`, `getMoveWithNegamax`, `_getTopKActions`, `_evaluatePosition`) call `GetBoardState()`, so the flip is handled automatically. The value output is always from the current player's perspective.

## AI Inference (`onnx-ai-player.js`)

ONNX Runtime Web with WASM backend. Two inference modes:

**Policy sampling** (junior/intermediate): `getMove()` — run model once, mask illegal moves, apply temperature-scaled softmax, sample categorically.

**Negamax search** (advanced): `getMoveWithNegamax(board, depth, topK)` — at each node, get top-k legal moves by policy logit (`_getTopKActions`), try each on a cloned board:
- Terminal state → `_getTerminalValue`: +1 if current player won, -1 if lost, 0 draw.
- Leaf (depth=0) → `_evaluatePosition`: value network output (already from current player's perspective via `GetBoardState` flip).
- Otherwise → recurse and negate: `value = -childResult.value` (negamax convention).

Each inference call is independent (no batching, no caching across nodes).

## Game Flow (`game-controller.js`)

Loading screen (model load, min 3s on first load) → setup panel (color + difficulty) → game panel. Player clicks board → pending move shown semi-transparent → confirm/cancel. AI move: 50ms UI yield → inference → draw. Undo pops 2 moves (player + AI) and restores from history snapshots. Game end: board redraws in record mode with move numbers and winning line highlight.

## Build

```bash
cd deploy/web_app && python3 build.py   # → gomoku-standalone.html
```

Regex-replaces `<link href="styles.css">` and `<script src="js/...">` tags with inline content. The standalone file is self-contained except for ONNX Runtime (CDN) and model files (`models/*.onnx`).
