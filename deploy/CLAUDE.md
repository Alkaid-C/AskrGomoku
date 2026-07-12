# deploy/ — ONNX Export & Browser Web App

The frontend (HTML/CSS/JS) is fully AI-generated with no human review. The user has no frontend expertise. You must be able to independently write, debug, and verify all frontend changes — keep code straightforward enough to reason about correctly.

## File Responsibilities

- `export_onnx.py` — PyTorch checkpoint → ONNX conversion with verification. See `ONNX_DEPLOYMENT_GUIDE.md` for the exported model's I/O contract.
- `web_app/index.html` — App HTML (external CSS/JS references).
- `web_app/styles.css` — Responsive styling.
- `web_app/js/i18n.js` — Chinese/English localization. `t(key)` lookup, `tFormat(key, params)` for `{placeholder}` substitution, language stored in `localStorage.gomoku-lang`.
- `web_app/js/gomoku-board.js` — Board state, move execution, win detection, legal move mask.
- `web_app/js/mcts.js` — `MCTSSearch`: JS port of the training MCTS (advanced difficulty).
- `web_app/js/ep-probe.js` — Execution-provider probe + ORT env setup + localStorage cache.
- `web_app/js/ep-probe-worker.js` — Dedicated worker running the WebGPU probe so it can be hard-killed (`worker.terminate()`).
- `web_app/js/model-manager.js` — Difficulty → model mapping; junior/intermediate session loading; advanced download + probe orchestration (caches the advanced player across games).
- `web_app/js/onnx-ai-player.js` — ONNX inference: `evaluate()` primitive, policy sampling (`getMove`), MCTS entry (`getMoveWithMCTS`).
- `web_app/js/game-controller.js` — Main orchestrator: UI drawing, game flow, event handling, animations, the advanced-difficulty time-estimate acknowledgment dialog.
- `web_app/fetch_vendor.sh` — Rebuilds `vendor/` from the npm registry (pinned onnxruntime-web version).
- `web_app/vendor/` — Self-hosted onnxruntime-web runtime (git-ignored; ~24 MB). Only the files the browser actually fetches: `ort.webgpu.min.js` + the asyncify wasm build (`.mjs` glue + `.wasm`) — the ort.webgpu bundle uses the asyncify build for both the wasm and webgpu EPs.
- `web_app/models/*.onnx` — Exported models (git-ignored; regenerate with `export_onnx.py` from their checkpoints).
- `web_app/bench.html` — Dev-only latency benchmark page (per-EP/thread presets via URL params, results in localStorage). Not linked from the app.
- `web_app/test.html` — Dev-only MCTS test page: tactical/invariant assertions + cross-check visit-distribution dumps, runnable per EP. Not linked from the app.

There is no build step and no single-file bundle. A former standalone build (`build.py` → `gomoku-standalone.html`) was removed: multithreaded WASM needs COOP/COEP response headers plus same-origin runtime files, which a copy-paste single file cannot provide.

## Difficulties

| Difficulty | Model | Play |
|---|---|---|
| junior | `models/dial.onnx` | policy sampling, temperature 1.0 |
| intermediate | `models/cello.onnx` | policy sampling, temperature 0.7 |
| advanced | `models/mcts_test.onnx` (interim; re-export from `mcts/release/stage2/final_policy.pt` when training finishes) | MCTS, deterministic argmax-visits |

Junior/intermediate load a plain WASM session per game. Advanced goes through the EP probe flow below.

## MCTS (advanced difficulty, `js/mcts.js`)

JS port of the stage-2 training search (`mcts/mcts.py` + `mcts/mcts_ext.cpp`), inference-only. MCTS (not negamax) is the correct inference operator for stage-2 checkpoints: the value head is trained against `root_Q`, a visit-weighted soft average, and stage 2 repeatedly distills raw → MCTS(raw), so search at inference is one more policy-improvement step.

Constants (`MCTS_C_PUCT`, `MCTS_DISCOUNT_GAMMA`, `MCTS_FPU_MULTIPLIER`) mirror `mcts/main.py` — **keep them in sync**. Simulation budget: `MCTS_SIMS` (never shown to the user; they only see estimated seconds per move = sims × measured mean latency).

Port-fidelity points (silently wrong if missed; verified against Python — see Verification):

- Root gets `visitCount = 1` virtual visit **before** expansion.
- Untried children start at `Q = nodeValue * FPU_MULTIPLIER`, not 0.
- `backup` applies `v = -v * gamma` at every level **including the leaf**, before accumulation.
- Terminal value is stored from the **parent's** perspective (+1 win / 0 draw) and backed up as `-terminalValue`; terminal leaves re-backup their cached value on later visits without board replay.
- Priors: mask illegal squares, softmax over 225, stored over legal actions only. No Dirichlet noise, no entropy rescaling, no D4 cache, no tree reuse between moves.
- Move = argmax child visits; `rootQ` = visit-weighted mean child Q.

Search runs on the main thread: each `session.run` await yields to the event loop, and tree operations are microseconds.

## EP Probe (`js/ep-probe.js`)

The fastest execution provider varies wildly by device (desktop Chrome: WASM-4T ≈ WebGPU; desktop Firefox: WebGPU 10× worse; iOS: WebGPU 2.4× better), so it is measured, not hardcoded:

- `epProbeConfigureOrtEnv()` runs once at app init, **before any session creation**: absolute `wasmPaths` (bare relative URLs break Firefox) and `numThreads = crossOriginIsolated ? min(4, cores) : 1` (thread count is fixed at first wasm init).
- The model is fetched once into a buffer (download progress UI); all probe sessions are created from it.
- **Both EPs are probed inside dedicated workers, hard-killed via `worker.terminate()` — never on the main thread.** Two field-observed reasons: (1) `session.run` is not abortable, and the ort.webgpu bundle's wasm and webgpu EPs share one asyncify wasm instance (no reentrancy) — an orphaned timed-out run poisons the main runtime, making every later inference ~6× slower (25 ms probe → 150 ms in-game); (2) main-thread timings are inflated several-fold by the loading screen's rAF animations (forced layout per frame) — observed 120 ms medians for a runtime whose quiet steady state is ~20 ms. Browsers whose workers lack the EP fall back to a main-thread probe (risks accepted only there).
- Probe order: WASM, then WebGPU unless WASM ≤ 20 ms. Warmup is a fixed-duration spin (≥1 s AND ≥5 runs — covers wasm JIT tier-up, WebGPU shader compile, CPU frequency ramp; a single-run warmup polluted the timed samples), then ~1.5 s of timed runs. Two statistics come out of it: the **median** picks the winning EP (robust — a stray outlier can't flip the comparison on ≤20 samples), the **mean** feeds the per-move time estimate (the total of a many-inference move is n × mean, and long-tail events recur during real moves; old caches without `meanMs` fall back to the median).
- Watchdog: an inactivity timer re-armed by per-run worker heartbeats (15 s — distinguishes "stuck" from "legitimately slow", e.g. a >10 s first-run shader compile) plus a 30 s absolute per-EP cap.
- Probe sessions die with their workers; the winner's game session is created fresh on the main thread (browser wasm-code/shader caches are shared across workers, so this is far cheaper than the cold path the probe just paid).
- Result + a `confirmed` flag (the user's one-time acknowledgment of the time estimate) cached in `localStorage['gomoku-ep-config']`; invalidated when the ORT version, isolation, or GPU availability changes — which also re-requires acknowledgment, since the estimate changed. Cached visits skip the probe (a session is still created and warmed) and, once confirmed, skip the dialog too: advanced games start with no extra click.

## Board Perspective Flip (`gomoku-board.js`)

The board internally stores `blackPieces` and `whitePieces` as separate 15x15 arrays. `GetBoardState()` returns `[c0, c1]` where c0 = current player's pieces and c1 = opponent's pieces, swapping based on `whoToPlay`. This matches the model's expected input convention — the model always sees the board from the side-to-move's perspective. All AI inference goes through `evaluate()` → `GetBoardState()`, so the flip is handled automatically; the value output is always from the current player's perspective.

## Game Flow (`game-controller.js`)

Setup panel (color + difficulty) → loading screen → game panel. For junior/intermediate the loading screen is the model load (min 3 s on first load). For advanced it has phases: download (progress %) → probe (usually ~10 s, skipped when cached) → time-estimate acknowledgment ("预计每步需要约 x 秒", 我知道了 / 重新测试 / 选择其他难度). 我知道了 persists `confirmed` — later advanced games skip the dialog entirely; 重新测试 forces a fresh probe (recovers from a polluted first measurement); 选择其他难度 returns to setup and keeps the probe result but **not** the acknowledgment, so the estimate is shown again next time. Player clicks board → pending move → confirm/cancel. AI move: 50 ms UI yield → inference/search → draw. Undo pops 2 moves and restores from history snapshots. Game end: record mode with move numbers and winning line.

## Deployment

`https://gomoku.alkaid-c.cc` → Caddy `file_server` of `deploy/web_app/` (block in `/etc/caddy/Caddyfile`) with `Cross-Origin-Opener-Policy: same-origin` + `Cross-Origin-Embedder-Policy: require-corp` — this makes the page `crossOriginIsolated`, enabling multithreaded WASM. **Edits to web_app/ are live on the public site immediately.** Caddy gotcha: standalone `caddy validate` fails on the Cloudflare token (env only injected via systemd drop-in) — just `systemctl reload caddy`.

ORT is served same-origin from `vendor/` instead of a CDN: under COEP every cross-origin resource needs cooperating CORS/CORP headers (fragile, third-party controlled), and CDN reachability is unreliable in mainland China. `vendor/` and `models/*.onnx` are git-ignored (binaries); rebuild vendor with `./fetch_vendor.sh`, regenerate models with `export_onnx.py`.

## Verification

**This machine is headless: no browser, and `onnxruntime-node` is not installed (and must not be installed). Do not run tests that need a Node ORT runtime or a browser — anything browser-based (`test.html`, `bench.html`, the app itself) is run by the user on their own devices.** What Claude can verify locally: Python-side checks (`export_onnx.py`'s built-in torch-vs-ONNX comparison, `onnxruntime` via pip), `node --check` syntax checks on JS, and static reasoning.

- `web_app/test.html` (served, e.g. `python3 -m http.server`; run by the user in a browser) runs tactical assertions (win-in-1 → 100% visit share, block-win-in-1, determinism, visit-count invariants) against the real model per EP.
- Port fidelity was cross-checked against `mcts/mcts.py` (`mcts_search_batched`, ε=0, canonicalization patched to identity, same ONNX weights): visit distributions match exactly. With canonicalization left on, Python evaluates rotated boards, so distributions differ by float noise — top moves still match.
