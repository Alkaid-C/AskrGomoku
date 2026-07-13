# deploy/ — ONNX Export & Browser Web App

The frontend (HTML/CSS/JS) is fully AI-generated with no human review. The user has no frontend expertise. You must be able to independently write, debug, and verify all frontend changes — keep code straightforward enough to reason about correctly.

## File Responsibilities

- `export_onnx.py` — PyTorch checkpoint → ONNX conversion with verification. See `ONNX_DEPLOYMENT_GUIDE.md` for the exported model's I/O contract.
- `web_app/index.html` — App HTML (external CSS/JS references).
- `web_app/styles.css` — Responsive styling.
- `web_app/js/i18n.js` — Chinese/English localization. `t(key)` lookup, `tFormat(key, params)` for `{placeholder}` substitution, language stored in `localStorage.gomoku-lang`.
- `web_app/js/gomoku-board.js` — Board state, move execution, win detection, legal move mask.
- `web_app/js/mcts.js` — `MCTSSearch`: JS port of the training MCTS (melody difficulty).
- `web_app/js/ep-probe.js` — Execution-provider probe + ORT env setup + localStorage cache.
- `web_app/js/ep-probe-worker.js` — Dedicated worker running the WebGPU probe so it can be hard-killed (`worker.terminate()`).
- `web_app/js/model-manager.js` — Difficulty → model mapping; dial/cello/curtain session loading; melody download (`fetchMelodyModel`, bytes kept in memory so a probe re-run never re-downloads) and probe orchestration (`probeMelody`; caches the melody player across games).
- `web_app/js/orbit-loader.js` — Reusable orbital loading animation: `createOrbitLoader(container, {size})` → `{start, stop, destroy}`. Fixed 200×200 internal coordinates scaled to any size; per-instance unique path ids, so multiple instances can coexist (loading screen now; in-game progress later).
- `web_app/js/onnx-ai-player.js` — ONNX inference: `evaluate()` primitive, policy sampling (`getMove`), MCTS entry (`getMoveWithMCTS`).
- `web_app/js/game-controller.js` — Main orchestrator: UI drawing, game flow, event handling, animations, the melody-difficulty time-estimate acknowledgment dialog.
- `web_app/fetch_vendor.sh` — Rebuilds `vendor/` from the npm registry (pinned onnxruntime-web version).
- `web_app/vendor/` — Self-hosted onnxruntime-web runtime (git-ignored; ~24 MB). Only the files the browser actually fetches: `ort.webgpu.min.js` + the asyncify wasm build (`.mjs` glue + `.wasm`) — the ort.webgpu bundle uses the asyncify build for both the wasm and webgpu EPs.
- `web_app/models/*.onnx` — Exported models (git-ignored; regenerate with `export_onnx.py` from their checkpoints).
- `web_app/bench.html` — Dev-only latency benchmark page (per-EP/thread presets via URL params, results in localStorage). Not linked from the app.
- `web_app/test.html` — Dev-only MCTS test page: tactical/invariant assertions + cross-check visit-distribution dumps, runnable per EP. Not linked from the app.

There is no build step and no single-file bundle. A former standalone build (`build.py` → `gomoku-standalone.html`) was removed: multithreaded WASM needs COOP/COEP response headers plus same-origin runtime files, which a copy-paste single file cannot provide.

## Difficulties

Difficulty keys, UI names, and model filenames all coincide. RL-trained models (dial, cello) have naturally sharp policies, so they sample at temperature 1.0; MCTS-trained policies (curtain, melody — the policy head is distilled from visit distributions) are naturally flat, so both play at temperature 0.5.

| Difficulty | UI level | UI description | Model | Play |
|---|---|---|---|---|
| dial | 初级 / Easy | Classic Model | `models/dial.onnx` | policy sampling, temperature 1.0 |
| cello | 中级 / Medium | Advanced Model | `models/cello.onnx` | policy sampling, temperature 1.0 |
| curtain | 高级 / Hard | Post-Trained | `models/curtain.onnx` (interim; re-export from `mcts/release/stage2/final_policy.pt` when training finishes) | policy sampling, temperature 0.5 |
| melody | 大师 / Master | Deep Think | `models/curtain.onnx` (same file as curtain) | MCTS, visit counts sampled at temperature 0.5 |

Dial/cello/curtain load a plain WASM session per game. Melody goes through the EP probe flow below.

## MCTS (melody difficulty, `js/mcts.js`)

JS port of the stage-2 training search (`mcts/mcts.py` + `mcts/mcts_ext.cpp`), inference-only. MCTS (not negamax) is the correct inference operator for stage-2 checkpoints: the value head is trained against `root_Q`, a visit-weighted soft average, and stage 2 repeatedly distills raw → MCTS(raw), so search at inference is one more policy-improvement step.

Constants (`MCTS_C_PUCT`, `MCTS_DISCOUNT_GAMMA`, `MCTS_FPU_MULTIPLIER`, `MCTS_ACTION_TEMPERATURE` = `STAGE2_ACTION_TEMPERATURE`) mirror `mcts/main.py` — **keep them in sync**. Simulation budget: `MCTS_SIMS` (never shown to the user; they only see estimated seconds per move = sims × measured mean latency).

Port-fidelity points (silently wrong if missed; verified against Python — see Verification):

- Root gets `visitCount = 1` virtual visit **before** expansion.
- Untried children start at `Q = nodeValue * FPU_MULTIPLIER`, not 0.
- `backup` applies `v = -v * gamma` at every level **including the leaf**, before accumulation.
- Terminal value is stored from the **parent's** perspective (+1 win / 0 draw) and backed up as `-terminalValue`; terminal leaves re-backup their cached value on later visits without board replay.
- Priors: mask illegal squares, softmax over 225, stored over legal actions only. No Dirichlet noise, no entropy rescaling, no D4 cache, no tree reuse between moves.
- Move sampled from `visits ** (1/MCTS_ACTION_TEMPERATURE)`, renormalized — same as stage-2 self-play (`self_play.py`); `rootQ` = visit-weighted mean child Q.

Search runs on the main thread: each `session.run` await yields to the event loop, and tree operations are microseconds.

## EP Probe (`js/ep-probe.js`)

The fastest execution provider varies wildly by device (desktop Chrome: WASM-4T ≈ WebGPU; desktop Firefox: WebGPU 10× worse; iOS: WebGPU 2.4× better), so it is measured, not hardcoded:

- `epProbeConfigureOrtEnv()` runs once at app init, **before any session creation**: absolute `wasmPaths` (bare relative URLs break Firefox) and `numThreads = crossOriginIsolated ? min(4, cores) : 1` (thread count is fixed at first wasm init).
- The model is fetched once into a buffer (download progress UI); all probe sessions are created from it.
- **Both EPs are probed inside dedicated workers, hard-killed via `worker.terminate()` — never on the main thread.** Two field-observed reasons: (1) `session.run` is not abortable, and the ort.webgpu bundle's wasm and webgpu EPs share one asyncify wasm instance (no reentrancy) — an orphaned timed-out run poisons the main runtime, making every later inference ~6× slower (25 ms probe → 150 ms in-game); (2) main-thread timings are inflated several-fold by the loading screen's rAF animations (forced layout per frame) — observed 120 ms medians for a runtime whose quiet steady state is ~20 ms. There is deliberately no main-thread fallback: a browser whose workers lack the EP simply fails that EP's probe (WebGPU missing in workers → the WASM result stands; both EPs failing → the probe-failure dialog).
- Probe order: WASM, then WebGPU unless the WASM mean is already ≤ 10 ms. Warmup stops after either 5 runs or 5 s, then exactly 20 runs are timed. The 30 s per-EP worker cap rejects devices that cannot complete a useful sample in an acceptable time. The **mean** both picks the winning EP and feeds the per-move estimate (the total of a many-inference move is n × mean).
- Watchdog: an inactivity timer re-armed by per-run worker heartbeats (15 s — distinguishes "stuck" from "legitimately slow", e.g. a >10 s first-run shader compile) plus a 30 s absolute per-EP cap.
- Probe sessions die with their workers; the winner's game session is created fresh on the main thread (browser wasm-code/shader caches are shared across workers, so this is far cheaper than the cold path the probe just paid).
- Result + a `confirmed` flag (the user's one-time acknowledgment of the time estimate) cached in `localStorage['gomoku-ep-config']`; invalidated when the ORT version, isolation, or GPU availability changes — which also re-requires acknowledgment, since the estimate changed. Cached visits skip the probe (a game session is still created) and, once confirmed, skip the dialog too: melody games start with no extra click.

## Board Perspective Flip (`gomoku-board.js`)

The board internally stores `blackPieces` and `whitePieces` as separate 15x15 arrays. `GetBoardState()` returns `[c0, c1]` where c0 = current player's pieces and c1 = opponent's pieces, swapping based on `whoToPlay`. This matches the model's expected input convention — the model always sees the board from the side-to-move's perspective. All AI inference goes through `evaluate()` → `GetBoardState()`, so the flip is handled automatically; the value output is always from the current player's perspective.

## Game Flow (`game-controller.js`)

Setup panel (color + difficulty) → loading screen → game panel. For dial/cello/curtain the loading screen is the model load (min 3 s on first load). For melody the loading screen has two visually distinct phases, then a dialog:

- **Download** (progress %): the standard loading look — 200 px orbit + rotating poem.
- **Probe** (skipped when cached): its own look — 160 px orbit, no poem, a minimal progress bar with one segment per probe phase (WASM setup/warmup, WASM timing, and with a GPU: WebGPU setup/warmup, WebGPU timing) and a per-phase status line ("准备 CPU 推理…" etc.). Within a segment the fill advances linearly on `EP_PROBE_TOTAL_CAP_MS` (the worst-case budget after which the probe worker is killed anyway), so the bar always moves but can never overshoot a still-running phase.
- **Acknowledgment dialog**: explains that Deep Think takes long per move (written assuming the reader saw none of the probe screens; on fast/cached paths the probe phase may not have been visible) and highlights the one-decimal Melody estimate. Buttons: 开始对局 (persists `confirmed` — later melody games skip the dialog entirely) / 改用 Curtain（高级，附单步 WASM 估时；低于 0.1 秒显示 `<0.1`） (starts a **curtain** game directly with the chosen color, syncs the setup panel's selection; keeps the probe result but **not** the acknowledgment) / 重新测试 as a small text link (re-probes without re-downloading; recovers from a polluted first measurement).

Failures get in-page dialogs on the loading screen (no native `alert` for melody): download failure → 返回设置; probe failure → 改用高级难度 / 返回设置. The loading status line and dialog text are JS-managed (no `data-i18n` — it used to clobber them on language switch); their key+params are re-rendered via the `gomoku-langchange` event that `setLang` dispatches.

Player clicks board → pending move → confirm/cancel. AI move: 50 ms UI yield → inference/search → draw. Undo pops 2 moves and restores from history snapshots. Game end: record mode with move numbers and winning line.

## Deployment

`https://gomoku.alkaid-c.cc` → Caddy `file_server` of `deploy/web_app/` (block in `/etc/caddy/Caddyfile`) with `Cross-Origin-Opener-Policy: same-origin` + `Cross-Origin-Embedder-Policy: require-corp` — this makes the page `crossOriginIsolated`, enabling multithreaded WASM. **Edits to web_app/ are live on the public site immediately.** Caddy gotcha: standalone `caddy validate` fails on the Cloudflare token (env only injected via systemd drop-in) — just `systemctl reload caddy`.

ORT is served same-origin from `vendor/` instead of a CDN: under COEP every cross-origin resource needs cooperating CORS/CORP headers (fragile, third-party controlled), and CDN reachability is unreliable in mainland China. `vendor/` and `models/*.onnx` are git-ignored (binaries); rebuild vendor with `./fetch_vendor.sh`, regenerate models with `export_onnx.py`.

## Verification

**This machine is headless: no browser, and `onnxruntime-node` is not installed (and must not be installed). Do not run tests that need a Node ORT runtime or a browser — anything browser-based (`test.html`, `bench.html`, the app itself) is run by the user on their own devices.** What Claude can verify locally: Python-side checks (`export_onnx.py`'s built-in torch-vs-ONNX comparison, `onnxruntime` via pip), `node --check` syntax checks on JS, and static reasoning.

- `web_app/test.html` (served, e.g. `python3 -m http.server`; run by the user in a browser) runs tactical assertions (win-in-1 visit share + argmax-visits, block-win-in-1 argmax-visits, search determinism, visit-count invariants — the played move itself is sampled, so assertions target search statistics) against the real model per EP.
- Port fidelity was cross-checked against `mcts/mcts.py` (`mcts_search_batched`, ε=0, canonicalization patched to identity, same ONNX weights): visit distributions match exactly. With canonicalization left on, Python evaluates rotated boards, so distributions differ by float noise — top moves still match.
