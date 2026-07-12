/**
 * Execution Provider Probe
 *
 * The fastest ONNX Runtime execution provider (multithreaded WASM vs WebGPU)
 * varies wildly by device and browser, so it is measured at load time rather
 * than hardcoded. The result (winning EP + median single-inference latency)
 * is cached in localStorage and drives the per-move time estimate shown to
 * the user before an advanced-difficulty game.
 */

const EP_PROBE_STORAGE_KEY = 'gomoku-ep-config';
const EP_PROBE_WARMUP_SPIN_MS = 1000;    // warmup: spin at least this long...
const EP_PROBE_WARMUP_MIN_RUNS = 5;      // ...and at least this many runs
const EP_PROBE_TIMED_BUDGET_MS = 1500;   // timing-phase budget
const EP_PROBE_MAX_RUNS = 20;            // timing-phase max runs
const EP_PROBE_INACTIVITY_MS = 15000;    // kill worker if no progress this long
const EP_PROBE_TOTAL_CAP_MS = 30000;     // absolute per-EP worker budget
const EP_PROBE_MAIN_FALLBACK_TIMEOUT_MS = 20000; // soft timeout, main-thread fallback only
const EP_PROBE_WASM_FAST_MS = 20;        // skip WebGPU if WASM is already this fast

/**
 * Detect the environment features that determine which EPs are worth probing.
 */
function epProbeEnvironment() {
    return {
        isolated: !!window.crossOriginIsolated,
        hasGpu: !!navigator.gpu,
        cores: navigator.hardwareConcurrency || 4,
    };
}

/**
 * Configure global ORT env. Must run before the FIRST session creation on
 * the page: numThreads is read once at first wasm init and cannot change
 * afterwards. wasmPaths must be an absolute URL (a bare relative path breaks
 * Firefox's module specifier resolution).
 */
function epProbeConfigureOrtEnv() {
    ort.env.wasm.wasmPaths = new URL('vendor/', location.href).href;
    const env = epProbeEnvironment();
    // Multithreaded WASM needs cross-origin isolation (COOP/COEP headers).
    // More than 4 threads never helped in benchmarks (scheduling overhead).
    ort.env.wasm.numThreads = env.isolated ? Math.min(4, env.cores) : 1;
}

/**
 * Fetch the model into memory, reporting download progress. A single buffer
 * feeds every probe session, so the model is downloaded exactly once.
 * @param {string} url - Model URL
 * @param {function} onProgress - Called with fraction in [0, 1]
 * @returns {Promise<Uint8Array>}
 */
async function epProbeFetchModel(url, onProgress) {
    const resp = await fetch(url);
    if (!resp.ok) {
        throw new Error(`Model download failed: HTTP ${resp.status} for ${url}`);
    }
    const total = parseInt(resp.headers.get('Content-Length') || '0', 10);
    if (!resp.body || !total) {
        const buf = await resp.arrayBuffer();
        if (onProgress) onProgress(1);
        return new Uint8Array(buf);
    }
    const reader = resp.body.getReader();
    const chunks = [];
    let received = 0;
    for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        chunks.push(value);
        received += value.length;
        // Content-Length is the transfer size; clamp in case of encoding.
        if (onProgress) onProgress(Math.min(1, received / total));
    }
    const out = new Uint8Array(received);
    let offset = 0;
    for (const chunk of chunks) {
        out.set(chunk, offset);
        offset += chunk.length;
    }
    return out;
}

/**
 * Random plausible board input. Latency does not depend on stone content,
 * but varying inputs rules out any caching effects during timing.
 */
function _epProbeRandomInput() {
    const data = new Float32Array(450);
    const nStones = Math.floor(Math.random() * 61);
    const used = new Set();
    for (let s = 0; s < nStones; s++) {
        let idx;
        do { idx = Math.floor(Math.random() * 225); } while (used.has(idx));
        used.add(idx);
        data[(s % 2) * 225 + idx] = 1;
    }
    return new ort.Tensor('float32', data, [2, 15, 15]);
}

function _epProbeWithTimeout(promise, ms, label) {
    return Promise.race([
        promise,
        new Promise((_, reject) => setTimeout(
            () => reject(new Error(`${label} probe timed out after ${ms} ms`)), ms)),
    ]);
}

/**
 * Create a session for one EP and warm it up (WebGPU's first runs include
 * shader compilation). Used both when probing and when restoring a cached
 * EP choice without re-timing.
 * @returns {Promise<Object>} {session}
 */
async function epProbeCreateSession(ep, modelBytes) {
    const session = await ort.InferenceSession.create(modelBytes, {
        executionProviders: [ep],
        graphOptimizationLevel: 'all',
    });
    const warmups = ep === 'webgpu' ? 3 : 1;
    for (let i = 0; i < warmups; i++) {
        await session.run({ board_state: _epProbeRandomInput() });
    }
    return { session };
}

/**
 * Probe one EP: create a session, warm it with a fixed-duration spin, then
 * time runs. The warmup spin (rather than a fixed run count) covers wasm JIT
 * tier-up, WebGPU shader compilation, and CPU frequency ramp — a single-run
 * warmup let those pollute the timed samples (observed 120 ms medians for a
 * runtime whose steady state is ~20 ms).
 * @param {string} ep - 'wasm' or 'webgpu'
 * @param {Uint8Array} modelBytes
 * @param {function} [onProgress] - Called with a phase name after the setup
 *     phase and after every run; drives the parent's inactivity watchdog
 *     when probing inside a worker.
 * @returns {Promise<Object>} {session, medianMs, meanMs} — median is robust
 *     for comparing EPs; mean is what predicts the total time of a
 *     many-inference MCTS move (sum = n × mean, and the long tail is a real
 *     recurring cost, not noise to discard).
 */
async function _epProbeOne(ep, modelBytes, onProgress) {
    const progress = onProgress || (() => {});
    const session = await ort.InferenceSession.create(modelBytes, {
        executionProviders: [ep],
        graphOptimizationLevel: 'all',
    });
    progress('setup');

    const warmupStart = performance.now();
    let warmupRuns = 0;
    while (warmupRuns < EP_PROBE_WARMUP_MIN_RUNS
           || performance.now() - warmupStart < EP_PROBE_WARMUP_SPIN_MS) {
        await session.run({ board_state: _epProbeRandomInput() });
        warmupRuns++;
        progress('warmup');
    }

    const times = [];
    const phaseStart = performance.now();
    while (times.length < EP_PROBE_MAX_RUNS
           && performance.now() - phaseStart < EP_PROBE_TIMED_BUDGET_MS) {
        const t0 = performance.now();
        await session.run({ board_state: _epProbeRandomInput() });
        times.push(performance.now() - t0);
        progress('timing');
    }
    const meanMs = times.reduce((a, b) => a + b, 0) / times.length;
    times.sort((a, b) => a - b);
    return { session, medianMs: times[Math.floor(times.length / 2)], meanMs };
}

/**
 * Probe one EP inside a dedicated worker, hard-killed via worker.terminate().
 * Probing MUST NOT run on the main thread, for two reasons observed in the
 * field: (1) session.run() is not abortable, and an abandoned run poisons
 * the shared asyncify wasm runtime (no reentrancy) — every later main-thread
 * inference then runs ~6x slower; (2) the loading screen's rAF animations
 * (forced layout every frame) inflate main-thread timings several-fold.
 *
 * Watchdog structure: an inactivity timer re-armed by per-run progress
 * heartbeats (distinguishes "stuck" from "slow" — a WebGPU first run may
 * legitimately spend >10 s compiling shaders), plus an absolute per-EP cap.
 *
 * Browsers whose workers lack the EP (or workers at all) fall back to a
 * main-thread probe with a soft timeout — the poisoning risk returns there,
 * but only where the worker path is unavailable.
 * @returns {Promise<Object>} {medianMs, meanMs}
 */
async function _epProbeHardKilled(ep, modelBytes) {
    try {
        return await new Promise((resolve, reject) => {
            let worker;
            try {
                worker = new Worker('js/ep-probe-worker.js');
            } catch (e) {
                reject(Object.assign(new Error('worker unavailable: ' + e), { unsupported: true }));
                return;
            }
            let inactivityTimer = null;
            const die = (err) => {
                clearTimeout(inactivityTimer);
                clearTimeout(capTimer);
                worker.terminate();
                reject(err);
            };
            const armInactivity = () => {
                clearTimeout(inactivityTimer);
                inactivityTimer = setTimeout(() => die(new Error(
                    `${ep} probe stalled: no progress for ${EP_PROBE_INACTIVITY_MS} ms (worker killed)`)),
                    EP_PROBE_INACTIVITY_MS);
            };
            const capTimer = setTimeout(() => die(new Error(
                `${ep} probe exceeded ${EP_PROBE_TOTAL_CAP_MS} ms (worker killed)`)),
                EP_PROBE_TOTAL_CAP_MS);
            worker.onmessage = (e) => {
                const msg = e.data;
                if (msg.status === 'progress') {
                    armInactivity();
                    return;
                }
                clearTimeout(inactivityTimer);
                clearTimeout(capTimer);
                worker.terminate();
                if (msg.status === 'ok') {
                    resolve({ medianMs: msg.medianMs, meanMs: msg.meanMs });
                } else if (msg.status === 'unsupported') {
                    reject(Object.assign(new Error(`${ep} not available in workers`), { unsupported: true }));
                } else {
                    reject(new Error(`${ep} probe failed in worker: ` + msg.message));
                }
            };
            worker.onerror = (e) => {
                // Realistically a script-load failure; runtime errors are
                // caught inside the worker and reported as status 'error'.
                die(Object.assign(new Error('probe worker error: ' + e.message), { unsupported: true }));
            };
            armInactivity();
            worker.postMessage({
                ortUrl: new URL('vendor/ort.webgpu.min.js', location.href).href,
                probeUrl: new URL('js/ep-probe.js', location.href).href,
                wasmPaths: ort.env.wasm.wasmPaths,
                numThreads: ort.env.wasm.numThreads,
                ep: ep,
                modelBytes: modelBytes,
            });
        });
    } catch (e) {
        if (!e.unsupported) throw e;
        console.warn(`EP probe: worker path unavailable, probing ${ep} on main thread:`, e);
        const mainResult = await _epProbeWithTimeout(
            _epProbeOne(ep, modelBytes), EP_PROBE_MAIN_FALLBACK_TIMEOUT_MS, ep);
        try { await mainResult.session.release(); } catch (relErr) { console.warn(relErr); }
        return { medianMs: mainResult.medianMs, meanMs: mainResult.meanMs };
    }
}

/**
 * Run the full probe: WASM first, then WebGPU (if present and WASM is not
 * already clearly fast), each in its own hard-killable worker. Probe
 * sessions die with their workers; the winner's game session is created
 * fresh on the main thread afterwards — the browser's wasm-code and shader
 * caches (shared across workers and the main thread) make that far cheaper
 * than the cold path the probe just paid for.
 * The winning EP is chosen by median (robust to a stray outlier flipping
 * the comparison on so few samples); the mean is carried along for the
 * per-move time estimate.
 * @param {Uint8Array} modelBytes
 * @returns {Promise<Object>} {ep, medianMs, meanMs, session, threads}
 */
async function runEpProbe(modelBytes) {
    const env = epProbeEnvironment();
    let wasm = null;
    let webgpu = null;

    try {
        wasm = await _epProbeHardKilled('wasm', modelBytes);
        console.log(`EP probe: wasm median ${wasm.medianMs.toFixed(1)} ms, `
            + `mean ${wasm.meanMs.toFixed(1)} ms`);
    } catch (e) {
        console.warn('EP probe: wasm failed:', e);
    }

    const wasmIsFast = wasm !== null && wasm.medianMs <= EP_PROBE_WASM_FAST_MS;
    if (env.hasGpu && !wasmIsFast) {
        try {
            webgpu = await _epProbeHardKilled('webgpu', modelBytes);
            console.log(`EP probe: webgpu median ${webgpu.medianMs.toFixed(1)} ms, `
                + `mean ${webgpu.meanMs.toFixed(1)} ms`);
        } catch (e) {
            console.warn('EP probe: webgpu failed:', e);
        }
    }

    if (wasm === null && webgpu === null) {
        throw new Error('All execution provider probes failed');
    }

    const ep = (webgpu !== null
                && (wasm === null || webgpu.medianMs < wasm.medianMs))
        ? 'webgpu' : 'wasm';
    const winner = ep === 'webgpu' ? webgpu : wasm;
    const { session } = await epProbeCreateSession(ep, modelBytes);
    return {
        ep: ep,
        medianMs: winner.medianMs,
        meanMs: winner.meanMs,
        session: session,
        threads: ort.env.wasm.numThreads,
    };
}

/**
 * Load the cached probe result, or null if absent or stale (different ORT
 * version or changed environment — isolation and GPU availability affect
 * which EP wins, and battery/thermal state already makes results only
 * approximately reproducible).
 */
function epProbeLoadCache() {
    let cfg;
    try {
        cfg = JSON.parse(localStorage.getItem(EP_PROBE_STORAGE_KEY));
    } catch {
        return null;
    }
    if (!cfg || typeof cfg.medianMs !== 'number') return null;
    const env = epProbeEnvironment();
    if (cfg.ortVersion !== ort.version) return null;
    if (cfg.isolated !== env.isolated || cfg.hasGpu !== env.hasGpu) return null;
    return cfg;
}

/**
 * Persist the probe result (and later the user's one-time acknowledgment of
 * the time estimate) so subsequent visits skip the probe entirely.
 */
function epProbeSaveCache(cfg) {
    try {
        localStorage.setItem(EP_PROBE_STORAGE_KEY, JSON.stringify(cfg));
    } catch (e) {
        console.warn('Failed to persist EP config:', e);
    }
}
