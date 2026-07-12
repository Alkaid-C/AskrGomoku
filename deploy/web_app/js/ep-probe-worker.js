/**
 * EP Probe Worker
 *
 * Probes one execution provider inside a dedicated worker so the parent can
 * hard-kill it with worker.terminate(). This isolation matters twice over:
 * session.run() is not abortable, and an abandoned run poisons the shared
 * main-thread wasm runtime (the ort.webgpu bundle's wasm and webgpu EPs
 * share one asyncify wasm instance, which does not support reentrancy) —
 * observed as every later inference running ~6x slower; and main-thread
 * timings are inflated several-fold by the loading screen's rAF animations.
 * Killing the whole worker discards its wasm instance with it.
 *
 * Reuses the timing code from ep-probe.js via importScripts. Per-run
 * progress heartbeats drive the parent's inactivity watchdog, so a slow
 * probe (e.g. a >10 s first-run shader compile) is distinguished from a
 * stuck one.
 */
self.onmessage = async (event) => {
    const cfg = event.data;
    try {
        importScripts(cfg.ortUrl, cfg.probeUrl);
        if (cfg.ep === 'webgpu' && !navigator.gpu) {
            self.postMessage({ status: 'unsupported' });
            return;
        }
        ort.env.wasm.wasmPaths = cfg.wasmPaths;
        ort.env.wasm.numThreads = cfg.numThreads;
        const { medianMs, meanMs } = await _epProbeOne(cfg.ep, cfg.modelBytes,
            (phase) => self.postMessage({ status: 'progress', phase: phase }));
        // No session cleanup: the parent terminates this worker either way.
        self.postMessage({ status: 'ok', medianMs: medianMs, meanMs: meanMs });
    } catch (err) {
        self.postMessage({
            status: 'error',
            message: String((err && err.message) || err),
        });
    }
};
