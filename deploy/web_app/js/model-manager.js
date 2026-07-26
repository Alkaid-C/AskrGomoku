/**
 * Model Manager
 *
 * Handles model selection and loading. Dial/cello/curtain load a plain
 * WASM session per game. Melody goes through the EP probe (ep-probe.js)
 * once and caches the resulting player + probe info for later games.
 */

class ModelManager {
    constructor() {
        this.models = {
            dial: {
                path: 'models/dial.onnx',
                temperature: 1.0,
            },
            cello: {
                path: 'models/cello.onnx',
                temperature: 1.0,
            },
            curtain: {
                path: 'models/curtain.onnx',
                temperature: 0.5,
            },
            // Same model as curtain; played with MCTS instead of raw policy.
            melody: {
                path: 'models/curtain.onnx',
            }
        };

        this.selectedModel = null;

        // Plain policy-sampling sessions are replaced between games. Keep an
        // explicit owner so their native ORT resources can be released.
        this.policyPlayer = null;

        // Melody-difficulty state, filled by fetchMelodyModel()/probeMelody().
        this.melodyModelBytes = null;
        this.melodyPlayer = null;
        this.melodyProbe = null; // {ep, medianMs, meanMs, wasmMeanMs, threads}
    }

    /**
     * Initialize model manager.
     */
    initialize() {
        console.log('Model Manager initialized');
    }

    /**
     * Set selected model.
     * @param {string} modelType - 'dial', 'cello', 'curtain', or 'melody'
     */
    setSelectedModel(modelType) {
        this.selectedModel = modelType;
    }

    /**
     * Get selected model path.
     * @returns {string} Path to ONNX model file
     */
    getModelPath() {
        return this.models[this.selectedModel].path;
    }

    /**
     * Get selected model temperature.
     * @returns {number} Temperature for softmax sampling
     */
    getModelTemperature() {
        return this.models[this.selectedModel].temperature;
    }

    /**
     * Load selected model (dial/cello/curtain policy-sampling path).
     * @returns {Promise<OnnxAIPlayer>} Loaded AI player
     */
    async loadSelectedModel() {
        const modelPath = this.getModelPath();
        const temperature = this.getModelTemperature();
        const aiPlayer = new OnnxAIPlayer(modelPath, temperature);
        await aiPlayer.loadModel();
        this.policyPlayer = aiPlayer;
        return aiPlayer;
    }

    /**
     * Release the current dial/cello/curtain session, if any. The controller
     * waits for active inference to finish before calling this method.
     */
    async releasePolicyPlayer() {
        if (!this.policyPlayer) return;
        const player = this.policyPlayer;
        this.policyPlayer = null;
        try {
            await player.session.release();
        } catch (e) {
            console.warn('Failed to release previous policy session:', e);
        }
    }

    /**
     * Download the melody model into memory (once; later calls are no-ops).
     * Kept on the manager so a probe re-run never re-downloads.
     * @param {function} onProgress - Called with fraction in [0, 1]
     */
    async fetchMelodyModel(onProgress) {
        if (!this.melodyModelBytes) {
            this.melodyModelBytes =
                await epProbeFetchModel(this.models.melody.path, onProgress);
        }
    }

    /**
     * Prepare the melody player from the already-downloaded model: either
     * restore the cached EP choice or run the full probe. The player (with
     * its game session) is cached for later games.
     * @param {Object} [opts]
     * @param {function} [opts.onProbeStart] - Called when the timing probe
     *     begins (skipped when a cached EP config is valid)
     * @param {function} [opts.onPhase] - Forwarded to runEpProbe: called
     *     with (ep, phase) as the probe advances
     * @param {boolean} [opts.force] - Discard the cached player and probe
     *     result and re-run the full probe (the "re-run test" button)
     * @returns {Promise<Object>} {player, probe} with winning-EP timings and
     *     the independently measured WASM mean used for Curtain's estimate
     */
    async probeMelody({ onProbeStart, onPhase, force = false } = {}) {
        if (!this.melodyModelBytes) {
            throw new Error('probeMelody called before fetchMelodyModel');
        }
        if (this.melodyPlayer) {
            if (!force) {
                return { player: this.melodyPlayer, probe: this.melodyProbe };
            }
            this.melodyPlayer.resetEvalCache();
            try {
                await this.melodyPlayer.session.release();
            } catch (e) {
                console.warn('Failed to release previous melody session:', e);
            }
            this.melodyPlayer = null;
            this.melodyProbe = null;
        }

        const modelBytes = this.melodyModelBytes;

        let probe = null;
        const cached = force ? null : epProbeLoadCache();
        if (cached) {
            try {
                const { session } = await epProbeCreateSession(cached.ep, modelBytes);
                probe = {
                    ep: cached.ep,
                    medianMs: cached.medianMs,
                    meanMs: cached.meanMs,
                    wasmMeanMs: cached.wasmMeanMs,
                    threads: cached.threads,
                    session: session,
                };
                console.log(`EP probe: using cached config (${cached.ep}, `
                    + `${cached.meanMs.toFixed(1)} ms mean)`);
            } catch (e) {
                // e.g. WebGPU adapter no longer usable — fall through to re-probe
                console.warn('Cached EP config failed to restore, re-probing:', e);
            }
        }

        if (!probe) {
            if (onProbeStart) onProbeStart();
            probe = await runEpProbe(modelBytes, onPhase);
            const env = epProbeEnvironment();
            epProbeSaveCache({
                probeVersion: EP_PROBE_VERSION,
                ortVersion: ort.version,
                ep: probe.ep,
                threads: probe.threads,
                medianMs: probe.medianMs,
                meanMs: probe.meanMs,
                wasmMeanMs: probe.wasmMeanMs,
                isolated: env.isolated,
                hasGpu: env.hasGpu,
                ts: Date.now(),
                // A fresh probe means a fresh time estimate: the user must
                // acknowledge it once before it stops being shown.
                confirmed: false,
            });
        }

        const player = new OnnxAIPlayer(
            this.models.melody.path,
            1.0,
            MCTS_EVAL_CACHE_MAX_ENTRIES,
        );
        player.session = probe.session;

        this.melodyPlayer = player;
        this.melodyProbe = {
            ep: probe.ep,
            medianMs: probe.medianMs,
            meanMs: probe.meanMs,
            wasmMeanMs: probe.wasmMeanMs,
            threads: probe.threads,
        };
        return { player: this.melodyPlayer, probe: this.melodyProbe };
    }
}
