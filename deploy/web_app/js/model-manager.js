/**
 * Model Manager
 *
 * Handles model selection and loading. Junior/intermediate load a plain
 * WASM session per game. Advanced goes through the EP probe (ep-probe.js)
 * once and caches the resulting player + probe info for later games.
 */

class ModelManager {
    constructor() {
        this.models = {
            junior: {
                path: 'models/dial.onnx',
                temperature: 1.0,
            },
            intermediate: {
                path: 'models/cello.onnx',
                temperature: 0.7,
            },
            advanced: {
                path: 'models/mcts_test.onnx',
            }
        };

        this.selectedModel = null;

        // Advanced-difficulty state, filled by loadAdvancedModel().
        this.advancedPlayer = null;
        this.advancedProbe = null; // {ep, medianMs, threads}
    }

    /**
     * Initialize model manager.
     */
    initialize() {
        console.log('Model Manager initialized');
    }

    /**
     * Set selected model.
     * @param {string} modelType - 'junior', 'intermediate', or 'advanced'
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
     * Load selected model (junior/intermediate policy-sampling path).
     * @returns {Promise<OnnxAIPlayer>} Loaded AI player
     */
    async loadSelectedModel() {
        const modelPath = this.getModelPath();
        const temperature = this.getModelTemperature();
        const aiPlayer = new OnnxAIPlayer(modelPath, temperature);
        await aiPlayer.loadModel();
        return aiPlayer;
    }

    /**
     * Load the advanced model through the EP probe. Downloads the model,
     * then either restores the cached EP choice or runs the full probe.
     * The player (with its warm session) is cached for later games.
     * @param {function} onDownloadProgress - Called with fraction in [0, 1]
     * @param {function} onProbeStart - Called when the timing probe begins
     *     (skipped when a cached EP config is valid)
     * @param {boolean} [forceProbe] - Discard the cached player and probe
     *     result and re-run the full probe (the "re-run test" button)
     * @returns {Promise<Object>} {player, probe: {ep, medianMs, threads}}
     */
    async loadAdvancedModel(onDownloadProgress, onProbeStart, forceProbe = false) {
        if (this.advancedPlayer) {
            if (!forceProbe) {
                return { player: this.advancedPlayer, probe: this.advancedProbe };
            }
            try {
                await this.advancedPlayer.session.release();
            } catch (e) {
                console.warn('Failed to release previous advanced session:', e);
            }
            this.advancedPlayer = null;
            this.advancedProbe = null;
        }

        const modelPath = this.models.advanced.path;
        const modelBytes = await epProbeFetchModel(modelPath, onDownloadProgress);

        let probe = null;
        const cached = forceProbe ? null : epProbeLoadCache();
        if (cached) {
            try {
                const { session } = await epProbeCreateSession(cached.ep, modelBytes);
                probe = {
                    ep: cached.ep,
                    medianMs: cached.medianMs,
                    meanMs: cached.meanMs, // may be undefined in old caches
                    threads: cached.threads,
                    session: session,
                };
                console.log(`EP probe: using cached config (${cached.ep}, `
                    + `${cached.medianMs.toFixed(1)} ms)`);
            } catch (e) {
                // e.g. WebGPU adapter no longer usable — fall through to re-probe
                console.warn('Cached EP config failed to restore, re-probing:', e);
            }
        }

        if (!probe) {
            if (onProbeStart) onProbeStart();
            probe = await runEpProbe(modelBytes);
            const env = epProbeEnvironment();
            epProbeSaveCache({
                ortVersion: ort.version,
                ep: probe.ep,
                threads: probe.threads,
                medianMs: probe.medianMs,
                meanMs: probe.meanMs,
                isolated: env.isolated,
                hasGpu: env.hasGpu,
                ts: Date.now(),
                // A fresh probe means a fresh time estimate: the user must
                // acknowledge it once before it stops being shown.
                confirmed: false,
            });
        }

        const player = new OnnxAIPlayer(modelPath);
        player.session = probe.session;

        this.advancedPlayer = player;
        this.advancedProbe = {
            ep: probe.ep,
            medianMs: probe.medianMs,
            meanMs: probe.meanMs,
            threads: probe.threads,
        };
        return { player: this.advancedPlayer, probe: this.advancedProbe };
    }
}
