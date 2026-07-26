/**
 * AI Player Module using ONNX Runtime Web
 *
 * Loads and runs the Gomoku policy network in the browser.
 */

class OnnxAIPlayer {
    /**
     * @param {string} modelPath - ONNX model URL
     * @param {number} temperature - Policy-sampling temperature
     * @param {number} evalCacheMaxEntries - 0 disables the evaluation LRU
     */
    constructor(modelPath, temperature = 1.0, evalCacheMaxEntries = 0) {
        this.modelPath = modelPath;
        this.temperature = temperature;
        this.session = null;
        this.evalCacheMaxEntries = evalCacheMaxEntries;
        this.evalCache = evalCacheMaxEntries > 0 ? new Map() : null;
        this.evalCacheSeed = [];
        if (globalThis.GOMOKU_DEBUG) {
            this.evalCacheHits = 0;
            this.evalCacheMisses = 0;
            this.evalCacheEvictions = 0;
        }
    }

    /**
     * Load the ONNX model.
     * @returns {Promise<void>}
     */
    async loadModel() {
        console.log(`Loading model from: ${this.modelPath}`);

        const options = {
            executionProviders: ['wasm'],
            graphOptimizationLevel: 'all'
        };

        this.session = await ort.InferenceSession.create(this.modelPath, options);

        console.log('Model loaded successfully');
    }

    /**
     * Run one inference on a board position.
     * @param {GomokuBoard} board - Position to evaluate
     * @returns {Promise<Object>} {logits: Float32Array[225] raw policy logits,
     *     value: number in [-1, 1] from the side-to-move's perspective}
     */
    async evaluate(board) {
        let cacheKey = null;
        if (this.evalCache !== null) {
            cacheKey = board.GetEvalCacheKey();
            const cached = this.evalCache.get(cacheKey);
            if (cached !== undefined) {
                // Map iteration order is the LRU order. Reinsert a hit so it
                // becomes the most-recently-used entry.
                this.evalCache.delete(cacheKey);
                this.evalCache.set(cacheKey, cached);
                if (globalThis.GOMOKU_DEBUG) this.evalCacheHits++;
                return cached;
            }
            if (globalThis.GOMOKU_DEBUG) this.evalCacheMisses++;
        }

        const [c0, c1] = board.GetBoardState();
        const inputTensor = this._boardToTensor(c0, c1);
        const results = await this.session.run({ board_state: inputTensor });
        const evaluation = {
            // Cache entries own their data rather than depending on the
            // lifetime or buffer-reuse behavior of an ORT output tensor.
            logits: this.evalCache === null
                ? results.policy_logits.data
                : new Float32Array(results.policy_logits.data),
            value: results.value.data[0],
        };

        if (this.evalCache !== null) {
            this.evalCache.set(cacheKey, evaluation);
            if (this.evalCache.size > this.evalCacheMaxEntries) {
                const oldestKey = this.evalCache.keys().next().value;
                this.evalCache.delete(oldestKey);
                if (globalThis.GOMOKU_DEBUG) this.evalCacheEvictions++;
            }
        }
        return evaluation;
    }

    /**
     * Install the server-provided opening cache in LRU order (oldest first).
     * The entries are immutable seed data and are restored for every new game.
     * @param {Array<Object>} entries
     */
    setEvalCacheSeed(entries) {
        if (this.evalCache === null) {
            if (entries.length !== 0) {
                throw new Error('Cannot seed a disabled evaluation cache');
            }
            return;
        }
        if (entries.length > this.evalCacheMaxEntries) {
            throw new Error(
                `Eval cache seed has ${entries.length} entries; `
                + `capacity is ${this.evalCacheMaxEntries}`);
        }
        this.evalCacheSeed = entries.slice();
        this.resetEvalCache();
    }

    /**
     * Start a new per-game cache lifetime. The player and its ORT session stay
     * alive; the server seed is restored and runtime statistics are cleared.
     */
    resetEvalCache() {
        if (this.evalCache !== null) {
            this.evalCache.clear();
            for (const entry of this.evalCacheSeed) {
                this.evalCache.set(entry.key, entry.evaluation);
            }
        }
        if (globalThis.GOMOKU_DEBUG) {
            this.evalCacheHits = 0;
            this.evalCacheMisses = 0;
            this.evalCacheEvictions = 0;
        }
    }

    /**
     * Get AI's next move by sampling the raw policy (dial/cello/curtain).
     * @param {GomokuBoard} board - Current board state
     * @returns {Promise<Array>} [row, col, value] of AI's move
     */
    async getMove(board) {
        const { logits, value } = await this.evaluate(board);

        console.log(`Position value: ${value.toFixed(3)}`);

        // Get legal moves mask
        const [legalMask, _] = board.GetLegalMoves();
        const legalMaskFlat = this._flattenMask(legalMask);

        // Masked softmax with temperature
        const probs = this._maskedSoftmax(logits, legalMaskFlat, this.temperature);

        // Sample from probability distribution
        const actionIdx = this._sampleCategorical(probs);

        // Convert flat index to (row, col)
        const row = Math.floor(actionIdx / 15);
        const col = actionIdx % 15;

        return [row, col, value];
    }

    /**
     * Get AI's next move using MCTS (melody difficulty).
     * @param {GomokuBoard} board - Current board state
     * @param {number} numSims - Simulation budget
     * @returns {Promise<Array>} [row, col, rootQ] of AI's move
     */
    async getMoveWithMCTS(board, numSims) {
        const search = new MCTSSearch(this);
        const cacheBefore = globalThis.GOMOKU_DEBUG
            ? this.getEvalCacheStats() : null;
        const startTime = globalThis.GOMOKU_DEBUG
            ? performance.now() : 0;
        const { row, col, rootQ } = await search.search(board, numSims);
        if (globalThis.GOMOKU_DEBUG) {
            const elapsed = (performance.now() - startTime) / 1000;
            console.log(`MCTS: ${numSims} sims in ${elapsed.toFixed(2)}s, `
                + `move (${row}, ${col}), rootQ = ${rootQ.toFixed(4)}`);
            if (cacheBefore.enabled) {
                const cacheAfter = this.getEvalCacheStats();
                const hits = cacheAfter.hits - cacheBefore.hits;
                const misses = cacheAfter.misses - cacheBefore.misses;
                const evictions = cacheAfter.evictions - cacheBefore.evictions;
                const lookups = hits + misses;
                const hitRate = lookups > 0 ? hits / lookups : 0;
                console.log(`MCTS eval cache: ${hits}/${lookups} hits `
                    + `(${(100 * hitRate).toFixed(1)}%), ${misses} ONNX runs, `
                    + `${evictions} evictions, size ${cacheAfter.size}/${cacheAfter.maxEntries}`);
            }
        }
        return [row, col, rootQ];
    }

    /**
     * Convert board state to ONNX tensor.
     * @param {Array} c0 - Current player's pieces [15][15]
     * @param {Array} c1 - Opponent's pieces [15][15]
     * @returns {ort.Tensor} Input tensor [2, 15, 15] (no batch dimension)
     */
    _boardToTensor(c0, c1) {
        const data = new Float32Array(450);

        // Fill channel 0 (current player)
        for (let i = 0; i < 15; i++) {
            for (let j = 0; j < 15; j++) {
                data[i * 15 + j] = c0[i][j];
            }
        }

        // Fill channel 1 (opponent)
        for (let i = 0; i < 15; i++) {
            for (let j = 0; j < 15; j++) {
                data[225 + i * 15 + j] = c1[i][j];
            }
        }

        return new ort.Tensor('float32', data, [2, 15, 15]);
    }

    /**
     * Apply masked softmax with temperature over logits.
     * Only legal positions participate; illegal positions get probability 0.
     * Uses max subtraction for numerical stability.
     * @param {Array} logits - Raw logits [225]
     * @param {Array} legalMask - Binary mask [225] (1 = legal, 0 = illegal)
     * @param {number} temperature - Softmax temperature
     * @returns {Array} Probability distribution [225]
     */
    _maskedSoftmax(logits, legalMask, temperature) {
        // Find max logit among legal moves (for numerical stability)
        let maxLogit = -Infinity;
        for (let i = 0; i < 225; i++) {
            if (legalMask[i] === 1) {
                const scaled = logits[i] / temperature;
                if (scaled > maxLogit) maxLogit = scaled;
            }
        }

        // Compute exp(logit/T - max) for legal moves, sum them
        const probs = new Array(225);
        let sum = 0;
        for (let i = 0; i < 225; i++) {
            if (legalMask[i] === 0) {
                probs[i] = 0;
            } else {
                probs[i] = Math.exp(logits[i] / temperature - maxLogit);
                sum += probs[i];
            }
        }

        // Normalize
        for (let i = 0; i < 225; i++) {
            probs[i] /= sum;
        }

        return probs;
    }

    /**
     * Flatten 2D legal mask to 1D array.
     * @param {Array} mask - 15x15 legal moves mask
     * @returns {Array} Flattened 225-element array
     */
    _flattenMask(mask) {
        const flat = new Array(225);
        for (let i = 0; i < 15; i++) {
            for (let j = 0; j < 15; j++) {
                flat[i * 15 + j] = mask[i][j];
            }
        }
        return flat;
    }

    /**
     * Sample from categorical distribution.
     * @param {Array} probs - Probability distribution (must sum to 1)
     * @returns {number} Sampled index
     * @throws {Error} If the cumulative sum never reaches rand — with a
     *     valid distribution that is astronomically rare float rounding, so
     *     in practice it means the distribution is broken (e.g. all NaN from
     *     non-finite logits). Failing loudly beats silently returning an
     *     index that may be an illegal move.
     */
    _sampleCategorical(probs) {
        const rand = Math.random();
        let cumsum = 0;

        for (let i = 0; i < probs.length; i++) {
            cumsum += probs[i];
            if (rand <= cumsum) {
                return i;
            }
        }

        throw new Error(`Invalid probability distribution: cumulative sum ${cumsum}`);
    }
}

if (globalThis.GOMOKU_DEBUG) {
    /**
     * Development-only cache statistics used by test.html.
     * @returns {Object} Snapshot of cumulative statistics for the current game
     */
    OnnxAIPlayer.prototype.getEvalCacheStats = function() {
        return {
            enabled: this.evalCache !== null,
            hits: this.evalCacheHits,
            misses: this.evalCacheMisses,
            evictions: this.evalCacheEvictions,
            size: this.evalCache === null ? 0 : this.evalCache.size,
            maxEntries: this.evalCacheMaxEntries,
            seedEntries: this.evalCacheSeed.length,
        };
    };
}
