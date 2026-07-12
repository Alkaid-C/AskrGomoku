/**
 * AI Player Module using ONNX Runtime Web
 *
 * Loads and runs the Gomoku policy network in the browser.
 */

class OnnxAIPlayer {
    constructor(modelPath, temperature = 1.0) {
        this.modelPath = modelPath;
        this.temperature = temperature;
        this.session = null;
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
        const [c0, c1] = board.GetBoardState();
        const inputTensor = this._boardToTensor(c0, c1);
        const results = await this.session.run({ board_state: inputTensor });
        return {
            logits: results.policy_logits.data,
            value: results.value.data[0],
        };
    }

    /**
     * Get AI's next move by sampling the raw policy (junior/intermediate).
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
     * Get AI's next move using MCTS (advanced difficulty).
     * @param {GomokuBoard} board - Current board state
     * @param {number} numSims - Simulation budget
     * @returns {Promise<Array>} [row, col, rootQ] of AI's move
     */
    async getMoveWithMCTS(board, numSims) {
        const search = new MCTSSearch(this);
        const startTime = performance.now();
        const { row, col, rootQ } = await search.search(board, numSims);
        const elapsed = (performance.now() - startTime) / 1000;
        console.log(`MCTS: ${numSims} sims in ${elapsed.toFixed(2)}s, `
            + `move (${row}, ${col}), rootQ = ${rootQ.toFixed(4)}`);
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

        return probs.length - 1;
    }
}
