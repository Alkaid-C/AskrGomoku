/**
 * AI Player Module using ONNX Runtime Web
 *
 * Loads and runs the Gomoku policy network in the browser.
 */

class OnnxAIPlayer {
    constructor(modelPath) {
        this.modelPath = modelPath;
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
     * Get AI's next move for a given board state.
     * @param {GomokuBoard} board - Current board state
     * @returns {Promise<Array>} [row, col] of AI's move
     */
    async getMove(board) {
        // Get board state from current player's perspective
        const [c0, c1, whoToPlay] = board.GetBoardState();

        // Convert to tensor [2, 15, 15] (no batch dimension)
        const inputTensor = this._boardToTensor(c0, c1);

        // Run inference
        const feeds = { board_state: inputTensor };
        const results = await this.session.run(feeds);

        // Get output probabilities [15, 15] (flattened to [225])
        const policyOutput = results.policy_probs;
        const probs = Array.from(policyOutput.data);

        console.log('Policy probs sum:', probs.reduce((a, b) => a + b, 0).toFixed(6));
        console.log(`Position value: ${results.value.data[0].toFixed(3)}`);

        // Get legal moves mask
        const [legalMask, _] = board.GetLegalMoves();
        const legalMaskFlat = this._flattenMask(legalMask);

        // Mask illegal moves and renormalize
        let maskedProbs = new Array(225);
        let sum = 0;
        for (let i = 0; i < 225; i++) {
            if (legalMaskFlat[i] === 0) {
                maskedProbs[i] = 0;
            } else {
                maskedProbs[i] = probs[i];
                sum += probs[i];
            }
        }

        // Renormalize to sum to 1.0
        for (let i = 0; i < 225; i++) {
            maskedProbs[i] /= sum;
        }

        // Sample from probability distribution (temperature is baked into model)
        const actionIdx = this._sampleCategorical(maskedProbs);

        // Convert flat index to (row, col)
        const row = Math.floor(actionIdx / 15);
        const col = actionIdx % 15;

        return [row, col];
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
