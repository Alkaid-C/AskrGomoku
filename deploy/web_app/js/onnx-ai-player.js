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
        const [c0, c1] = board.GetBoardState();

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

        return [row, col, results.value.data[0]];
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

    // ========================================================================
    // Negamax Search Methods
    // ========================================================================

    /**
     * Get AI's next move using negamax search.
     * @param {GomokuBoard} board - Current board state
     * @param {number} depth - Search depth (default 3)
     * @param {number} topK - Number of top candidates to consider (default 3)
     * @returns {Promise<Array>} [row, col] of AI's move
     */
    async getMoveWithNegamax(board, depth = 3, topK = 3) {
        console.log(`Starting negamax search: depth=${depth}, topK=${topK}`);

        // Get top-k candidates at root
        const candidates = await this._getTopKActions(board, topK);

        if (candidates.length === 0) {
            console.error('No legal moves available');
            return [7, 7]; // Fallback to center
        }

        if (candidates.length === 1) {
            // Only one legal move
            return [candidates[0].row, candidates[0].col];
        }

        // Evaluate each candidate with negamax
        let bestAction = candidates[0];
        let bestQ = -Infinity;
        let nodeCount = 0;

        for (const action of candidates) {
            const clonedBoard = board.clone();
            const result = clonedBoard.Move(action.row, action.col);

            let value;
            if (result !== GameState.CONTINUE) {
                // Terminal state
                value = this._getTerminalValue(result, board.whoToPlay);
                nodeCount++;
            } else {
                // Recursive search
                const searchResult = await this._negamax(clonedBoard, depth - 1, topK);
                value = -searchResult.value;
                nodeCount += searchResult.nodeCount;
            }

            console.log(`  Candidate (${action.row}, ${action.col}): Q = ${value.toFixed(4)}`);

            if (value > bestQ) {
                bestQ = value;
                bestAction = action;
            }
        }

        console.log(`Negamax complete: ${nodeCount} nodes evaluated`);
        console.log(`Best move: (${bestAction.row}, ${bestAction.col}) with Q = ${bestQ.toFixed(4)}`);

        return [bestAction.row, bestAction.col, bestQ];
    }

    /**
     * Recursive negamax search.
     * @param {GomokuBoard} board - Current board state
     * @param {number} depth - Remaining search depth
     * @param {number} topK - Number of candidates per node
     * @returns {Promise<Object>} {value, nodeCount}
     */
    async _negamax(board, depth, topK) {
        // Base case: leaf node - evaluate with value network
        if (depth === 0) {
            const value = await this._evaluatePosition(board);
            return { value, nodeCount: 1 };
        }

        // Get top-k candidates
        const candidates = await this._getTopKActions(board, topK);

        if (candidates.length === 0) {
            // No legal moves (shouldn't happen in Gomoku before terminal)
            const value = await this._evaluatePosition(board);
            return { value, nodeCount: 1 };
        }

        let maxValue = -Infinity;
        let totalNodeCount = 0;

        for (const action of candidates) {
            const clonedBoard = board.clone();
            const result = clonedBoard.Move(action.row, action.col);

            let value;
            if (result !== GameState.CONTINUE) {
                // Terminal state - value from current player's perspective
                value = this._getTerminalValue(result, board.whoToPlay);
                totalNodeCount++;
            } else {
                // Recursive search
                const searchResult = await this._negamax(clonedBoard, depth - 1, topK);
                value = -searchResult.value;
                totalNodeCount += searchResult.nodeCount;
            }

            if (value > maxValue) {
                maxValue = value;
            }
        }

        return { value: maxValue, nodeCount: totalNodeCount };
    }

    /**
     * Get top-k actions by policy probability.
     * @param {GomokuBoard} board - Current board state
     * @param {number} k - Number of top actions to return
     * @returns {Promise<Array>} Array of {row, col, idx, prob}
     */
    async _getTopKActions(board, k) {
        // Get board state from current player's perspective
        const [c0, c1] = board.GetBoardState();

        // Convert to tensor
        const inputTensor = this._boardToTensor(c0, c1);

        // Run inference
        const feeds = { board_state: inputTensor };
        const results = await this.session.run(feeds);

        // Get policy probabilities
        const policyOutput = results.policy_probs;
        const probs = Array.from(policyOutput.data);

        // Get legal moves mask
        const [legalMask, _] = board.GetLegalMoves();

        // Collect legal actions with their probabilities
        const legalActions = [];
        for (let row = 0; row < 15; row++) {
            for (let col = 0; col < 15; col++) {
                if (legalMask[row][col] === 1) {
                    const idx = row * 15 + col;
                    legalActions.push({
                        row,
                        col,
                        idx,
                        prob: probs[idx]
                    });
                }
            }
        }

        // Sort by probability descending
        legalActions.sort((a, b) => b.prob - a.prob);

        // Return top-k
        return legalActions.slice(0, k);
    }

    /**
     * Evaluate a position using the value network.
     * @param {GomokuBoard} board - Board state to evaluate
     * @returns {Promise<number>} Position value from current player's perspective
     */
    async _evaluatePosition(board) {
        const [c0, c1] = board.GetBoardState();
        const inputTensor = this._boardToTensor(c0, c1);

        const feeds = { board_state: inputTensor };
        const results = await this.session.run(feeds);

        // Value is from current player's perspective
        return results.value.data[0];
    }

    /**
     * Get terminal state value.
     * @param {number} result - GameState result
     * @param {number} playerAtMove - Player who was about to move (before terminal)
     * @returns {number} Value: +1 for win, -1 for loss, 0 for draw
     */
    _getTerminalValue(result, playerAtMove) {
        if (result === GameState.DRAW) {
            return 0;
        }

        // The player who just moved (not playerAtMove) caused the terminal state
        // If BLACK_WIN and playerAtMove was WHITE, then BLACK (opponent of WHITE) won
        // which means the move made by BLACK resulted in a win for BLACK

        if (result === GameState.BLACK_WIN) {
            // Black won - if playerAtMove was Black (meaning Black just moved and won)
            // then from Black's perspective this is +1
            return playerAtMove === Player.BLACK ? 1 : -1;
        } else if (result === GameState.WHITE_WIN) {
            return playerAtMove === Player.WHITE ? 1 : -1;
        }

        return 0; // Shouldn't reach here
    }
}
