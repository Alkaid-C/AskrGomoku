/**
 * MCTS Search Module
 *
 * JavaScript port of the Python/C++ MCTS used in stage-2 training
 * (mcts/mcts.py + mcts/mcts_ext.cpp), inference-only. Must mirror the
 * training conventions exactly; constants below come from mcts/main.py.
 *
 * Deliberately omitted (training-only, no effect on inference numerics):
 * Dirichlet root noise, entropy rescaling, D4 canonicalization, subtree
 * harvesting. Melody instead uses an exact-position eval LRU owned by its
 * OnnxAIPlayer, without merging or reusing tree nodes.
 */

const MCTS_C_PUCT = 1.25;
const MCTS_DISCOUNT_GAMMA = 63 / 64;
const MCTS_FPU_MULTIPLIER = 0.95;
// Move-sampling temperature over root visit counts (STAGE2_ACTION_TEMPERATURE).
const MCTS_ACTION_TEMPERATURE = 0.5;

// Simulation budget for melody-difficulty moves.
const MCTS_SIMS = 384;
// Maximum neural-network evaluations retained within one melody game.
const MCTS_EVAL_CACHE_MAX_ENTRIES = 2048;

class MCTSNode {
    constructor(parent, parentK) {
        this.parent = parent;     // null for root
        this.parentK = parentK;   // index into parent's child arrays
        // Parallel child arrays over legal actions, filled by expand().
        this.childActions = null; // flat indices (row*15 + col)
        this.childPriors = null;
        this.childQ = null;
        this.childN = null;
        this.childTotal = null;
        this.childNodes = null;   // materialized lazily in selectChild()
        this.isExpanded = false;
        this.isTerminal = false;
        this.terminalValue = 0;   // from the PARENT's perspective (+1 win, 0 draw)
        this.visitCount = 0;
    }

    /**
     * Expand with legal actions and their priors. Every untried child's Q
     * starts at fpuValue = nodeValue * MCTS_FPU_MULTIPLIER, not 0 — a first
     * play urgency matching mcts_ext.cpp.
     */
    expand(actions, priors, fpuValue) {
        const n = actions.length;
        this.childActions = actions;
        this.childPriors = priors;
        this.childQ = new Float32Array(n).fill(fpuValue);
        this.childN = new Int32Array(n);
        this.childTotal = new Float32Array(n);
        this.childNodes = new Array(n).fill(null);
        this.isExpanded = true;
    }

    /**
     * PUCT selection: argmax_k childQ[k] + cPuct*sqrt(N)*P[k]/(1+n[k]).
     * First-max tie-break (strict >), matching the C++ implementation.
     * @returns {number} Selected child index k (child node materialized).
     */
    selectChild() {
        const cSqrt = MCTS_C_PUCT * Math.sqrt(this.visitCount);
        let bestK = 0;
        let bestScore = -Infinity;
        for (let k = 0; k < this.childActions.length; k++) {
            const score = this.childQ[k]
                + cSqrt * this.childPriors[k] / (1 + this.childN[k]);
            if (score > bestScore) {
                bestScore = score;
                bestK = k;
            }
        }
        if (this.childNodes[bestK] === null) {
            this.childNodes[bestK] = new MCTSNode(this, bestK);
        }
        return bestK;
    }

    /**
     * Backup a value up the tree. The sign flip + discount is applied at
     * EVERY level including the leaf itself, BEFORE accumulation — the
     * first flip converts the leaf's side-to-move value to the parent's
     * perspective (negamax convention, matching mcts_ext.cpp).
     */
    backup(value) {
        let v = value;
        let node = this;
        while (node !== null) {
            v = -v * MCTS_DISCOUNT_GAMMA;
            node.visitCount += 1;
            if (node.parent !== null) {
                const k = node.parentK;
                node.parent.childTotal[k] += v;
                node.parent.childN[k] += 1;
                node.parent.childQ[k] = node.parent.childTotal[k] / node.parent.childN[k];
            }
            node = node.parent;
        }
    }
}

class MCTSSearch {
    /**
     * @param {OnnxAIPlayer} aiPlayer - Provides evaluate(board) -> {logits, value}
     */
    constructor(aiPlayer) {
        this.aiPlayer = aiPlayer;
    }

    /**
     * Run MCTS from the given position and sample a move from the root
     * visit counts at MCTS_ACTION_TEMPERATURE.
     * Fresh tree every call (no reuse between moves, matching self_play.py).
     * @param {GomokuBoard} board - Position to search from (not mutated)
     * @param {number} numSims - Simulation budget
     * @returns {Promise<Object>} {row, col, rootQ}
     */
    async search(board, numSims) {
        const root = new MCTSNode(null, -1);
        // Virtual visit so the first PUCT selection uses priors
        // (sqrt(visitCount)=1 rather than 0), matching mcts.py.
        root.visitCount = 1;

        const rootEval = await this.aiPlayer.evaluate(board);
        const rootLegal = this._legalPriors(board, rootEval.logits);
        root.expand(rootLegal.actions, rootLegal.priors,
                    rootEval.value * MCTS_FPU_MULTIPLIER);

        for (let sim = 0; sim < numSims; sim++) {
            // PUCT descent; stops at unexpanded or terminal nodes.
            let node = root;
            const actionPath = [];
            while (node.isExpanded && !node.isTerminal) {
                const k = node.selectChild();
                actionPath.push(node.childActions[k]);
                node = node.childNodes[k];
            }

            // Terminal shortcut: re-backup the cached value, no replay.
            if (node.isTerminal) {
                node.backup(-node.terminalValue);
                continue;
            }

            // Replay the path on a cloned board to reach the leaf position.
            const simBoard = board.clone();
            let reachedTerminal = false;
            for (const action of actionPath) {
                const outcome = simBoard.Move(Math.floor(action / 15), action % 15);
                if (outcome !== GameState.CONTINUE) {
                    // The player who just moved (= the parent's side) won.
                    node.isTerminal = true;
                    node.terminalValue = outcome === GameState.DRAW ? 0.0 : 1.0;
                    node.backup(-node.terminalValue);
                    reachedTerminal = true;
                    break;
                }
            }
            if (reachedTerminal) continue;

            // Non-terminal leaf: evaluate, expand, backup the NN value
            // (side-to-move perspective; backup's first flip handles signs).
            const leafEval = await this.aiPlayer.evaluate(simBoard);
            const leafLegal = this._legalPriors(simBoard, leafEval.logits);
            node.expand(leafLegal.actions, leafLegal.priors,
                        leafEval.value * MCTS_FPU_MULTIPLIER);
            node.backup(leafEval.value);
        }

        // Sample the move from visits ** (1/T), renormalized — the same
        // action-temperature sampling as stage-2 self-play (self_play.py).
        let totalN = 0;
        let weightedQ = 0;
        for (let k = 0; k < root.childN.length; k++) {
            totalN += root.childN[k];
            weightedQ += root.childN[k] * root.childQ[k];
        }
        const rootQ = totalN > 0 ? weightedQ / totalN : rootEval.value;
        const sampleDist = new Float64Array(root.childN.length);
        let distSum = 0;
        for (let k = 0; k < root.childN.length; k++) {
            sampleDist[k] = Math.pow(root.childN[k], 1 / MCTS_ACTION_TEMPERATURE);
            distSum += sampleDist[k];
        }
        for (let k = 0; k < sampleDist.length; k++) {
            sampleDist[k] /= distSum;
        }
        const sampledK = this.aiPlayer._sampleCategorical(sampleDist);
        const action = root.childActions[sampledK];
        return {
            row: Math.floor(action / 15),
            col: action % 15,
            rootQ: rootQ,
            root: root,
        };
    }

    /**
     * Compute priors over legal actions: mask illegal squares, softmax over
     * all 225 (temperature 1), then keep only the legal entries — the same
     * masked-softmax pipeline as mcts.py's entropy_multiplier=None branch.
     * @returns {Object} {actions: number[], priors: Float32Array}
     */
    _legalPriors(board, logits) {
        const [legalMask] = board.GetLegalMoves();
        const flatMask = this.aiPlayer._flattenMask(legalMask);
        const probs = this.aiPlayer._maskedSoftmax(logits, flatMask, 1.0);
        const actions = [];
        for (let i = 0; i < 225; i++) {
            if (flatMask[i] === 1) actions.push(i);
        }
        const priors = new Float32Array(actions.length);
        for (let k = 0; k < actions.length; k++) {
            priors[k] = probs[actions[k]];
        }
        return { actions, priors };
    }
}
