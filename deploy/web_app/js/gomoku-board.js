/**
 * Gomoku Game Logic (JavaScript port of gomoku_board.py)
 *
 * Implements the game board and rules for Gomoku (Five-in-a-Row).
 */

// Game state constants
const GameState = {
    CONTINUE: 0,
    BLACK_WIN: 1,
    WHITE_WIN: 2,
    DRAW: 3
};

// Player constants
const Player = {
    BLACK: 1,
    WHITE: 2
};

/**
 * Gomoku Board Class
 * Manages a 15x15 board with piece placement and win detection.
 */
class GomokuBoard {
    constructor() {
        // Initialize 15x15 board
        this.blackPieces = Array(15).fill(null).map(() => Array(15).fill(0));
        this.whitePieces = Array(15).fill(null).map(() => Array(15).fill(0));
        this.whoToPlay = Player.BLACK;
        this.occupiedCount = 0;
    }

    /**
     * Make a move on the board.
     * @param {number} row - Row index (0-14)
     * @param {number} col - Column index (0-14)
     * @returns {number} GameState indicating the result
     */
    Move(row, col) {
        // Place piece
        if (this.whoToPlay === Player.BLACK) {
            this.blackPieces[row][col] = 1;
        } else {
            this.whitePieces[row][col] = 1;
        }

        this.occupiedCount++;

        // Check for win
        if (this._checkWin(this.whoToPlay, row, col)) {
            return this.whoToPlay === Player.BLACK ? GameState.BLACK_WIN : GameState.WHITE_WIN;
        }

        // Check for draw (board full)
        if (this.occupiedCount === 225) {
            return GameState.DRAW;
        }

        // Switch players
        this.whoToPlay = (this.whoToPlay === Player.BLACK) ? Player.WHITE : Player.BLACK;

        return GameState.CONTINUE;
    }

    /**
     * Check if the current move resulted in a win.
     * @param {number} player - Player who just moved
     * @param {number} row - Row of the last move
     * @param {number} col - Column of the last move
     * @returns {boolean} True if player won
     */
    _checkWin(player, row, col) {
        const pieces = (player === Player.BLACK) ? this.blackPieces : this.whitePieces;

        // Four directions to check: horizontal, vertical, diagonal, anti-diagonal
        const directions = [
            [0, 1],   // Horizontal
            [1, 0],   // Vertical
            [1, 1],   // Diagonal (\)
            [1, -1]   // Anti-diagonal (/)
        ];

        for (const [dr, dc] of directions) {
            let count = 1; // Count the placed stone itself

            // Count in positive direction
            let r = row + dr;
            let c = col + dc;
            while (r >= 0 && r < 15 && c >= 0 && c < 15 && pieces[r][c] === 1) {
                count++;
                r += dr;
                c += dc;
            }

            // Count in negative direction
            r = row - dr;
            c = col - dc;
            while (r >= 0 && r < 15 && c >= 0 && c < 15 && pieces[r][c] === 1) {
                count++;
                r -= dr;
                c -= dc;
            }

            if (count >= 5) {
                return true;
            }
        }

        return false;
    }

    /**
     * Get legal moves mask.
     * @returns {Array} [legalMask, nextPlayer] where legalMask is 15x15 array
     */
    GetLegalMoves() {
        const legalMask = Array(15).fill(null).map(() => Array(15).fill(0));

        for (let row = 0; row < 15; row++) {
            for (let col = 0; col < 15; col++) {
                if (this.blackPieces[row][col] === 0 && this.whitePieces[row][col] === 0) {
                    legalMask[row][col] = 1;
                }
            }
        }

        return [legalMask, this.whoToPlay];
    }

    /**
     * Get board state for neural network input.
     * Returns perspective of current player (current player's pieces as c0).
     * @returns {Array} [c0, c1, whoToPlay] where c0 is current player, c1 is opponent
     */
    GetBoardState() {
        let c0, c1;

        if (this.whoToPlay === Player.BLACK) {
            c0 = this.blackPieces.map(row => [...row]); // Deep copy
            c1 = this.whitePieces.map(row => [...row]);
        } else {
            c0 = this.whitePieces.map(row => [...row]);
            c1 = this.blackPieces.map(row => [...row]);
        }

        return [c0, c1, this.whoToPlay];
    }

    /**
     * Render board as text (for debugging).
     * @returns {string} Text representation of the board
     */
    Render() {
        let output = "   " + Array.from({length: 15}, (_, i) => String(i).padStart(2)).join(" ") + "\n";

        for (let row = 0; row < 15; row++) {
            let line = String(row).padStart(2) + " ";

            for (let col = 0; col < 15; col++) {
                if (this.blackPieces[row][col] === 1) {
                    line += " x";
                } else if (this.whitePieces[row][col] === 1) {
                    line += " o";
                } else {
                    line += " .";
                }
            }

            output += line + "\n";
        }

        return output;
    }
}
