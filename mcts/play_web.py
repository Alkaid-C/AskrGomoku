"""
Gomoku MCTS Web Interface - Flask Server (Standalone)

Simple web server for interactive testing of trained Gomoku models, with the
suggestion driven by an actual AlphaZero MCTS search (PUCT + visit distribution)
on top of the network — matching how an mcts/ checkpoint behaves in deployment,
rather than the raw single-forward policy. Lets humans play against the search
and tune the search knobs (budget, c_puct, Dirichlet noise, discount) live.

All HTML, CSS, and JavaScript are embedded in this file for portability.
"""

import glob
import os
import sys

# Ensure imports resolve from cwd (needed when this file is symlinked from another directory)
sys.path.insert(0, os.getcwd())
import numpy as np
import torch
from flask import Flask, jsonify, request
from gomoku import Player, board_from_observation, encode_observation, idx_to_pos
from model import GomokuPolicyNet

from mcts import _evaluate_with_cache, clear_nn_eval_cache, mcts_search_batched

app = Flask(__name__)

# MCTS inference runs on CPU (keep the default budget modest so search stays
# responsive). entropy_multiplier is fixed to None throughout = raw masked-softmax
# priors = vanilla AlphaZero = stage-2 deployment behavior.
DEVICE = torch.device("cuda")

# Defaults for the search knobs exposed in the UI; mirror mcts/main.py.
DEFAULT_NUM_SIMULATIONS = 256
DEFAULT_C_PUCT = 1.25
DEFAULT_DIRICHLET_ALPHA = 0.125
DEFAULT_DIRICHLET_EPSILON = 0.0  # noise off by default for reproducible analysis
DEFAULT_GAMMA = 63.0 / 64.0
DEFAULT_FPU_MULTIPLIER = 0.95  # First Play Urgency scale; matches stage-2 training

# Global state
current_model = None
current_checkpoint_name = None

# ============================================================================
# Embedded Static Files
# ============================================================================

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Gomoku MCTS Playground</title>
    <style>
/* Global Styles */
* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    min-height: 100vh;
    padding: 20px;
}

.container {
    max-width: 1400px;
    margin: 0 auto;
}

h1 {
    text-align: center;
    color: white;
    margin-bottom: 20px;
    font-size: 2.5em;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
}

/* Control Panel */
.control-panel {
    background: white;
    border-radius: 10px;
    padding: 20px;
    margin-bottom: 20px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}

.checkpoint-selector {
    display: flex;
    align-items: center;
    gap: 10px;
    flex-wrap: wrap;
}

.checkpoint-selector label {
    font-weight: bold;
}

.checkpoint-selector select {
    flex: 1;
    min-width: 200px;
    padding: 8px;
    border: 2px solid #667eea;
    border-radius: 5px;
    font-size: 14px;
}

.checkpoint-selector button {
    padding: 8px 20px;
    background: #667eea;
    color: white;
    border: none;
    border-radius: 5px;
    cursor: pointer;
    font-weight: bold;
    transition: background 0.3s;
}

.checkpoint-selector button:hover {
    background: #5568d3;
}

.status {
    padding: 5px 10px;
    border-radius: 5px;
    font-weight: bold;
}

.status.loaded {
    background: #4caf50;
    color: white;
}

.status.error {
    background: #f44336;
    color: white;
}

/* Game Container */
.game-container {
    display: grid;
    grid-template-columns: auto 400px;
    gap: 20px;
    align-items: start;
}

/* Board Panel */
.board-panel {
    background: white;
    border-radius: 10px;
    padding: 20px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}

.board-info {
    display: flex;
    justify-content: space-around;
    margin-bottom: 15px;
    padding: 10px;
    background: #f5f5f5;
    border-radius: 5px;
}

.player-indicator {
    padding: 3px 10px;
    border-radius: 3px;
    font-weight: bold;
}

.player-indicator.black {
    background: #333;
    color: white;
}

.player-indicator.white {
    background: #eee;
    color: #333;
    border: 1px solid #999;
}

/* Gomoku Board */
.board {
    display: inline-grid;
    grid-template-columns: repeat(15, 40px);
    grid-template-rows: repeat(15, 40px);
    gap: 0;
    background: #deb887;
    border: 3px solid #8b4513;
    padding: 10px;
    position: relative;
}

.cell {
    width: 40px;
    height: 40px;
    position: relative;
    cursor: pointer;
    background: #deb887;
}

/* Draw grid lines using pseudo-elements */
.cell::before {
    content: '';
    position: absolute;
    top: 50%;
    left: 0;
    right: 0;
    height: 1px;
    background: #8b4513;
    transform: translateY(-50%);
    pointer-events: none;
}

.cell::after {
    content: '';
    position: absolute;
    left: 50%;
    top: 0;
    bottom: 0;
    width: 1px;
    background: #8b4513;
    transform: translateX(-50%);
    pointer-events: none;
}

/* Edge cells - lines stop at center */
.cell.edge-left::before { left: 50%; }
.cell.edge-right::before { right: 50%; left: auto; width: 50%; }
.cell.edge-top::after { top: 50%; }
.cell.edge-bottom::after { bottom: 50%; top: auto; height: 50%; }

.cell:hover:not(.occupied)::before,
.cell:hover:not(.occupied)::after {
    background: #5568d3;
}

.cell.occupied {
    cursor: default;
}

.cell .stone {
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    width: 32px;
    height: 32px;
    border-radius: 50%;
    box-shadow: 2px 2px 4px rgba(0,0,0,0.3);
}

.cell .stone.black {
    background: radial-gradient(circle at 30% 30%, #555, #000);
}

.cell .stone.white {
    background: radial-gradient(circle at 30% 30%, #fff, #ddd);
    border: 1px solid #999;
}

.cell.highlight {
    background: rgba(76, 175, 80, 0.3);
}

.cell .prob-overlay {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 10px;
    font-weight: bold;
    pointer-events: none;
}

/* Board Controls */
.board-controls {
    margin-top: 15px;
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
    align-items: center;
}

.board-controls button {
    padding: 8px 15px;
    background: #667eea;
    color: white;
    border: none;
    border-radius: 5px;
    cursor: pointer;
    font-weight: bold;
    transition: background 0.3s;
}

.board-controls button:hover {
    background: #5568d3;
}

.mode-selector {
    display: flex;
    gap: 10px;
    align-items: center;
    flex-wrap: wrap;
}

.mode-selector label {
    display: flex;
    align-items: center;
    gap: 5px;
}

.mode-selector select {
    padding: 5px;
    border: 2px solid #667eea;
    border-radius: 5px;
}

/* Info Panel */
.info-panel {
    background: white;
    border-radius: 10px;
    padding: 20px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    display: flex;
    flex-direction: column;
    gap: 20px;
}

.info-panel h3 {
    margin-bottom: 10px;
    color: #667eea;
    border-bottom: 2px solid #667eea;
    padding-bottom: 5px;
}

.ai-controls {
    display: flex;
    flex-direction: column;
    gap: 15px;
}

.control-group {
    display: flex;
    flex-direction: column;
    gap: 5px;
}

.control-group label {
    font-weight: bold;
}

.control-group input[type="range"] {
    width: 100%;
}

.control-group select {
    padding: 6px;
    border: 2px solid #667eea;
    border-radius: 5px;
    font-size: 14px;
}

.puct-inputs {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px;
}

.puct-field {
    display: flex;
    flex-direction: column;
    gap: 3px;
    font-size: 13px;
    font-weight: normal;
}

.puct-field input[type="number"] {
    padding: 5px;
    border: 2px solid #667eea;
    border-radius: 5px;
    width: 100%;
}

.primary-btn {
    padding: 10px 15px;
    background: #4caf50;
    color: white;
    border: none;
    border-radius: 5px;
    cursor: pointer;
    font-weight: bold;
    font-size: 14px;
    transition: background 0.3s;
}

.primary-btn:hover {
    background: #45a049;
}

.primary-btn:disabled {
    background: #ccc;
    cursor: not-allowed;
}

/* Value Bar */
.value-display {
    margin-bottom: 10px;
}

#value-bar-container {
    position: relative;
    width: 100%;
    height: 40px;
    background: linear-gradient(to right, #f44336 0%, #fff 50%, #4caf50 100%);
    border-radius: 5px;
    margin-top: 10px;
    border: 2px solid #333;
}

#value-bar {
    position: absolute;
    top: 0;
    left: 50%;
    width: 4px;
    height: 100%;
    background: #000;
    transform: translateX(-50%);
    transition: left 0.3s;
}

#value-text {
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    font-weight: bold;
    font-size: 16px;
    color: #000;
    text-shadow: 0 0 3px #fff, 0 0 3px #fff;
}

/* Best Move */
.best-move {
    padding: 10px;
    background: #f5f5f5;
    border-radius: 5px;
    margin-bottom: 10px;
}

#suggested-move {
    font-size: 18px;
    font-weight: bold;
    color: #667eea;
}

/* Top Moves List */
.top-moves {
    max-height: 300px;
    overflow-y: auto;
}

#top-moves-list {
    display: flex;
    flex-direction: column;
    gap: 5px;
    margin-top: 10px;
}

.move-item {
    display: flex;
    justify-content: space-between;
    padding: 8px;
    background: #f5f5f5;
    border-radius: 5px;
    cursor: pointer;
    transition: background 0.2s;
}

.move-item:hover {
    background: #e0e0e0;
}

.move-item.top-1 {
    background: #fff3cd;
    border: 2px solid #ffc107;
}

/* Move History */
.move-history {
    max-height: 200px;
    overflow-y: auto;
}

#history-list {
    display: flex;
    flex-direction: column;
    gap: 5px;
    margin-top: 10px;
}

.history-item {
    padding: 5px 10px;
    background: #f5f5f5;
    border-radius: 5px;
    font-size: 14px;
}

/* Responsive Design */
@media (max-width: 1200px) {
    .game-container {
        grid-template-columns: 1fr;
    }

    .info-panel {
        order: -1;
    }
}

@media (max-width: 768px) {
    .board {
        grid-template-columns: repeat(15, 30px);
        grid-template-rows: repeat(15, 30px);
    }

    .cell {
        width: 30px;
        height: 30px;
    }

    .cell .stone {
        width: 24px;
        height: 24px;
    }
}
    </style>
</head>
<body>
    <div class="container">
        <h1>Gomoku MCTS Playground</h1>

        <!-- Checkpoint Selection -->
        <div class="control-panel">
            <div class="checkpoint-selector">
                <label for="checkpoint-select">Select Checkpoint:</label>
                <select id="checkpoint-select" onchange="loadCheckpoint()">
                    <option value="">-- Loading checkpoints --</option>
                </select>
                <span id="model-status" class="status">No model loaded</span>
            </div>
        </div>

        <!-- Main Game Area -->
        <div class="game-container">
            <!-- Left Panel: Board -->
            <div class="board-panel">
                <div class="board-info">
                    <div>
                        <strong>Current Player:</strong>
                        <span id="current-player" class="player-indicator black">Black</span>
                    </div>
                    <div>
                        <strong>Game Status:</strong>
                        <span id="game-status">In Progress</span>
                    </div>
                </div>

                <div id="board" class="board"></div>

                <div class="board-controls">
                    <button onclick="clearBoard()">Clear Board</button>
                    <button onclick="undoMove()">Undo Move</button>
                    <div class="mode-selector">
                        <label>
                            <input type="radio" name="mode" value="auto" checked onchange="updateMode()">
                            Auto alternate colors
                        </label>
                        <label>
                            <input type="radio" name="mode" value="manual" onchange="updateMode()">
                            Manual placement:
                        </label>
                        <select id="manual-color" disabled>
                            <option value="black">Black</option>
                            <option value="white">White</option>
                        </select>
                    </div>
                </div>
            </div>

            <!-- Right Panel: AI Controls and Info -->
            <div class="info-panel">
                <div class="ai-controls">
                    <h3>AI Controls</h3>

                    <div class="control-group">
                        <label for="temperature">Temperature (on visit counts):</label>
                        <input type="range" id="temperature" min="0" max="2" step="0.1" value="1.0" oninput="updateTemperatureDisplay()">
                        <span id="temperature-value">1.0</span>
                    </div>

                    <div class="control-group">
                        <label for="mcts-budget">MCTS Budget (simulations):</label>
                        <input type="range" id="mcts-budget" min="16" max="2048" step="16" value="256" oninput="updateBudgetDisplay()">
                        <span id="mcts-budget-value">256</span>
                    </div>

                    <div class="control-group">
                        <label>Search constants:</label>
                        <div class="puct-inputs">
                            <label class="puct-field">c_puct
                                <input type="number" id="c-puct" value="1.25" step="0.05" min="0">
                            </label>
                            <label class="puct-field">Dirichlet &alpha;
                                <input type="number" id="dirichlet-alpha" value="0.125" step="0.005" min="0">
                            </label>
                            <label class="puct-field">Dirichlet &epsilon;
                                <input type="number" id="dirichlet-epsilon" value="0" step="0.05" min="0" max="1">
                            </label>
                            <label class="puct-field">discount &gamma;
                                <input type="number" id="gamma" value="0.984375" step="0.001" min="0" max="1">
                            </label>
                        </div>
                    </div>

                    <div class="control-group">
                        <label for="heatmap-mode">Heatmap:</label>
                        <select id="heatmap-mode" onchange="toggleHeatmap()">
                            <option value="off" selected>Off</option>
                            <option value="visits">Visit counts</option>
                            <option value="diff">MCTS &minus; Raw</option>
                        </select>
                    </div>

                    <button id="ai-suggest-btn" onclick="getAISuggestion()" class="primary-btn">Get AI Suggestion</button>
                    <button onclick="makeAIMove()" class="primary-btn">Make AI Move</button>
                </div>

                <div class="ai-output">
                    <h3>AI Analysis</h3>

                    <div class="value-display">
                        <strong>Position Value (Black's perspective):</strong>
                        <div id="value-bar-container">
                            <div id="value-bar"></div>
                            <div id="value-text">--</div>
                        </div>
                    </div>

                    <div class="entropy-display" style="margin-bottom: 10px;">
                        <strong>Visit-distribution entropy:</strong> <span id="entropy-text">--</span> <span id="entropy-nats" style="color: #888; font-size: 12px;">nats</span>
                    </div>

                    <div class="entropy-display" style="margin-bottom: 10px;">
                        <strong>Raw→MCTS KL:</strong> <span id="raw-mcts-kl-text">--</span> <span style="color: #888; font-size: 12px;">nats (policy-improvement gap)</span>
                    </div>

                    <div class="best-move">
                        <strong>Suggested Move:</strong>
                        <span id="suggested-move">--</span>
                    </div>

                    <div class="top-moves">
                        <strong>Top Moves:</strong>
                        <div id="top-moves-list">
                            <em>Click "Get AI Suggestion" to analyze</em>
                        </div>
                    </div>
                </div>

                <div class="move-history">
                    <h3>Move History</h3>
                    <div id="history-list"></div>
                </div>
            </div>
        </div>
    </div>

    <script>
// Game State
let boardState = {
    black: Array(15).fill(null).map(() => Array(15).fill(0)),
    white: Array(15).fill(null).map(() => Array(15).fill(0)),
    currentPlayer: 'black',
    gameOver: false,
    moveHistory: []
};

let aiData = null; // Store latest AI analysis
let mode = 'auto'; // 'auto' or 'manual'

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    initBoard();
    loadCheckpoints();
    updateTemperatureDisplay();
    updateBudgetDisplay();
});

// ============================================================================
// Board Rendering
// ============================================================================

function initBoard() {
    const board = document.getElementById('board');
    board.innerHTML = '';

    for (let row = 0; row < 15; row++) {
        for (let col = 0; col < 15; col++) {
            const cell = document.createElement('div');
            cell.className = 'cell';
            cell.dataset.row = row;
            cell.dataset.col = col;
            // Add edge classes for grid line rendering
            if (col === 0) cell.classList.add('edge-left');
            if (col === 14) cell.classList.add('edge-right');
            if (row === 0) cell.classList.add('edge-top');
            if (row === 14) cell.classList.add('edge-bottom');
            cell.onclick = () => handleCellClick(row, col);
            board.appendChild(cell);
        }
    }
}

function renderBoard() {
    for (let row = 0; row < 15; row++) {
        for (let col = 0; col < 15; col++) {
            const cell = getCell(row, col);
            // Preserve edge classes when re-rendering
            const edgeClasses = [];
            if (col === 0) edgeClasses.push('edge-left');
            if (col === 14) edgeClasses.push('edge-right');
            if (row === 0) edgeClasses.push('edge-top');
            if (row === 14) edgeClasses.push('edge-bottom');
            cell.className = 'cell ' + edgeClasses.join(' ');
            cell.innerHTML = '';

            // Add stone if occupied
            if (boardState.black[row][col] === 1) {
                cell.classList.add('occupied');
                const stone = document.createElement('div');
                stone.className = 'stone black';
                cell.appendChild(stone);
            } else if (boardState.white[row][col] === 1) {
                cell.classList.add('occupied');
                const stone = document.createElement('div');
                stone.className = 'stone white';
                cell.appendChild(stone);
            }
        }
    }

    updatePlayerDisplay();
    updateHistoryDisplay();
}

function getCell(row, col) {
    return document.querySelector(`.cell[data-row="${row}"][data-col="${col}"]`);
}

function updatePlayerDisplay() {
    const indicator = document.getElementById('current-player');
    indicator.textContent = boardState.currentPlayer.charAt(0).toUpperCase() + boardState.currentPlayer.slice(1);
    indicator.className = `player-indicator ${boardState.currentPlayer}`;
}

function updateHistoryDisplay() {
    const historyList = document.getElementById('history-list');
    historyList.innerHTML = '';

    if (boardState.moveHistory.length === 0) {
        historyList.innerHTML = '<em>No moves yet</em>';
        return;
    }

    boardState.moveHistory.forEach((move, idx) => {
        const item = document.createElement('div');
        item.className = 'history-item';
        item.textContent = `${idx + 1}. ${move.player} at (${move.row}, ${move.col})`;
        historyList.appendChild(item);
    });

    historyList.scrollTop = historyList.scrollHeight;
}

// ============================================================================
// Board Interaction
// ============================================================================

function handleCellClick(row, col) {
    if (boardState.gameOver) {
        alert('Game is over! Clear the board to start a new game.');
        return;
    }

    // Check if cell is already occupied
    if (boardState.black[row][col] === 1 || boardState.white[row][col] === 1) {
        return;
    }

    let playerToPlace;
    if (mode === 'auto') {
        playerToPlace = boardState.currentPlayer;
    } else {
        playerToPlace = document.getElementById('manual-color').value;
    }

    placeStone(row, col, playerToPlace);

    if (mode === 'auto') {
        togglePlayer();
    }

    checkWin(row, col, playerToPlace);
    renderBoard();
}

function placeStone(row, col, player) {
    if (player === 'black') {
        boardState.black[row][col] = 1;
    } else {
        boardState.white[row][col] = 1;
    }

    boardState.moveHistory.push({ row, col, player });
}

function togglePlayer() {
    boardState.currentPlayer = boardState.currentPlayer === 'black' ? 'white' : 'black';
}

function clearBoard() {
    boardState.black = Array(15).fill(null).map(() => Array(15).fill(0));
    boardState.white = Array(15).fill(null).map(() => Array(15).fill(0));
    boardState.currentPlayer = 'black';
    boardState.gameOver = false;
    boardState.moveHistory = [];
    aiData = null;

    document.getElementById('game-status').textContent = 'In Progress';
    document.getElementById('suggested-move').textContent = '--';
    document.getElementById('value-text').textContent = '--';
    document.getElementById('entropy-text').textContent = '--';
    document.getElementById('raw-mcts-kl-text').textContent = '--';
    document.getElementById('top-moves-list').innerHTML = '<em>Click "Get AI Suggestion" to analyze</em>';

    renderBoard();
    clearHeatmap();
}

function undoMove() {
    if (boardState.moveHistory.length === 0) return;

    const lastMove = boardState.moveHistory.pop();
    if (lastMove.player === 'black') {
        boardState.black[lastMove.row][lastMove.col] = 0;
    } else {
        boardState.white[lastMove.row][lastMove.col] = 0;
    }

    if (mode === 'auto') {
        boardState.currentPlayer = lastMove.player;
    }

    boardState.gameOver = false;
    document.getElementById('game-status').textContent = 'In Progress';

    renderBoard();
}

function updateMode() {
    const modeRadios = document.getElementsByName('mode');
    for (const radio of modeRadios) {
        if (radio.checked) {
            mode = radio.value;
        }
    }

    const manualColorSelect = document.getElementById('manual-color');
    manualColorSelect.disabled = mode === 'auto';
}

// ============================================================================
// Win Detection
// ============================================================================

function checkWin(row, col, player) {
    const pieces = player === 'black' ? boardState.black : boardState.white;
    const directions = [
        [0, 1],   // horizontal
        [1, 0],   // vertical
        [1, 1],   // diagonal
        [1, -1]   // anti-diagonal
    ];

    for (const [dr, dc] of directions) {
        let count = 1;

        // Count forward
        let r = row + dr;
        let c = col + dc;
        while (r >= 0 && r < 15 && c >= 0 && c < 15 && pieces[r][c] === 1) {
            count++;
            r += dr;
            c += dc;
        }

        // Count backward
        r = row - dr;
        c = col - dc;
        while (r >= 0 && r < 15 && c >= 0 && c < 15 && pieces[r][c] === 1) {
            count++;
            r -= dr;
            c -= dc;
        }

        if (count >= 5) {
            boardState.gameOver = true;
            document.getElementById('game-status').textContent = `${player.toUpperCase()} WINS!`;
            return true;
        }
    }

    // Check for draw
    let emptyCount = 0;
    for (let i = 0; i < 15; i++) {
        for (let j = 0; j < 15; j++) {
            if (boardState.black[i][j] === 0 && boardState.white[i][j] === 0) {
                emptyCount++;
            }
        }
    }

    if (emptyCount === 0) {
        boardState.gameOver = true;
        document.getElementById('game-status').textContent = 'DRAW';
        return true;
    }

    return false;
}

// ============================================================================
// Checkpoint Management
// ============================================================================

async function loadCheckpoints() {
    try {
        const response = await fetch('/api/checkpoints');
        const data = await response.json();

        const select = document.getElementById('checkpoint-select');
        select.innerHTML = '<option value="">-- Select a checkpoint --</option>';

        data.checkpoints.forEach(cp => {
            const option = document.createElement('option');
            option.value = cp.filename;
            option.textContent = cp.display;
            select.appendChild(option);
        });

        if (data.current) {
            select.value = data.current;
            document.getElementById('model-status').textContent = `Loaded: ${data.current}`;
            document.getElementById('model-status').className = 'status loaded';
        }
    } catch (error) {
        console.error('Error loading checkpoints:', error);
        document.getElementById('model-status').textContent = 'Error loading checkpoints';
        document.getElementById('model-status').className = 'status error';
    }
}

async function loadCheckpoint() {
    const select = document.getElementById('checkpoint-select');
    const filename = select.value;

    if (!filename) {
        return;
    }

    document.getElementById('model-status').textContent = 'Loading...';
    document.getElementById('model-status').className = 'status';

    try {
        const response = await fetch('/api/load_checkpoint', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ filename })
        });

        const data = await response.json();

        if (data.success) {
            document.getElementById('model-status').textContent = `Loaded: ${filename}`;
            document.getElementById('model-status').className = 'status loaded';
        } else {
            document.getElementById('model-status').textContent = `Error: ${data.error}`;
            document.getElementById('model-status').className = 'status error';
        }
    } catch (error) {
        console.error('Error loading checkpoint:', error);
        document.getElementById('model-status').textContent = 'Error loading model';
        document.getElementById('model-status').className = 'status error';
    }
}

// ============================================================================
// AI Inference
// ============================================================================

async function getAISuggestion() {
    const temperature = parseFloat(document.getElementById('temperature').value);
    const numSimulations = parseInt(document.getElementById('mcts-budget').value);
    const cPuct = parseFloat(document.getElementById('c-puct').value);
    const dirichletAlpha = parseFloat(document.getElementById('dirichlet-alpha').value);
    const dirichletEpsilon = parseFloat(document.getElementById('dirichlet-epsilon').value);
    const gamma = parseFloat(document.getElementById('gamma').value);

    try {
        document.getElementById('ai-suggest-btn').disabled = true;
        document.getElementById('ai-suggest-btn').textContent = 'Searching...';

        const response = await fetch('/api/inference', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                black_pieces: boardState.black,
                white_pieces: boardState.white,
                current_player: boardState.currentPlayer,
                temperature: temperature,
                num_simulations: numSimulations,
                c_puct: cPuct,
                dirichlet_alpha: dirichletAlpha,
                dirichlet_epsilon: dirichletEpsilon,
                gamma: gamma
            })
        });

        const data = await response.json();

        if (data.error) {
            alert('Error: ' + data.error);
            return;
        }

        aiData = data;
        displayAIAnalysis(data);

    } catch (error) {
        console.error('Error getting AI suggestion:', error);
        alert('Error communicating with server');
    } finally {
        document.getElementById('ai-suggest-btn').disabled = false;
        document.getElementById('ai-suggest-btn').textContent = 'Get AI Suggestion';
    }
}

function displayAIAnalysis(data) {
    // Display best move
    const [row, col] = data.best_move;
    document.getElementById('suggested-move').textContent = `(${row}, ${col})`;

    // Highlight best move on board
    document.querySelectorAll('.cell').forEach(cell => cell.classList.remove('highlight'));
    const bestCell = getCell(row, col);
    if (bestCell) {
        bestCell.classList.add('highlight');
    }

    // Display value
    const value = data.value;
    document.getElementById('value-text').textContent = value.toFixed(3);

    // Update value bar position (map -1 to 1 -> 0% to 100%)
    const percentage = ((value + 1) / 2) * 100;
    document.getElementById('value-bar').style.left = `${percentage}%`;

    // Display entropy
    document.getElementById('entropy-text').textContent = data.entropy.toFixed(3);

    // Display raw->MCTS KL (policy-improvement gap)
    document.getElementById('raw-mcts-kl-text').textContent = data.raw_mcts_kl.toFixed(3);

    // Display top moves
    const topMovesList = document.getElementById('top-moves-list');
    topMovesList.innerHTML = '';

    const topN = Math.min(10, data.probabilities.length);
    for (let i = 0; i < topN; i++) {
        const move = data.probabilities[i];
        const item = document.createElement('div');
        item.className = i === 0 ? 'move-item top-1' : 'move-item';
        item.innerHTML = `
            <span>(${move.row}, ${move.col})</span>
            <span>${(move.prob * 100).toFixed(2)}% <span style="color: #888; font-size: 12px;">(${move.count} visits)</span></span>
        `;
        item.onclick = () => highlightMove(move.row, move.col);
        topMovesList.appendChild(item);
    }

    // Update heatmap if enabled
    toggleHeatmap();
}

function highlightMove(row, col) {
    document.querySelectorAll('.cell').forEach(cell => cell.classList.remove('highlight'));
    const cell = getCell(row, col);
    if (cell) {
        cell.classList.add('highlight');
    }
}

async function makeAIMove() {
    // Always get fresh AI analysis for current board state
    await getAISuggestion();
    if (!aiData) return;

    const [row, col] = aiData.best_move;

    // Check if move is legal
    if (boardState.black[row][col] === 1 || boardState.white[row][col] === 1) {
        alert('AI suggested move is already occupied. This should not happen!');
        return;
    }

    placeStone(row, col, boardState.currentPlayer);
    togglePlayer();
    checkWin(row, col, boardState.currentPlayer === 'black' ? 'white' : 'black');
    renderBoard();

    // Clear AI data so next click gets fresh analysis
    aiData = null;
    document.querySelectorAll('.cell').forEach(cell => cell.classList.remove('highlight'));
}

// ============================================================================
// Heatmap Visualization
// ============================================================================

function toggleHeatmap() {
    const mode = document.getElementById('heatmap-mode').value;

    if (!aiData || mode === 'off') {
        clearHeatmap();
    } else if (mode === 'visits') {
        renderHeatmap(aiData.all_probs_grid);
    } else {
        renderDiffHeatmap(aiData.all_probs_grid, aiData.raw_prior_grid);
    }
}

function renderHeatmap(probsGrid) {
    clearHeatmap();

    // Find max probability for normalization
    let maxProb = 0;
    for (let row = 0; row < 15; row++) {
        for (let col = 0; col < 15; col++) {
            if (probsGrid[row][col] > maxProb) {
                maxProb = probsGrid[row][col];
            }
        }
    }

    for (let row = 0; row < 15; row++) {
        for (let col = 0; col < 15; col++) {
            const prob = probsGrid[row][col];
            if (prob > 0.001) { // Only show significant probabilities
                const cell = getCell(row, col);
                if (!cell.classList.contains('occupied')) {
                    // Color intensity based on probability
                    const intensity = prob / maxProb;
                    const color = `rgba(255, 0, 0, ${intensity * 0.6})`;
                    cell.style.backgroundColor = color;

                    // Add probability text
                    const overlay = document.createElement('div');
                    overlay.className = 'prob-overlay';
                    overlay.textContent = (prob * 100).toFixed(1) + '%';
                    overlay.style.color = intensity > 0.5 ? 'white' : 'black';
                    cell.appendChild(overlay);
                }
            }
        }
    }
}

// Signed heatmap of (visit distribution - raw prior): where the search moved
// probability mass relative to the network's own policy. Red = MCTS raised the
// move, blue = MCTS demoted it; intensity is normalized by the largest |diff|
// on the board, so the scale is per-position.
function renderDiffHeatmap(probsGrid, rawGrid) {
    clearHeatmap();

    let maxAbsDiff = 0;
    for (let row = 0; row < 15; row++) {
        for (let col = 0; col < 15; col++) {
            const d = Math.abs(probsGrid[row][col] - rawGrid[row][col]);
            if (d > maxAbsDiff) {
                maxAbsDiff = d;
            }
        }
    }
    if (maxAbsDiff === 0) return;

    for (let row = 0; row < 15; row++) {
        for (let col = 0; col < 15; col++) {
            const diff = probsGrid[row][col] - rawGrid[row][col];
            if (Math.abs(diff) > 0.001) { // Only show significant shifts
                const cell = getCell(row, col);
                if (!cell.classList.contains('occupied')) {
                    const intensity = Math.abs(diff) / maxAbsDiff;
                    const rgb = diff > 0 ? '255, 0, 0' : '0, 80, 255';
                    cell.style.backgroundColor = `rgba(${rgb}, ${intensity * 0.6})`;

                    const overlay = document.createElement('div');
                    overlay.className = 'prob-overlay';
                    const pct = (diff * 100).toFixed(1);  // already signed
                    overlay.textContent = (diff > 0 ? '+' : '') + pct;
                    overlay.style.color = intensity > 0.5 ? 'white' : 'black';
                    cell.appendChild(overlay);
                }
            }
        }
    }
}

function clearHeatmap() {
    document.querySelectorAll('.cell').forEach(cell => {
        cell.style.backgroundColor = '';
        const overlay = cell.querySelector('.prob-overlay');
        if (overlay) {
            overlay.remove();
        }
    });
}

// ============================================================================
// UI Controls
// ============================================================================

function updateTemperatureDisplay() {
    const temp = document.getElementById('temperature').value;
    document.getElementById('temperature-value').textContent = temp;
}

function updateBudgetDisplay() {
    const budget = document.getElementById('mcts-budget').value;
    document.getElementById('mcts-budget-value').textContent = budget;
}
    </script>
</body>
</html>
"""


def get_available_checkpoints():
    """Get list of all available checkpoint files (recursively searches all subdirectories)."""
    # Recursively find all .pt files
    checkpoint_files = glob.glob("**/*.pt", recursive=True)

    checkpoints = []
    for filepath in checkpoint_files:
        checkpoints.append({
            'filename': filepath,
            'display': filepath,
            'mtime': os.path.getmtime(filepath)
        })

    # Sort by modification time, newest first
    checkpoints.sort(key=lambda x: x['mtime'], reverse=True)
    return checkpoints


def load_checkpoint(checkpoint_path):
    """Load a model from checkpoint file."""
    global current_model, current_checkpoint_name

    try:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'), weights_only=False)
        model = GomokuPolicyNet()
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(DEVICE)
        model.eval()

        current_model = model
        current_checkpoint_name = os.path.basename(checkpoint_path)
        # The NN-eval cache keys on canonical obs; entries from a previous model
        # would be stale. Clear it whenever the loaded model changes.
        clear_nn_eval_cache()
        return True
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return False


def board_to_observation(black_pieces, white_pieces, current_player):
    """
    Convert board state to model observation.

    Args:
        black_pieces: 15x15 array (0 or 1)
        white_pieces: 15x15 array (0 or 1)
        current_player: 'black' or 'white'

    Returns:
        obs: [3, 15, 15] numpy array in canonical form
    """
    black_np = np.array(black_pieces, dtype=np.uint8)
    white_np = np.array(white_pieces, dtype=np.uint8)

    # Canonical representation: current player is channel 0, opponent is channel 1
    if current_player == 'black':
        c0, c1 = black_np, white_np
    else:
        c0, c1 = white_np, black_np

    obs = encode_observation(c0, c1)
    return obs


def get_legal_mask(black_pieces, white_pieces):
    """Get legal moves mask (empty cells)."""
    black_np = np.array(black_pieces, dtype=np.uint8)
    white_np = np.array(white_pieces, dtype=np.uint8)
    legal_mask = ((black_np == 0) & (white_np == 0)).astype(np.uint8)
    return legal_mask


def run_inference(
    black_pieces,
    white_pieces,
    current_player,
    temperature,
    num_simulations,
    c_puct,
    dirichlet_alpha,
    dirichlet_epsilon,
    gamma,
    fpu_multiplier,
):
    """
    Run an MCTS search on the current board state and return analysis.

    Unlike the raw single-forward frontend, the suggestion here comes from an
    actual AlphaZero PUCT search over the network (entropy_multiplier=None =
    raw masked-softmax priors). The reported move distribution is the search's
    normalized root visit counts — what stage-2 self-play samples from.

    Args:
        black_pieces, white_pieces: 15x15 lists of lists (0/1).
        current_player: 'black' or 'white' (side to move).
        temperature: applied to the visit distribution at move-selection time
            (visits ** (1/T) renormalized; 0 = argmax). The displayed
            distribution itself is the untempered visit distribution.
        num_simulations: MCTS simulation budget.
        c_puct: PUCT exploration constant.
        dirichlet_alpha, dirichlet_epsilon: root Dirichlet noise (epsilon 0 =
            no noise, deterministic search).
        gamma: per-ply backup discount.

    Returns:
        dict with best_move, value (BLACK's perspective), probabilities,
        all_probs_grid, raw_prior_grid (the network's masked-softmax prior at
        the root, pre-Dirichlet), entropy (of the visit distribution),
        raw_mcts_kl (KL(visit_dist || raw prior) = the policy-improvement gap
        MCTS opens over the network's raw prior).

    Note: Caller must ensure current_model is not None before calling.
    """
    obs = board_to_observation(black_pieces, white_pieces, current_player)
    legal_mask = get_legal_mask(black_pieces, white_pieces)

    next_player = Player.BLACK if current_player == 'black' else Player.WHITE
    board = board_from_observation(obs, next_player)

    visit_dists, root_values, _raw_entropies, raw_mcts_kls, _harvested = mcts_search_batched(
        current_model,
        [board],
        num_simulations=num_simulations,
        c_puct=c_puct,
        entropy_multiplier=None,
        device=DEVICE,
        dirichlet_alpha=dirichlet_alpha,
        dirichlet_epsilon=dirichlet_epsilon,
        gamma=gamma,
        fpu_multiplier=fpu_multiplier,
        harvest_min_visits=None,
    )
    visits = visit_dists[0]  # [225], already 0 on illegal squares

    # Select a move from the visit distribution (mirrors self_play.py action
    # sampling): argmax at T=0, else sample from visits ** (1/T) renormalized.
    if temperature == 0:
        best_idx = int(visits.argmax())
    else:
        sample_dist = np.maximum(visits, 1e-30) ** (1.0 / temperature)
        sample_dist = sample_dist / sample_dist.sum()
        best_idx = int(np.random.choice(225, p=sample_dist))
    best_row, best_col = idx_to_pos(best_idx)

    # Entropy of the visit distribution over its support.
    nz = visits[visits > 0]
    entropy = float(-(nz * np.log(nz)).sum()) if nz.size else 0.0

    # root_values[0] is from the side-to-move's perspective; convert to BLACK's.
    value_stm = float(root_values[0])
    value_black = -value_stm if current_player == 'white' else value_stm

    # Per-legal-move visit fractions, sorted high → low. Every simulation
    # contributes exactly one visit to a root child, so sum(child_n) ==
    # num_simulations and the raw count is recovered as prob * num_simulations.
    legal_positions = np.argwhere(legal_mask == 1)
    move_probs = []
    for pos in legal_positions:
        row, col = int(pos[0]), int(pos[1])
        prob = float(visits[row * 15 + col])
        move_probs.append({
            'row': row,
            'col': col,
            'prob': prob,
            'count': round(prob * num_simulations),
        })
    move_probs.sort(key=lambda x: x['prob'], reverse=True)

    probs_grid = visits.reshape(15, 15).tolist()

    # The network's raw prior at the root — the same masked-softmax,
    # pre-Dirichlet distribution the search itself started from. The root obs
    # was evaluated during the search above, so this is a pure cache hit.
    raw_priors, _ = _evaluate_with_cache(current_model, [obs], None, DEVICE)
    raw_grid = raw_priors[0].reshape(15, 15).tolist()

    return {
        'best_move': [int(best_row), int(best_col)],
        'value': float(value_black),
        'probabilities': move_probs,
        'all_probs_grid': probs_grid,
        'raw_prior_grid': raw_grid,
        'entropy': float(entropy),
        'raw_mcts_kl': float(raw_mcts_kls[0]),
    }


# ============================================================================
# Flask Routes
# ============================================================================

@app.route('/')
def index():
    """Serve main page."""
    return HTML_TEMPLATE


@app.route('/api/checkpoints', methods=['GET'])
def api_checkpoints():
    """Get list of available checkpoints."""
    checkpoints = get_available_checkpoints()
    return jsonify({
        'checkpoints': checkpoints,
        'current': current_checkpoint_name
    })


@app.route('/api/load_checkpoint', methods=['POST'])
def api_load_checkpoint():
    """Load a checkpoint."""
    data = request.json
    filename = data.get('filename')

    if not filename:
        return jsonify({'success': False, 'error': 'No filename provided'})

    success = load_checkpoint(filename)
    return jsonify({
        'success': success,
        'checkpoint': filename if success else None
    })


@app.route('/api/inference', methods=['POST'])
def api_inference():
    """Run inference on board state."""
    data = request.json

    black_pieces = data.get('black_pieces')
    white_pieces = data.get('white_pieces')
    current_player = data.get('current_player')
    temperature = data.get('temperature', 1.0)
    num_simulations = int(data.get('num_simulations', DEFAULT_NUM_SIMULATIONS))
    c_puct = float(data.get('c_puct', DEFAULT_C_PUCT))
    dirichlet_alpha = float(data.get('dirichlet_alpha', DEFAULT_DIRICHLET_ALPHA))
    dirichlet_epsilon = float(data.get('dirichlet_epsilon', DEFAULT_DIRICHLET_EPSILON))
    gamma = float(data.get('gamma', DEFAULT_GAMMA))
    fpu_multiplier = float(data.get('fpu_multiplier', DEFAULT_FPU_MULTIPLIER))

    if current_model is None:
        return jsonify({'error': 'No model loaded'})

    result = run_inference(
        black_pieces, white_pieces, current_player, temperature,
        num_simulations, c_puct, dirichlet_alpha, dirichlet_epsilon, gamma,
        fpu_multiplier,
    )

    return jsonify(result)


if __name__ == '__main__':
    print("Starting Gomoku MCTS Web Interface...")
    print("Open http://localhost:5000 in your browser")
    app.run(debug=True, host='0.0.0.0', port=5000)
