"""
Gomoku AI Web Interface - Flask Server (Standalone)

Simple web server for interactive testing of trained Gomoku models.
Allows humans to play against AI and explore model predictions.

All HTML, CSS, and JavaScript are embedded in this file for portability.
"""

import glob
import os

import numpy as np
import torch
import torch.nn.functional as F
from flask import Flask, jsonify, request
from gomoku import encode_observation, idx_to_pos
from model import GomokuPolicyNet

app = Flask(__name__)

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
    <title>Gomoku AI Playground</title>
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
        <h1>Gomoku AI Playground</h1>

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
                        <label for="temperature">Temperature:</label>
                        <input type="range" id="temperature" min="0" max="2" step="0.1" value="1.0" oninput="updateTemperatureDisplay()">
                        <span id="temperature-value">1.0</span>
                    </div>

                    <div class="control-group">
                        <label>
                            <input type="checkbox" id="show-heatmap" onchange="toggleHeatmap()">
                            Show probability heatmap
                        </label>
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

    try {
        document.getElementById('ai-suggest-btn').disabled = true;
        document.getElementById('ai-suggest-btn').textContent = 'Analyzing...';

        const response = await fetch('/api/inference', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                black_pieces: boardState.black,
                white_pieces: boardState.white,
                current_player: boardState.currentPlayer,
                temperature: temperature
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
            <span>${(move.prob * 100).toFixed(2)}%</span>
        `;
        item.onclick = () => highlightMove(move.row, move.col);
        topMovesList.appendChild(item);
    }

    // Update heatmap if enabled
    if (document.getElementById('show-heatmap').checked) {
        renderHeatmap(data.all_probs_grid);
    }
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
    const showHeatmap = document.getElementById('show-heatmap').checked;

    if (showHeatmap && aiData) {
        renderHeatmap(aiData.all_probs_grid);
    } else {
        clearHeatmap();
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
        model.eval()

        current_model = model
        current_checkpoint_name = os.path.basename(checkpoint_path)
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


def run_inference(black_pieces, white_pieces, current_player, temperature):
    """
    Run model inference on current board state.

    Args:
        black_pieces: 15x15 list of lists
        white_pieces: 15x15 list of lists
        current_player: 'black' or 'white'
        temperature: float for probability scaling

    Returns:
        dict with:
            - best_move: (row, col)
            - value: float (from BLACK's perspective)
            - probabilities: list of {row, col, prob} for all legal moves
            - all_probs_grid: 15x15 grid of probabilities (0 for illegal)

    Note: Caller must ensure current_model is not None before calling.
    """
    obs = board_to_observation(black_pieces, white_pieces, current_player)
    legal_mask = get_legal_mask(black_pieces, white_pieces)

    # Convert to tensors
    obs_tensor = torch.from_numpy(obs).float().unsqueeze(0)  # [1, 3, 15, 15]
    legal_mask_tensor = torch.from_numpy(legal_mask).bool().unsqueeze(0)  # [1, 15, 15]

    with torch.no_grad():
        logits_grid, value = current_model(obs_tensor)
        logits = logits_grid.squeeze(0).squeeze(0)  # [15, 15]
        value = value.squeeze(0).squeeze(0).item()  # scalar

    # Mask illegal moves
    logits_masked = logits.clone()
    logits_masked[~legal_mask_tensor.squeeze(0)] = -1e9

    # Apply temperature and select move
    if temperature == 0:
        # Deterministic: pick highest probability move (argmax)
        best_idx = logits_masked.view(-1).argmax().item()
        best_row, best_col = idx_to_pos(best_idx)
        # For display purposes, still compute softmax probabilities
        probs = F.softmax(logits_masked.view(-1), dim=0).view(15, 15)
    else:
        # Stochastic: sample from temperature-scaled distribution
        logits_scaled = logits_masked / temperature
        probs = F.softmax(logits_scaled.view(-1), dim=0).view(15, 15)
        best_idx = torch.multinomial(probs.view(-1), num_samples=1).item()
        best_row, best_col = idx_to_pos(best_idx)

    # Convert value to BLACK's perspective
    # value is from current player's perspective
    if current_player == 'white':
        value_black = -value
    else:
        value_black = value

    # Collect probabilities for all legal moves
    legal_positions = np.argwhere(legal_mask == 1)
    move_probs = []
    for pos in legal_positions:
        row, col = int(pos[0]), int(pos[1])
        prob = probs[row, col].item()
        move_probs.append({
            'row': row,
            'col': col,
            'prob': prob
        })

    # Sort by probability (highest first)
    move_probs.sort(key=lambda x: x['prob'], reverse=True)

    # Create full probability grid (with 0 for illegal moves)
    probs_grid = probs.numpy().tolist()

    return {
        'best_move': [int(best_row), int(best_col)],
        'value': float(value_black),
        'probabilities': move_probs,
        'all_probs_grid': probs_grid
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

    if current_model is None:
        return jsonify({'error': 'No model loaded'})

    result = run_inference(black_pieces, white_pieces, current_player, temperature)

    return jsonify(result)


if __name__ == '__main__':
    print("Starting Gomoku AI Web Interface...")
    print("Open http://localhost:5000 in your browser")
    app.run(debug=True, host='0.0.0.0', port=5000)
