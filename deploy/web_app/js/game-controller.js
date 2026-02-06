/**
 * Main Game Controller
 *
 * Orchestrates the UI, game flow, and AI interaction.
 */

// ============================================================================
// Global State
// ============================================================================

const gameState = {
    board: null,
    aiPlayer: null,
    modelManager: null,
    playerColor: Player.BLACK, // User's color
    aiColor: Player.WHITE,
    history: [],  // History of moves for undo
    isAIThinking: false,
    pendingMove: null,  // {row, col} for move confirmation
    lastAIThinkTime: 0  // Last AI thinking time in seconds
};

// ============================================================================
// DOM Elements
// ============================================================================

const loadingScreen = document.getElementById('loading-screen');
const setupPanel = document.getElementById('setup-panel');
const gamePanel = document.getElementById('game-panel');
const resultModal = document.getElementById('result-modal');

const canvas = document.getElementById('game-board');
const ctx = canvas.getContext('2d');

// ============================================================================
// Initialization
// ============================================================================

/**
 * Initialize the game on page load.
 */
async function init() {
    console.log('Initializing Gomoku game...');

    // Show setup panel
    setupPanel.style.display = 'block';

    // Initialize model manager
    gameState.modelManager = new ModelManager();
    await gameState.modelManager.initialize();

    // Set up event listeners
    setupEventListeners();

    console.log('Initialization complete');
}

/**
 * Set up all event listeners.
 */
function setupEventListeners() {
    // Color selection
    document.querySelectorAll('.btn-color').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.btn-color').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
        });
    });

    // Difficulty selection
    document.querySelectorAll('.btn-difficulty').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.btn-difficulty').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
        });
    });

    // Start game button
    document.getElementById('btn-start').addEventListener('click', startGame);

    // Game controls
    document.getElementById('btn-undo').addEventListener('click', undoMove);
    document.getElementById('btn-restart').addEventListener('click', restartGame);

    // Board click
    canvas.addEventListener('click', handleBoardClick);

    // Move confirmation
    document.getElementById('btn-confirm-move').addEventListener('click', confirmMove);
    document.getElementById('btn-cancel-move').addEventListener('click', cancelMove);

    // Result modal
    document.getElementById('btn-play-again').addEventListener('click', playAgain);
    document.getElementById('btn-new-setup').addEventListener('click', newSetup);

    // Window resize
    window.addEventListener('resize', updateCanvasSize);
}

// ============================================================================
// Game Flow
// ============================================================================

/**
 * Start a new game with selected settings.
 */
async function startGame() {
    console.log('Starting game...');

    // Get selected color
    const colorBtn = document.querySelector('.btn-color.active');
    gameState.playerColor = parseInt(colorBtn.dataset.color);
    gameState.aiColor = (gameState.playerColor === Player.BLACK) ? Player.WHITE : Player.BLACK;

    // Get selected difficulty
    const difficultyBtn = document.querySelector('.btn-difficulty.active');
    const difficulty = difficultyBtn.dataset.difficulty;
    gameState.modelManager.setSelectedModel(difficulty);

    console.log(`  Player color: ${gameState.playerColor === Player.BLACK ? 'Black' : 'White'}`);
    console.log(`  AI color: ${gameState.aiColor === Player.BLACK ? 'Black' : 'White'}`);
    console.log(`  Difficulty: ${difficulty}`);

    // Show loading screen
    setupPanel.style.display = 'none';
    loadingScreen.style.display = 'block';

    // Start orbital animation
    startOrbitAnimation();

    try {
        // Load model
        gameState.aiPlayer = await gameState.modelManager.loadSelectedModel();

        // Stop orbital animation
        stopOrbitAnimation();

        // Initialize board
        gameState.board = new GomokuBoard();
        gameState.history = [];
        gameState.isAIThinking = false;
        gameState.pendingMove = null;

        // Show game panel
        loadingScreen.style.display = 'none';
        gamePanel.style.display = 'block';

        // Initialize canvas
        updateCanvasSize();
        drawBoard();

        // If AI plays first (black), make AI move
        if (gameState.aiColor === Player.BLACK) {
            await makeAIMove();
        } else {
            updateStatus('你的回合');
        }

    } catch (error) {
        console.error('Failed to start game:', error);
        alert('加载模型失败,请刷新页面重试。');
        loadingScreen.style.display = 'none';
        setupPanel.style.display = 'block';
    }
}

/**
 * Restart game with same settings.
 */
async function restartGame() {
    console.log('Restarting game...');

    // Show setup panel
    gamePanel.style.display = 'none';
    setupPanel.style.display = 'block';
}

/**
 * Play again (after game ends).
 */
function playAgain() {
    resultModal.style.display = 'none';

    // Reset board
    gameState.board = new GomokuBoard();
    gameState.history = [];
    gameState.isAIThinking = false;
    gameState.pendingMove = null;

    drawBoard();

    // If AI plays first, make AI move
    if (gameState.aiColor === Player.BLACK) {
        makeAIMove();
    } else {
        updateStatus('你的回合');
    }
}

/**
 * New setup (return to setup panel).
 */
function newSetup() {
    resultModal.style.display = 'none';
    gamePanel.style.display = 'none';
    setupPanel.style.display = 'block';
}

// ============================================================================
// Move Handling
// ============================================================================

/**
 * Handle board click.
 */
function handleBoardClick(event) {
    if (gameState.isAIThinking) return;
    if (gameState.pendingMove) return; // Already have a pending move
    if (gameState.board.whoToPlay !== gameState.playerColor) return;

    const rect = canvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;

    // Convert to board coordinates
    const pos = screenToBoard(x, y);
    if (!pos) return;

    const [row, col] = pos;

    // Show pending move
    gameState.pendingMove = { row, col };
    drawBoard();

    // Show confirmation buttons
    document.getElementById('move-confirm').classList.add('visible');
}

/**
 * Confirm pending move.
 */
async function confirmMove() {
    if (!gameState.pendingMove) return;

    const { row, col } = gameState.pendingMove;

    // Hide confirmation buttons
    document.getElementById('move-confirm').classList.remove('visible');

    // Make move
    await makePlayerMove(row, col);

    gameState.pendingMove = null;
}

/**
 * Cancel pending move.
 */
function cancelMove() {
    gameState.pendingMove = null;
    document.getElementById('move-confirm').classList.remove('visible');
    drawBoard();
}

/**
 * Make a player move.
 */
async function makePlayerMove(row, col) {
    console.log(`Player move: (${row}, ${col})`);

    // Save to history
    gameState.history.push({
        row, col,
        player: gameState.playerColor,
        blackPieces: gameState.board.blackPieces.map(r => [...r]),
        whitePieces: gameState.board.whitePieces.map(r => [...r]),
        whoToPlay: gameState.board.whoToPlay,
        occupiedCount: gameState.board.occupiedCount
    });

    // Make move
    const result = gameState.board.Move(row, col);

    drawBoard();

    // Check game result
    if (result !== GameState.CONTINUE) {
        handleGameEnd(result);
        return;
    }

    // AI's turn
    await makeAIMove();
}

/**
 * Make an AI move.
 */
async function makeAIMove() {
    gameState.isAIThinking = true;
    updateStatus('AI思考中...');

    try {
        // Get AI move (samples from probability distribution, temperature is baked into model)
        const startTime = performance.now();
        const [aiRow, aiCol] = await gameState.aiPlayer.getMove(gameState.board);
        const endTime = performance.now();

        gameState.lastAIThinkTime = (endTime - startTime) / 1000; // Convert to seconds

        console.log(`AI move: (${aiRow}, ${aiCol}), think time: ${gameState.lastAIThinkTime.toFixed(2)}s`);

        // Save to history
        gameState.history.push({
            row: aiRow, col: aiCol,
            player: gameState.aiColor,
            blackPieces: gameState.board.blackPieces.map(r => [...r]),
            whitePieces: gameState.board.whitePieces.map(r => [...r]),
            whoToPlay: gameState.board.whoToPlay,
            occupiedCount: gameState.board.occupiedCount
        });

        // Make move
        const result = gameState.board.Move(aiRow, aiCol);

        drawBoard();

        gameState.isAIThinking = false;

        // Check game result
        if (result !== GameState.CONTINUE) {
            handleGameEnd(result);
            return;
        }

        // Player's turn
        updateStatus(`你的回合 (AI思考: ${gameState.lastAIThinkTime.toFixed(2)}秒)`);

    } catch (error) {
        console.error('AI move failed:', error);
        gameState.isAIThinking = false;
        updateStatus('AI出错,请重新开始');
    }
}

/**
 * Undo last move (player + AI).
 */
function undoMove() {
    if (gameState.history.length === 0) return;
    if (gameState.isAIThinking) return;
    if (gameState.pendingMove) return;

    // Pop the last 2 moves (AI + player), or as many as available
    // Since we save state BEFORE making a move, we need to pop twice
    const movesToUndo = Math.min(2, gameState.history.length);
    for (let i = 0; i < movesToUndo; i++) {
        gameState.history.pop();
    }

    // Restore board state
    if (gameState.history.length > 0) {
        // Restore to the last saved state
        const lastState = gameState.history[gameState.history.length - 1];
        gameState.board.blackPieces = lastState.blackPieces.map(r => [...r]);
        gameState.board.whitePieces = lastState.whitePieces.map(r => [...r]);
        gameState.board.whoToPlay = lastState.whoToPlay;
        gameState.board.occupiedCount = lastState.occupiedCount;
    } else {
        // No history left, reset to initial state
        gameState.board = new GomokuBoard();
    }

    drawBoard();
    updateStatus('你的回合');
}

/**
 * Handle game end.
 */
function handleGameEnd(result) {
    console.log(`Game ended: ${result}`);

    let title = '游戏结束';
    let message = '';

    if (result === GameState.BLACK_WIN) {
        title = gameState.playerColor === Player.BLACK ? '你赢了！' : '你输了！';
        message = '⚫ 黑棋获胜';
    } else if (result === GameState.WHITE_WIN) {
        title = gameState.playerColor === Player.WHITE ? '你赢了！' : '你输了！';
        message = '⚪ 白棋获胜';
    } else if (result === GameState.DRAW) {
        title = '平局';
        message = '棋盘已满,没有获胜者';
    }

    document.getElementById('result-title').textContent = title;
    document.getElementById('result-message').textContent = message;
    resultModal.style.display = 'flex';
}

// ============================================================================
// Drawing
// ============================================================================

/**
 * Update canvas size (responsive).
 */
function updateCanvasSize() {
    const container = document.querySelector('.board-container');
    const containerWidth = container.clientWidth;

    // Limit to 600px max
    const maxSize = Math.min(containerWidth, 600);

    // Apply device pixel ratio for high-DPI displays
    const dpr = window.devicePixelRatio || 1;

    // Set CSS size (display size)
    canvas.style.width = maxSize + 'px';
    canvas.style.height = maxSize + 'px';

    // Set internal resolution (physical pixels)
    canvas.width = maxSize * dpr;
    canvas.height = maxSize * dpr;

    // Scale context to match DPR
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.scale(dpr, dpr);

    drawBoard();
}

/**
 * Draw the game board.
 */
function drawBoard() {
    const rect = canvas.getBoundingClientRect();
    const size = rect.width;
    const padding = size * 0.05;
    const gridSize = (size - 2 * padding) / 14;

    // Clear canvas
    ctx.fillStyle = '#F8F8F8';
    ctx.fillRect(0, 0, size, size);

    // Draw grid
    ctx.strokeStyle = '#808080';
    ctx.lineWidth = 1;

    for (let i = 0; i < 15; i++) {
        // Vertical lines
        ctx.beginPath();
        ctx.moveTo(padding + i * gridSize, padding);
        ctx.lineTo(padding + i * gridSize, padding + 14 * gridSize);
        ctx.stroke();

        // Horizontal lines
        ctx.beginPath();
        ctx.moveTo(padding, padding + i * gridSize);
        ctx.lineTo(padding + 14 * gridSize, padding + i * gridSize);
        ctx.stroke();
    }

    // Draw star points
    ctx.fillStyle = '#404040';
    const starPoints = [
        [3, 3], [3, 11], [11, 3], [11, 11], [7, 7]
    ];

    for (const [row, col] of starPoints) {
        const x = padding + col * gridSize;
        const y = padding + row * gridSize;
        ctx.beginPath();
        ctx.arc(x, y, size * 0.008, 0, 2 * Math.PI);
        ctx.fill();
    }

    // Draw pieces
    const pieceRadius = gridSize * 0.4;

    for (let row = 0; row < 15; row++) {
        for (let col = 0; col < 15; col++) {
            const x = padding + col * gridSize;
            const y = padding + row * gridSize;

            if (gameState.board.blackPieces[row][col] === 1) {
                // Black piece
                ctx.fillStyle = '#000';
                ctx.beginPath();
                ctx.arc(x, y, pieceRadius, 0, 2 * Math.PI);
                ctx.fill();
            } else if (gameState.board.whitePieces[row][col] === 1) {
                // White piece
                ctx.fillStyle = '#FFF';
                ctx.strokeStyle = '#000';
                ctx.lineWidth = 1.5;
                ctx.beginPath();
                ctx.arc(x, y, pieceRadius, 0, 2 * Math.PI);
                ctx.fill();
                ctx.stroke();
            }
        }
    }

    // Draw pending move (semi-transparent)
    if (gameState.pendingMove) {
        const { row, col } = gameState.pendingMove;
        const x = padding + col * gridSize;
        const y = padding + row * gridSize;

        const color = gameState.playerColor === Player.BLACK ? '#000' : '#FFF';
        ctx.fillStyle = color;
        ctx.globalAlpha = 0.5;
        ctx.beginPath();
        ctx.arc(x, y, pieceRadius, 0, 2 * Math.PI);
        ctx.fill();

        if (gameState.playerColor === Player.WHITE) {
            ctx.globalAlpha = 1.0;
            ctx.strokeStyle = '#000';
            ctx.lineWidth = 1.5;
            ctx.stroke();
        }

        ctx.globalAlpha = 1.0;
    }
}

/**
 * Convert screen coordinates to board position.
 */
function screenToBoard(x, y) {
    const rect = canvas.getBoundingClientRect();
    const size = rect.width;
    const padding = size * 0.05;
    const gridSize = (size - 2 * padding) / 14;

    const col = Math.round((x - padding) / gridSize);
    const row = Math.round((y - padding) / gridSize);

    if (row < 0 || row >= 15 || col < 0 || col >= 15) {
        return null;
    }

    return [row, col];
}

/**
 * Update status text.
 */
function updateStatus(text) {
    document.getElementById('status-text').textContent = text;
}

// ============================================================================
// Orbital Loading Animation
// ============================================================================

let orbitAnimationRunning = false;
let orbitAnimationFrame = null;
let orbitLines = [];

/**
 * Start the orbital animation.
 */
function startOrbitAnimation() {
    orbitAnimationRunning = true;

    // Clean up any existing lines
    cleanupOrbitLines();

    const svg = document.querySelector('.loading-orbit svg');
    const orbitLinesContainer = document.getElementById('orbit-lines');

    // Get planet elements
    const planet1 = document.querySelector('.planet-1');
    const planet2 = document.querySelector('.planet-2');
    const planet3 = document.querySelector('.planet-3');

    // Create lines (3 lines connecting the planets in a triangle)
    for (let i = 0; i < 3; i++) {
        const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
        line.setAttribute('stroke', '#999');
        line.setAttribute('stroke-width', '1');
        orbitLinesContainer.appendChild(line);
        orbitLines.push(line);
    }

    // Animation function
    function animate() {
        if (!orbitAnimationRunning) return;

        // Get planet positions
        const pos1 = getPlanetPosition(planet1);
        const pos2 = getPlanetPosition(planet2);
        const pos3 = getPlanetPosition(planet3);

        // Update lines
        // Line 0: planet1 -> planet2
        orbitLines[0].setAttribute('x1', pos1.x);
        orbitLines[0].setAttribute('y1', pos1.y);
        orbitLines[0].setAttribute('x2', pos2.x);
        orbitLines[0].setAttribute('y2', pos2.y);

        // Line 1: planet2 -> planet3
        orbitLines[1].setAttribute('x1', pos2.x);
        orbitLines[1].setAttribute('y1', pos2.y);
        orbitLines[1].setAttribute('x2', pos3.x);
        orbitLines[1].setAttribute('y2', pos3.y);

        // Line 2: planet3 -> planet1
        orbitLines[2].setAttribute('x1', pos3.x);
        orbitLines[2].setAttribute('y1', pos3.y);
        orbitLines[2].setAttribute('x2', pos1.x);
        orbitLines[2].setAttribute('y2', pos1.y);

        orbitAnimationFrame = requestAnimationFrame(animate);
    }

    animate();
}

/**
 * Stop the orbital animation.
 */
function stopOrbitAnimation() {
    orbitAnimationRunning = false;
    if (orbitAnimationFrame) {
        cancelAnimationFrame(orbitAnimationFrame);
        orbitAnimationFrame = null;
    }
    cleanupOrbitLines();
}

/**
 * Clean up orbit lines from DOM.
 */
function cleanupOrbitLines() {
    orbitLines.forEach(line => line.parentNode.removeChild(line));
    orbitLines = [];
}

/**
 * Get planet position in SVG coordinates.
 */
function getPlanetPosition(planet) {
    const svg = planet.ownerSVGElement;
    const ctm = planet.getCTM();

    return {
        x: ctm.e,
        y: ctm.f
    };
}

// ============================================================================
// Start
// ============================================================================

init();
