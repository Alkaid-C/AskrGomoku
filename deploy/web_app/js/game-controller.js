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
    lastAIThinkTime: 0, // Last AI thinking time in seconds
    undoCount: 0,        // Number of undos in this game
    gameOver: false      // Whether the game has ended
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
function init() {
    console.log('Initializing Gomoku game...');

    // Show setup panel
    setupPanel.style.display = 'block';

    // Initialize model manager
    gameState.modelManager = new ModelManager();
    gameState.modelManager.initialize();

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

    // Result modal (legacy, kept for standalone build compatibility)
    document.getElementById('btn-play-again').addEventListener('click', playAgain);
    document.getElementById('btn-new-setup').addEventListener('click', newSetup);
    document.getElementById('btn-share').addEventListener('click', showRecordScreen);
    document.getElementById('btn-record-close').addEventListener('click', hideRecordScreen);

    // End actions (in-place button after game ends)
    document.getElementById('btn-end-new-game').addEventListener('click', newSetup);

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
        gameState.undoCount = 0;
        gameState.gameOver = false;

        // Reset end mode UI in case previous game ended
        resetEndModeUI();

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
function restartGame() {
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
    resetEndModeUI();

    // Reset board
    gameState.board = new GomokuBoard();
    gameState.history = [];
    gameState.isAIThinking = false;
    gameState.pendingMove = null;
    gameState.undoCount = 0;
    gameState.gameOver = false;

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
    resetEndModeUI();
    gameState.gameOver = false;
    gamePanel.style.display = 'none';
    setupPanel.style.display = 'block';
}

/**
 * Reset end mode UI back to normal game state.
 */
function resetEndModeUI() {
    // Restore top-controls: show buttons, hide stats
    document.getElementById('btn-undo').style.display = '';
    document.getElementById('btn-restart').style.display = '';
    document.getElementById('end-stat-left').style.display = 'none';
    document.getElementById('end-stat-right').style.display = 'none';

    // Restore bottom area
    document.getElementById('end-actions').style.display = 'none';
    document.getElementById('move-confirm').style.display = '';

    canvas.style.cursor = 'pointer';
}

// ============================================================================
// Move Handling
// ============================================================================

/**
 * Handle board click.
 */
function handleBoardClick(event) {
    if (gameState.gameOver) return;
    if (gameState.isAIThinking) return;
    if (gameState.board.whoToPlay !== gameState.playerColor) return;

    const rect = canvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;

    // Convert to board coordinates
    const pos = screenToBoard(x, y);
    if (!pos) return;

    const [row, col] = pos;

    // Check if position is occupied
    if (gameState.board.blackPieces[row][col] === 1 || gameState.board.whitePieces[row][col] === 1) {
        return;
    }

    // Update pending move (allows changing position without canceling first)
    gameState.pendingMove = { row, col };
    drawBoard();

    // Show confirmation buttons
    document.getElementById('move-confirm').classList.add('visible');
}

/**
 * Confirm pending move.
 */
async function confirmMove(event) {
    // Prevent iOS Safari scroll jump when button disappears
    if (event) {
        event.preventDefault();
    }

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
function cancelMove(event) {
    // Prevent iOS Safari scroll jump when button disappears
    if (event) {
        event.preventDefault();
    }

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

    // Check if using advanced difficulty (negamax search)
    const useNegamax = gameState.modelManager.selectedModel === 'advanced';

    if (useNegamax) {
        updateStatus('深度思考中...');
    } else {
        updateStatus('AI思考中...');
    }

    try {
        // Give browser a chance to update UI before heavy computation
        await new Promise(resolve => setTimeout(resolve, 50));

        // Get AI move
        const startTime = performance.now();
        let aiRow, aiCol, aiValue;

        if (useNegamax) {
            // Advanced difficulty: use negamax search (depth=3, topK=3)
            [aiRow, aiCol, aiValue] = await gameState.aiPlayer.getMoveWithNegamax(gameState.board, 3, 3);
        } else {
            // Junior/Intermediate: sample from policy distribution
            [aiRow, aiCol, aiValue] = await gameState.aiPlayer.getMove(gameState.board);
        }

        const endTime = performance.now();

        gameState.lastAIThinkTime = (endTime - startTime) / 1000; // Convert to seconds

        console.log(`AI move: (${aiRow}, ${aiCol}), think time: ${gameState.lastAIThinkTime.toFixed(2)}s`);

        // Save to history (value is from AI's perspective before this move)
        gameState.history.push({
            row: aiRow, col: aiCol,
            player: gameState.aiColor,
            value: aiValue,
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
        updateStatus('你的回合');

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
    if (!gameState.board) return;
    if (gameState.gameOver) return;
    if (gameState.history.length === 0) return;
    if (gameState.isAIThinking) return;
    if (gameState.pendingMove) return;

    gameState.undoCount++;

    // Pop the last 2 moves (AI + player), or as many as available.
    // History stores snapshots BEFORE each move, so we must restore the
    // oldest popped snapshot to roll back exactly those moves.
    const movesToUndo = Math.min(2, gameState.history.length);
    let restoreState = null;
    for (let i = 0; i < movesToUndo; i++) {
        restoreState = gameState.history.pop();
    }

    // Restore board state from the oldest popped snapshot.
    if (restoreState) {
        gameState.board.blackPieces = restoreState.blackPieces.map(r => [...r]);
        gameState.board.whitePieces = restoreState.whitePieces.map(r => [...r]);
        gameState.board.whoToPlay = restoreState.whoToPlay;
        gameState.board.occupiedCount = restoreState.occupiedCount;
    } else {
        // No snapshot available, reset to initial state.
        gameState.board = new GomokuBoard();
    }

    drawBoard();
    if (gameState.board.whoToPlay === gameState.playerColor) {
        updateStatus('你的回合');
    } else {
        updateStatus('AI思考中...');
        makeAIMove();
    }
}

/**
 * Handle game end.
 */
function handleGameEnd(result) {
    console.log(`Game ended: ${result}`);
    gameState.gameOver = true;

    // Determine result title
    let title = '游戏结束';
    if (result === GameState.BLACK_WIN) {
        title = gameState.playerColor === Player.BLACK ? '你赢了！' : '你输了！';
    } else if (result === GameState.WHITE_WIN) {
        title = gameState.playerColor === Player.WHITE ? '你赢了！' : '你输了！';
    } else if (result === GameState.DRAW) {
        title = '平局';
    }

    // Player info
    const aiName = AI_DISPLAY_NAMES[gameState.modelManager.selectedModel] || 'AI';
    const fullAiName = 'Askr-' + aiName;
    const blackDot = '<span class="record-piece record-piece-black"></span>';
    const whiteDot = '<span class="record-piece record-piece-white"></span>';
    const blackLabel = gameState.playerColor === Player.BLACK ? '玩家' : fullAiName;
    const whiteLabel = gameState.playerColor === Player.WHITE ? '玩家' : fullAiName;

    // Stats
    const gameLength = gameState.history.length;
    const userWon = (result === GameState.BLACK_WIN && gameState.playerColor === Player.BLACK) ||
                    (result === GameState.WHITE_WIN && gameState.playerColor === Player.WHITE);
    let rightHtml = gameLength + '手';
    if (gameState.undoCount > 0 && userWon) {
        rightHtml += '<br>悔棋' + gameState.undoCount + '次';
    }

    // Replace top-controls content in-place: hide buttons, show stats
    document.getElementById('btn-undo').style.display = 'none';
    document.getElementById('btn-restart').style.display = 'none';
    const statLeft = document.getElementById('end-stat-left');
    const statRight = document.getElementById('end-stat-right');
    statLeft.innerHTML = blackDot + ' ' + blackLabel + '<br>' + whiteDot + ' ' + whiteLabel;
    statRight.innerHTML = rightHtml;
    statLeft.style.display = 'block';
    statRight.style.display = 'block';
    updateStatus(title);

    // Switch bottom area: hide move-confirm, show end actions
    document.getElementById('move-confirm').style.display = 'none';
    document.getElementById('end-actions').style.display = 'flex';

    // Disable board interaction
    canvas.style.cursor = 'default';

    // Draw record-style board with move numbers and winning line
    drawBoardRecord();
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

    if (gameState.gameOver) {
        drawBoardRecord();
    } else {
        drawBoard();
    }
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

    // Draw last move marker - only for AI moves
    if (gameState.history.length > 0) {
        const lastMove = gameState.history[gameState.history.length - 1];
        if (lastMove.player === gameState.aiColor) {
            const x = padding + lastMove.col * gridSize;
            const y = padding + lastMove.row * gridSize;

            if (lastMove.player === Player.BLACK) {
                // White cross on black piece
                const halfLen = pieceRadius * 0.375;
                ctx.strokeStyle = '#FFF';
                ctx.lineWidth = 1.5;
                ctx.beginPath();
                ctx.moveTo(x - halfLen, y);
                ctx.lineTo(x + halfLen, y);
                ctx.moveTo(x, y - halfLen);
                ctx.lineTo(x, y + halfLen);
                ctx.stroke();
            } else {
                // Black dot on white piece
                const dotRadius = pieceRadius * 0.2;
                ctx.fillStyle = '#000';
                ctx.beginPath();
                ctx.arc(x, y, dotRadius, 0, 2 * Math.PI);
                ctx.fill();
            }
        }
    }
}

/**
 * Draw the game board in record mode (with move numbers and winning line).
 */
function drawBoardRecord() {
    const rect = canvas.getBoundingClientRect();
    const size = rect.width;
    const padding = size * 0.05;
    const gridSize = (size - 2 * padding) / 14;
    const pieceRadius = gridSize * 0.4;

    // Clear canvas
    ctx.fillStyle = '#F8F8F8';
    ctx.fillRect(0, 0, size, size);

    // Draw grid
    ctx.strokeStyle = '#808080';
    ctx.lineWidth = 1;
    for (let i = 0; i < 15; i++) {
        ctx.beginPath();
        ctx.moveTo(padding + i * gridSize, padding);
        ctx.lineTo(padding + i * gridSize, padding + 14 * gridSize);
        ctx.stroke();

        ctx.beginPath();
        ctx.moveTo(padding, padding + i * gridSize);
        ctx.lineTo(padding + 14 * gridSize, padding + i * gridSize);
        ctx.stroke();
    }

    // Star points
    ctx.fillStyle = '#404040';
    const starPoints = [[3,3],[3,11],[11,3],[11,11],[7,7]];
    for (const [row, col] of starPoints) {
        ctx.beginPath();
        ctx.arc(padding + col * gridSize, padding + row * gridSize, size * 0.008, 0, 2 * Math.PI);
        ctx.fill();
    }

    // Winning line (drawn beneath pieces)
    const winLine = findWinningLine();
    if (winLine) {
        const first = winLine[0];
        const last = winLine[winLine.length - 1];
        ctx.strokeStyle = '#CC0000';
        ctx.lineWidth = size < 500 ? 3 : 5;
        ctx.lineCap = 'round';
        ctx.beginPath();
        ctx.moveTo(padding + first[1] * gridSize, padding + first[0] * gridSize);
        ctx.lineTo(padding + last[1] * gridSize, padding + last[0] * gridSize);
        ctx.stroke();
    }

    // Pieces with move numbers
    for (let i = 0; i < gameState.history.length; i++) {
        const move = gameState.history[i];
        const moveNum = i + 1;
        const x = padding + move.col * gridSize;
        const y = padding + move.row * gridSize;
        const isBlack = move.player === Player.BLACK;

        if (isBlack) {
            ctx.fillStyle = '#000';
            ctx.beginPath();
            ctx.arc(x, y, pieceRadius, 0, 2 * Math.PI);
            ctx.fill();
        } else {
            ctx.fillStyle = '#FFF';
            ctx.strokeStyle = '#000';
            ctx.lineWidth = 1.5;
            ctx.beginPath();
            ctx.arc(x, y, pieceRadius, 0, 2 * Math.PI);
            ctx.fill();
            ctx.stroke();
        }

        // Move number
        ctx.fillStyle = isBlack ? '#FFF' : '#000';
        const digits = String(moveNum).length;
        const fontSize = digits <= 1 ? pieceRadius * 1.2 : digits <= 2 ? pieceRadius * 1.0 : pieceRadius * 0.8;
        ctx.font = `bold ${fontSize}px -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif`;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(String(moveNum), x, y);
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
    cancelAnimationFrame(orbitAnimationFrame);
    orbitAnimationFrame = null;
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
    const ctm = planet.getCTM();
    return { x: ctm.e, y: ctm.f };
}

// ============================================================================
// Record Screen (numbered board for screenshots)
// ============================================================================

const AI_DISPLAY_NAMES = { junior: 'Vibe', intermediate: 'Mood', advanced: 'Melody' };

/**
 * Show the record screen with a numbered board.
 */
function showRecordScreen() {
    const aiName = AI_DISPLAY_NAMES[gameState.modelManager.selectedModel] || 'AI';

    // Populate player labels
    const blackLabel = gameState.playerColor === Player.BLACK ? '\u73a9\u5bb6' : aiName;
    const whiteLabel = gameState.playerColor === Player.WHITE ? '\u73a9\u5bb6' : aiName;
    const blackDot = '<span class="record-piece record-piece-black"></span>';
    const whiteDot = '<span class="record-piece record-piece-white"></span>';
    document.getElementById('record-black').innerHTML = blackDot + ' \u9ed1\u68cb\uff1a' + blackLabel;
    document.getElementById('record-white').innerHTML = whiteDot + ' \u767d\u68cb\uff1a' + whiteLabel;

    // Regret (only if > 0)
    if (gameState.undoCount > 0) {
        document.getElementById('record-regret').textContent = '\u608d\u68cb\u6b21\u6570\uff1a' + gameState.undoCount;
    } else {
        document.getElementById('record-regret').textContent = '';
    }

    // Game length
    document.getElementById('record-length').textContent = '\u68cb\u5c40\u957f\u5ea6\uff1a' + gameState.history.length;

    document.getElementById('record-screen').style.display = 'flex';
    drawRecordBoard();
}

/**
 * Hide the record screen.
 */
function hideRecordScreen() {
    document.getElementById('record-screen').style.display = 'none';
}

/**
 * Find the winning line (5+ in a row) from the last move.
 * @returns {Array|null} Array of [row, col] pairs sorted by position, or null
 */
function findWinningLine() {
    if (gameState.history.length === 0) return null;

    const lastMove = gameState.history[gameState.history.length - 1];
    const { row, col, player } = lastMove;
    const pieces = player === Player.BLACK ? gameState.board.blackPieces : gameState.board.whitePieces;

    const directions = [[0, 1], [1, 0], [1, 1], [1, -1]];

    for (const [dr, dc] of directions) {
        const line = [[row, col]];

        let r = row + dr, c = col + dc;
        while (r >= 0 && r < 15 && c >= 0 && c < 15 && pieces[r][c] === 1) {
            line.push([r, c]);
            r += dr; c += dc;
        }

        r = row - dr; c = col - dc;
        while (r >= 0 && r < 15 && c >= 0 && c < 15 && pieces[r][c] === 1) {
            line.push([r, c]);
            r -= dr; c -= dc;
        }

        if (line.length >= 5) {
            line.sort((a, b) => a[0] !== b[0] ? a[0] - b[0] : a[1] - b[1]);
            return line;
        }
    }

    return null;
}

/**
 * Draw the numbered board onto the record canvas.
 */
function drawRecordBoard() {
    const recordCanvas = document.getElementById('record-board');
    const c = recordCanvas.getContext('2d');

    // Size the canvas to fit the viewport (square, capped at 600px CSS)
    const vw = window.innerWidth;
    const vh = window.innerHeight;
    const cssSize = Math.min(vw - 40, vh - 80, 600);
    const dpr = window.devicePixelRatio || 1;

    recordCanvas.style.width = cssSize + 'px';
    recordCanvas.style.height = cssSize + 'px';
    recordCanvas.width = cssSize * dpr;
    recordCanvas.height = cssSize * dpr;

    // Match content wrapper width to canvas
    document.getElementById('record-content').style.width = cssSize + 'px';

    c.setTransform(1, 0, 0, 1, 0, 0);
    c.scale(dpr, dpr);

    const size = cssSize;
    const padding = size * 0.05;
    const gridSize = (size - 2 * padding) / 14;
    const pieceRadius = gridSize * 0.4;

    // Background
    c.fillStyle = '#F8F8F8';
    c.fillRect(0, 0, size, size);

    // Grid lines
    c.strokeStyle = '#808080';
    c.lineWidth = 1;
    for (let i = 0; i < 15; i++) {
        c.beginPath();
        c.moveTo(padding + i * gridSize, padding);
        c.lineTo(padding + i * gridSize, padding + 14 * gridSize);
        c.stroke();

        c.beginPath();
        c.moveTo(padding, padding + i * gridSize);
        c.lineTo(padding + 14 * gridSize, padding + i * gridSize);
        c.stroke();
    }

    // Star points
    c.fillStyle = '#404040';
    const starPoints = [[3,3],[3,11],[11,3],[11,11],[7,7]];
    for (const [row, col] of starPoints) {
        c.beginPath();
        c.arc(padding + col * gridSize, padding + row * gridSize, size * 0.008, 0, 2 * Math.PI);
        c.fill();
    }

    // Winning line (drawn beneath pieces)
    const winLine = findWinningLine();
    if (winLine) {
        const first = winLine[0];
        const last = winLine[winLine.length - 1];
        c.strokeStyle = '#CC0000';
        c.lineWidth = cssSize < 500 ? 3 : 5;
        c.lineCap = 'round';
        c.beginPath();
        c.moveTo(padding + first[1] * gridSize, padding + first[0] * gridSize);
        c.lineTo(padding + last[1] * gridSize, padding + last[0] * gridSize);
        c.stroke();
    }

    // Pieces with move numbers
    for (let i = 0; i < gameState.history.length; i++) {
        const move = gameState.history[i];
        const moveNum = i + 1;
        const x = padding + move.col * gridSize;
        const y = padding + move.row * gridSize;
        const isBlack = move.player === Player.BLACK;

        // Draw piece
        if (isBlack) {
            c.fillStyle = '#000';
            c.beginPath();
            c.arc(x, y, pieceRadius, 0, 2 * Math.PI);
            c.fill();
        } else {
            c.fillStyle = '#FFF';
            c.strokeStyle = '#000';
            c.lineWidth = 1.5;
            c.beginPath();
            c.arc(x, y, pieceRadius, 0, 2 * Math.PI);
            c.fill();
            c.stroke();
        }

        // Draw number
        c.fillStyle = isBlack ? '#FFF' : '#000';
        const digits = String(moveNum).length;
        const fontSize = digits <= 1 ? pieceRadius * 1.2 : digits <= 2 ? pieceRadius * 1.0 : pieceRadius * 0.8;
        c.font = `bold ${fontSize}px -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif`;
        c.textAlign = 'center';
        c.textBaseline = 'middle';
        c.fillText(String(moveNum), x, y);
    }
}

// ============================================================================
// Start
// ============================================================================

init();
