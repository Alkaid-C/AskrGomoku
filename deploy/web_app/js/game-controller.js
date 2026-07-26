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
    gameOver: false,     // Whether the game has ended
    // Timing tracking
    playerTotalTime: 0,  // Total player thinking time in ms
    playerMoveCount: 0,  // Number of player moves
    aiTotalTime: 0,      // Total AI thinking time in ms
    aiMoveCount: 0,      // Number of AI moves
    playerTurnStart: 0,  // Timestamp when player's turn started
    hasLoadedModel: false, // Whether a model has been loaded this session
    gameId: 0,           // Invalidates async work belonging to an older game
    activeAITask: null,  // Promise for inference/search still using a session
    gameResult: null     // Final GameState, retained for language re-rendering
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

    // Must run before the first session creation anywhere on the page
    // (wasm thread count is fixed at first runtime init).
    epProbeConfigureOrtEnv();

    // Show setup panel
    setupPanel.style.display = 'block';

    // Initialize model manager
    gameState.modelManager = new ModelManager();
    gameState.modelManager.initialize();

    // Set up event listeners
    setupEventListeners();
    renderStatus();

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

    // Legacy result-modal controls (the modal is not shown by the current
    // in-place game-end flow), plus the in-place game-record action.
    document.getElementById('btn-play-again').addEventListener('click', playAgain);
    document.getElementById('btn-new-setup').addEventListener('click', newSetup);
    document.getElementById('btn-share').addEventListener('click', showRecordScreen);
    document.getElementById('btn-record-close').addEventListener('click', hideRecordScreen);

    // Remaining end action (the record action is wired above)
    document.getElementById('btn-end-new-game').addEventListener('click', newSetup);

    // Window resize
    window.addEventListener('resize', updateCanvasSize);
    window.addEventListener('resize', () => {
        if (document.getElementById('record-screen').style.display !== 'none') {
            drawRecordBoard();
        }
    });

    // Re-render JS-managed dynamic texts (loading status, probe dialog)
    // when the language changes
    document.addEventListener('gomoku-langchange', refreshDynamicLoadingTexts);
}

// ============================================================================
// Game Flow
// ============================================================================

/**
 * Start a new game with selected settings.
 */
async function startGame() {
    console.log('Starting game...');

    // Invalidate any result still being computed for the previous board.
    // The unabortable inference itself is allowed to finish before its
    // session is released below.
    const gameId = ++gameState.gameId;

    // Get selected color
    const colorBtn = document.querySelector('.btn-color.active');
    gameState.playerColor = parseInt(colorBtn.dataset.color);
    gameState.aiColor = (gameState.playerColor === Player.BLACK) ? Player.WHITE : Player.BLACK;

    // Get selected difficulty (may be downgraded to curtain by the melody
    // dialog's "switch to Hard" choice below)
    const difficultyBtn = document.querySelector('.btn-difficulty.active');
    let difficulty = difficultyBtn.dataset.difficulty;
    gameState.modelManager.setSelectedModel(difficulty);

    console.log(`  Player color: ${gameState.playerColor === Player.BLACK ? 'Black' : 'White'}`);
    console.log(`  AI color: ${gameState.aiColor === Player.BLACK ? 'Black' : 'White'}`);
    console.log(`  Difficulty: ${difficulty}`);

    // Show loading screen
    setupPanel.style.display = 'none';
    loadingScreen.style.display = 'block';
    showLoadingPhase('load');
    setLoadingMessage('loading_model');

    const loadStartTime = performance.now();
    const isFirstLoad = !gameState.hasLoadedModel;

    try {
        await waitForActiveAITask();
        if (gameId !== gameState.gameId) return;
        await gameState.modelManager.releasePolicyPlayer();
        if (gameId !== gameState.gameId) return;

        if (difficulty === 'melody') {
            // Download → EP probe → per-move-time acknowledgment
            const outcome = await prepareMelodyPlayer();
            if (outcome === 'setup') {
                stopLoadingOrbit();
                stopPoemRotation();
                loadingScreen.style.display = 'none';
                setupPanel.style.display = 'block';
                return;
            }
            if (outcome === 'curtain') {
                // "Switch to Hard": start a curtain game directly, and sync
                // the setup panel's selection so a later "new game" agrees.
                difficulty = 'curtain';
                gameState.modelManager.setSelectedModel('curtain');
                document.querySelectorAll('.btn-difficulty').forEach(b =>
                    b.classList.toggle('active', b.dataset.difficulty === 'curtain'));
                showLoadingPhase('load');
                setLoadingMessage('loading_model');
            }
        }
        if (difficulty !== 'melody') {
            // Load model
            gameState.aiPlayer = await gameState.modelManager.loadSelectedModel();
            if (gameId !== gameState.gameId) {
                await gameState.modelManager.releasePolicyPlayer();
                return;
            }

            // Enforce minimum 3s loading screen on first load
            if (isFirstLoad) {
                const elapsed = performance.now() - loadStartTime;
                if (elapsed < 3000) {
                    await new Promise(resolve => setTimeout(resolve, 3000 - elapsed));
                }
            }
        }
        gameState.hasLoadedModel = true;

        // Stop animations
        stopLoadingOrbit();
        stopPoemRotation();

        // The melody evaluation cache spans AI moves within one game, but
        // unrelated late-game positions should not carry into a new game.
        if (difficulty === 'melody') {
            gameState.aiPlayer.resetEvalCache();
        }

        // Initialize board
        gameState.board = new GomokuBoard();
        gameState.history = [];
        gameState.isAIThinking = false;
        gameState.pendingMove = null;
        gameState.undoCount = 0;
        gameState.gameOver = false;
        gameState.gameResult = null;
        gameState.playerTotalTime = 0;
        gameState.playerMoveCount = 0;
        gameState.aiTotalTime = 0;
        gameState.aiMoveCount = 0;
        gameState.playerTurnStart = 0;

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
            await startAIMove(gameId);
        } else {
            gameState.playerTurnStart = performance.now();
            setStatus('your_turn');
        }

    } catch (error) {
        if (gameId !== gameState.gameId) return;
        console.error('Failed to start game:', error);
        stopLoadingOrbit();
        stopPoemRotation();
        alert(t('model_load_failed'));
        loadingScreen.style.display = 'none';
        setupPanel.style.display = 'block';
    }
}

// ============================================================================
// Loading Screen Phases & Dynamic Texts
// ============================================================================

// The loading screen has two distinct looks: 'load' (model download / plain
// load — large orbit + poem) and 'probe' (the EP performance test — smaller
// orbit + progress bar, no poem), so the test reads as its own step rather
// than part of the download.
const LOADING_ORBIT_SIZE = 200;
const PROBE_ORBIT_SIZE = 160;
const LOADING_ORBIT_FPS = 60;
const STATUS_ORBIT_SIZE = 72;
const STATUS_ORBIT_FPS = 15;

function showLoadingPhase(phase) {
    const poemContainer = document.querySelector('.loading-poem-container');
    const bar = document.getElementById('probe-progress');
    if (phase === 'probe') {
        stopPoemRotation();
        poemContainer.style.display = 'none';
        bar.style.display = '';
        showLoadingOrbit(PROBE_ORBIT_SIZE);
    } else {
        bar.style.display = 'none';
        poemContainer.style.display = '';
        startPoemRotation();
        showLoadingOrbit(LOADING_ORBIT_SIZE);
    }
}

// The loading status line and the probe dialog text are set from JS, so
// they carry no data-i18n. Their key/params are remembered and re-rendered
// on language switch (otherwise applyTranslations would either clobber them
// or leave them in the old language).
let loadingMessageState = null;
let probeDialogTextState = null;

function setLoadingMessage(key, params) {
    loadingMessageState = key ? { key: key, params: params } : null;
    document.getElementById('loading-message').textContent =
        key ? (params ? tFormat(key, params) : t(key)) : '';
}

function refreshDynamicLoadingTexts() {
    if (loadingMessageState) {
        setLoadingMessage(loadingMessageState.key, loadingMessageState.params);
    }
    if (probeDialogTextState) {
        renderProbeDialogText();
    }
    if (gameState.gameOver && gameState.gameResult !== null) {
        renderGameEnd(gameState.gameResult);
    } else {
        renderStatus();
    }
    if (document.getElementById('record-screen').style.display !== 'none') {
        showRecordScreen();
    }
}

function renderProbeDialogText() {
    const { key, params } = probeDialogTextState;
    const isResult = key === 'probe_done';
    const heading = document.getElementById('probe-result-heading');
    const time = document.getElementById('probe-result-time');
    const textEl = document.getElementById('probe-result-text');
    const curtainButton = document.getElementById('btn-probe-curtain');

    heading.style.display = isResult ? '' : 'none';
    time.style.display = isResult ? '' : 'none';
    heading.textContent = isResult ? t('probe_result_heading') : '';
    time.textContent = isResult ? tFormat('probe_result_time', params) : '';
    if (isResult) {
        textEl.innerHTML = tFormat(key, params);
        curtainButton.innerHTML = params.curtainSeconds
            ? tFormat('probe_choice_curtain_timed', { seconds: params.curtainSeconds })
            : t('probe_choice_curtain');
    } else {
        textEl.textContent = params ? tFormat(key, params) : t(key);
        curtainButton.textContent = t('probe_choice_curtain');
    }
}

// ============================================================================
// EP-Probe Progress Bar
// ============================================================================

// One segment per probe phase: wasm setup/warmup, wasm timing, then (when a
// GPU is present) webgpu setup/warmup and webgpu timing. Within a
// segment the fill advances linearly on the worst-case budget
// (EP_PROBE_TOTAL_CAP_MS — the point at which the probe worker would be
// killed anyway), so the bar is always moving yet can never overshoot a
// phase that is still running.
const probeBar = { nSegments: 2, segment: 0, segmentStart: 0, rafId: null };

const PROBE_PHASE_TEXT_KEYS = {
    wasm: { timing: 'probe_phase_wasm_timing', other: 'probe_phase_wasm_setup' },
    webgpu: { timing: 'probe_phase_webgpu_timing', other: 'probe_phase_webgpu_setup' },
};

function probeBarStart() {
    probeBar.nSegments = epProbeEnvironment().hasGpu ? 4 : 2;
    probeBar.segment = 0;
    probeBar.segmentStart = performance.now();
    cancelAnimationFrame(probeBar.rafId);
    const fill = document.getElementById('probe-progress-fill');
    const tick = () => {
        const frac = Math.min(
            (performance.now() - probeBar.segmentStart) / EP_PROBE_TOTAL_CAP_MS, 1);
        fill.style.width =
            ((probeBar.segment + frac) / probeBar.nSegments * 100) + '%';
        probeBar.rafId = requestAnimationFrame(tick);
    };
    tick();
}

function probeBarOnPhase(ep, phase) {
    const keys = PROBE_PHASE_TEXT_KEYS[ep];
    setLoadingMessage(phase === 'timing' ? keys.timing : keys.other);
    const segment = (ep === 'webgpu' ? 2 : 0) + (phase === 'timing' ? 1 : 0);
    if (segment > probeBar.segment) {
        probeBar.segment = segment;
        probeBar.segmentStart = performance.now();
    }
}

function probeBarStop() {
    cancelAnimationFrame(probeBar.rafId);
    probeBar.rafId = null;
    document.getElementById('probe-progress-fill').style.width = '100%';
}

// ============================================================================
// Melody Preparation (download → probe → acknowledgment)
// ============================================================================

/**
 * Prepare the melody-difficulty AI player. The model download is kept on
 * the ModelManager, so a probe re-run never re-downloads. After the probe
 * (or a cached restore), the per-move time estimate is shown for a one-time
 * acknowledgment — its text assumes the reader saw nothing of the probe
 * screens. Acknowledging persists `confirmed`, so later melody games skip
 * the dialog entirely; "switch to Hard" keeps the probe result but not the
 * acknowledgment, so the estimate is shown again next time.
 * @returns {Promise<string>} 'melody' (player ready) | 'curtain' (start a
 *     curtain game instead) | 'setup' (return to the setup panel)
 */
async function prepareMelodyPlayer() {
    try {
        await gameState.modelManager.fetchMelodyModel(frac =>
            setLoadingMessage('downloading_model', { percent: Math.round(frac * 100) }));
    } catch (e) {
        console.error('Melody model download failed:', e);
        return showProbeDialog('download_failed', null, ['back']);
    }

    let force = false;
    for (;;) {
        setLoadingMessage('loading_model'); // covers cached-EP session restore
        let probe;
        try {
            ({ probe } = await gameState.modelManager.probeMelody({
                onProbeStart: () => {
                    showLoadingPhase('probe');
                    probeBarStart();
                },
                onPhase: probeBarOnPhase,
                force: force,
            }));
        } catch (e) {
            console.error('EP probe failed:', e);
            probeBarStop();
            return showProbeDialog('probe_failed', null, ['curtain', 'back']);
        }
        probeBarStop();

        const cached = epProbeLoadCache();
        if (cached && cached.confirmed) break;

        // Mean predicts the many-inference total (sum = n × mean).
        const seconds = (MCTS_SIMS * probe.meanMs / 1000).toFixed(1);
        const curtainSeconds = typeof probe.wasmMeanMs === 'number'
            ? (probe.wasmMeanMs < 100 ? '<0.1' : (probe.wasmMeanMs / 1000).toFixed(1))
            : null;
        const choice = await showProbeDialog('probe_done', {
            seconds: seconds,
            curtainSeconds: curtainSeconds,
        },
            ['ok', 'curtain', 'retest']);
        if (choice === 'ok') {
            if (cached) {
                epProbeSaveCache(Object.assign({}, cached, { confirmed: true }));
            }
            break;
        }
        if (choice === 'curtain') return 'curtain';
        force = true; // 'retest': force a fresh probe (model already in memory)
    }

    gameState.aiPlayer = gameState.modelManager.melodyPlayer;
    return 'melody';
}

/**
 * Show the melody dialog on the loading screen: the per-move time estimate
 * (sim counts are never shown — only sims × measured mean latency) or a
 * download/probe failure message. The same markup serves all cases; which
 * buttons are visible varies.
 * @param {string} textKey - i18n key for the dialog text
 * @param {Object|null} params - tFormat params for the text
 * @param {Array} buttons - Visible buttons: 'ok' | 'curtain' | 'back' | 'retest'
 * @returns {Promise<string>} 'ok' | 'curtain' | 'setup' | 'retest'
 */
function showProbeDialog(textKey, params, buttons) {
    const dialog = document.getElementById('probe-dialog');
    const orbit = document.getElementById('loading-orbit');
    const btns = {
        ok: document.getElementById('btn-probe-ok'),
        curtain: document.getElementById('btn-probe-curtain'),
        back: document.getElementById('btn-probe-back'),
        retest: document.getElementById('btn-probe-retest'),
    };

    stopLoadingOrbit();
    stopPoemRotation();
    orbit.style.display = 'none';
    document.getElementById('probe-progress').style.display = 'none';
    document.querySelector('.loading-poem-container').style.display = 'none';
    setLoadingMessage(null);

    probeDialogTextState = { key: textKey, params: params };
    renderProbeDialogText();
    for (const [name, el] of Object.entries(btns)) {
        el.style.display = buttons.includes(name) ? '' : 'none';
    }
    dialog.style.display = 'block';

    return new Promise(resolve => {
        const finish = (choice) => {
            for (const el of Object.values(btns)) el.onclick = null;
            probeDialogTextState = null;
            dialog.style.display = 'none';
            orbit.style.display = '';
            resolve(choice);
        };
        btns.ok.onclick = () => finish('ok');
        btns.curtain.onclick = () => finish('curtain');
        btns.back.onclick = () => finish('setup');
        btns.retest.onclick = () => finish('retest');
    });
}

/**
 * Restart game with same settings.
 */
function restartGame() {
    console.log('Restarting game...');

    // session.run() cannot be aborted, but its result must no longer be able
    // to mutate global state after the setup panel is shown.
    gameState.gameId++;
    gameState.isAIThinking = false;
    gameState.pendingMove = null;
    setStatus('your_turn');

    // Show setup panel
    gamePanel.style.display = 'none';
    setupPanel.style.display = 'block';
}

/**
 * Play again (after game ends).
 */
function playAgain() {
    const gameId = ++gameState.gameId;
    resultModal.style.display = 'none';
    resetEndModeUI();

    if (gameState.modelManager.selectedModel === 'melody') {
        gameState.aiPlayer.resetEvalCache();
    }

    // Reset board
    gameState.board = new GomokuBoard();
    gameState.history = [];
    gameState.isAIThinking = false;
    gameState.pendingMove = null;
    gameState.undoCount = 0;
    gameState.gameOver = false;
    gameState.gameResult = null;
    gameState.playerTotalTime = 0;
    gameState.playerMoveCount = 0;
    gameState.aiTotalTime = 0;
    gameState.aiMoveCount = 0;
    gameState.playerTurnStart = 0;

    drawBoard();

    // If AI plays first, make AI move
    if (gameState.aiColor === Player.BLACK) {
        startAIMove(gameId);
    } else {
        gameState.playerTurnStart = performance.now();
        setStatus('your_turn');
    }
}

/**
 * New setup (return to setup panel).
 */
function newSetup() {
    gameState.gameId++;
    resultModal.style.display = 'none';
    resetEndModeUI();
    gameState.gameOver = false;
    gameState.gameResult = null;
    gameState.isAIThinking = false;
    gameState.pendingMove = null;
    setStatus('your_turn');
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

    if (!gameState.pendingMove || gameState.isAIThinking
        || gameState.board.whoToPlay !== gameState.playerColor) return;

    const gameId = gameState.gameId;
    const { row, col } = gameState.pendingMove;

    // Record player thinking time
    if (gameState.playerTurnStart > 0) {
        gameState.playerTotalTime += performance.now() - gameState.playerTurnStart;
        gameState.playerMoveCount++;
        gameState.playerTurnStart = 0;
    }

    // Hide confirmation buttons
    document.getElementById('move-confirm').classList.remove('visible');
    gameState.pendingMove = null;

    // Make move
    await makePlayerMove(row, col, gameId);
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
async function makePlayerMove(row, col, gameId) {
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
    await startAIMove(gameId);
}

/**
 * Start and track one AI task. New games wait for this promise before
 * releasing the policy session that the task may still be using.
 */
function startAIMove(gameId = gameState.gameId) {
    const task = makeAIMove(gameId);
    gameState.activeAITask = task;
    const clearTask = () => {
        if (gameState.activeAITask === task) gameState.activeAITask = null;
    };
    task.then(clearTask, clearTask);
    return task;
}

async function waitForActiveAITask() {
    const task = gameState.activeAITask;
    if (task) await task;
}

/**
 * Make an AI move for one game generation. The board, player, and model are
 * captured before yielding so a later game cannot be used accidentally.
 */
async function makeAIMove(gameId) {
    if (gameId !== gameState.gameId) return;

    const board = gameState.board;
    const aiPlayer = gameState.aiPlayer;
    const aiColor = gameState.aiColor;
    const useMCTS = gameState.modelManager.selectedModel === 'melody';
    gameState.isAIThinking = true;

    if (useMCTS) {
        setStatus('deep_thinking');
    } else {
        setStatus('ai_thinking');
    }

    try {
        // Give browser a chance to update UI before heavy computation
        await new Promise(resolve => setTimeout(resolve, 50));
        if (gameId !== gameState.gameId) return;

        // Get AI move
        const startTime = performance.now();
        let aiRow, aiCol, aiValue;

        if (useMCTS) {
            // Melody: MCTS
            [aiRow, aiCol, aiValue] = await aiPlayer.getMoveWithMCTS(board, MCTS_SIMS);
        } else {
            // Dial/cello/curtain: sample from policy distribution
            [aiRow, aiCol, aiValue] = await aiPlayer.getMove(board);
        }

        const endTime = performance.now();
        if (gameId !== gameState.gameId) return;

        gameState.lastAIThinkTime = (endTime - startTime) / 1000; // Convert to seconds
        gameState.aiTotalTime += endTime - startTime;
        gameState.aiMoveCount++;

        console.log(`AI move: (${aiRow}, ${aiCol}), think time: ${gameState.lastAIThinkTime.toFixed(2)}s`);

        // Save to history (value is from AI's perspective before this move)
        gameState.history.push({
            row: aiRow, col: aiCol,
            player: aiColor,
            value: aiValue,
            blackPieces: board.blackPieces.map(r => [...r]),
            whitePieces: board.whitePieces.map(r => [...r]),
            whoToPlay: board.whoToPlay,
            occupiedCount: board.occupiedCount
        });

        // Make move
        const result = board.Move(aiRow, aiCol);

        drawBoard();

        gameState.isAIThinking = false;

        // Check game result
        if (result !== GameState.CONTINUE) {
            handleGameEnd(result);
            return;
        }

        // Player's turn
        gameState.playerTurnStart = performance.now();
        setStatus('your_turn');

    } catch (error) {
        if (gameId !== gameState.gameId) return;
        console.error('AI move failed:', error);
        gameState.isAIThinking = false;
        setStatus('ai_error');
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
        const popped = gameState.history.pop();
        // Adjust move counts for timing averages
        if (popped.player === gameState.playerColor) {
            gameState.playerMoveCount = Math.max(0, gameState.playerMoveCount - 1);
        } else {
            gameState.aiMoveCount = Math.max(0, gameState.aiMoveCount - 1);
        }
        restoreState = popped;
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
        gameState.playerTurnStart = performance.now();
        setStatus('your_turn');
    } else {
        startAIMove(gameState.gameId);
    }
}

/**
 * Handle game end.
 */
function handleGameEnd(result) {
    console.log(`Game ended: ${result}`);
    gameState.gameOver = true;
    gameState.gameResult = result;
    renderGameEnd(result);
}

/**
 * Render the final result from state. Kept separate so language changes can
 * rebuild all translated result text without replaying game-end side effects.
 */
function renderGameEnd(result) {

    // Determine result title
    let titleKey = 'game_over';
    if (result === GameState.BLACK_WIN) {
        titleKey = gameState.playerColor === Player.BLACK ? 'you_won' : 'you_lost';
    } else if (result === GameState.WHITE_WIN) {
        titleKey = gameState.playerColor === Player.WHITE ? 'you_won' : 'you_lost';
    } else if (result === GameState.DRAW) {
        titleKey = 'draw';
    }

    // Player info
    const aiName = AI_DISPLAY_NAMES[gameState.modelManager.selectedModel] || 'AI';
    const fullAiName = 'Askr-' + aiName;
    const blackDot = '<span class="record-piece record-piece-black"></span>';
    const whiteDot = '<span class="record-piece record-piece-white"></span>';

    // Compute average time per move
    const playerAvg = gameState.playerMoveCount > 0
        ? (gameState.playerTotalTime / gameState.playerMoveCount / 1000).toFixed(2) : '0.00';
    const aiAvg = gameState.aiMoveCount > 0
        ? (gameState.aiTotalTime / gameState.aiMoveCount / 1000).toFixed(2) : '0.00';
    const playerTimeStr = ' <span class="time-per-move">(' + playerAvg + t('sec_per_move') + ')</span>';
    const aiTimeStr = ' <span class="time-per-move">(' + aiAvg + t('sec_per_move') + ')</span>';

    const blackIsPlayer = gameState.playerColor === Player.BLACK;
    const blackLabel = blackIsPlayer ? t('player') + playerTimeStr : fullAiName + aiTimeStr;
    const whiteLabel = !blackIsPlayer ? t('player') + playerTimeStr : fullAiName + aiTimeStr;

    // Stats
    const gameLength = gameState.history.length;
    const userWon = (result === GameState.BLACK_WIN && gameState.playerColor === Player.BLACK) ||
                    (result === GameState.WHITE_WIN && gameState.playerColor === Player.WHITE);
    let rightHtml = gameLength + t('move_unit');
    if (gameState.undoCount > 0 && userWon) {
        rightHtml += '<br>' + t('undo_label') + gameState.undoCount + t('times');
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
    setStatus(titleKey);

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
    // Resize can fire before any game has started (no board yet), e.g. the
    // mobile address bar showing/hiding during the loading/probe screens.
    if (!gameState.board) return;

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

let statusMessageState = { key: 'your_turn', params: null };
let statusOrbitLoader = null;

/**
 * Store status as an i18n key so language changes preserve its meaning.
 */
function setStatus(key, params = null) {
    statusMessageState = { key: key, params: params };
    renderStatus();
}

function renderStatus() {
    const { key, params } = statusMessageState;
    document.getElementById('status-label').textContent =
        params ? tFormat(key, params) : t(key);

    const orbit = document.getElementById('status-orbit');
    const showOrbit = key === 'deep_thinking';
    document.getElementById('status-text').classList.toggle(
        'is-deep-thinking', showOrbit);
    orbit.style.display = showOrbit ? 'block' : 'none';

    if (showOrbit) {
        if (!statusOrbitLoader) {
            statusOrbitLoader = createOrbitLoader(orbit, {
                size: STATUS_ORBIT_SIZE,
                fps: STATUS_ORBIT_FPS,
            });
        }
        statusOrbitLoader.start();
    } else if (statusOrbitLoader) {
        statusOrbitLoader.stop();
    }
}

// ============================================================================
// Loading Poem Rotation
// ============================================================================

const poemLines = [
    'Midway through, a haunted computer types its own questions:',
    '\u201CWould you like to meet a ghost?\u201D',
    '\u201CDo you live to shovel sand or shovel sand to live?\u201D',
    'It\u2019s the best part of the movie,',
    'I think you\u2019d like it.',
    'There\u2019s this melody one character hears after',
    'in his head\u2014it is the answer, we discover, to everything',
    'not yet asked; a sort of dial tone',
    'overtakes you with dread while you\u2019re watching',
    'him listen to a wind-blown curtain swell',
    'into a cello or a pear-shaped person, illegibility\u2019s the point',
    'and also the mood. Or is it a vibe? I think moods',
    'are for people with choices',
    'and children\u2014',
    'Anyway the room\u2019s crummy,',
    'how was karaoke?',
    'Will you call again later and sing it for me?'
];

let poemInterval = null;
let poemIndex = 0;

function startPoemRotation() {
    stopPoemRotation(); // idempotent: never stack a second interval
    const el = document.getElementById('loading-poem');
    poemIndex = Math.floor(Math.random() * poemLines.length);
    el.textContent = poemLines[poemIndex];
    el.className = 'loading-poem';

    poemInterval = setInterval(() => {
        el.classList.add('slide-out');

        setTimeout(() => {
            poemIndex = (poemIndex + 1) % poemLines.length;
            el.classList.remove('slide-out');
            el.classList.add('slide-in-prep');
            el.textContent = poemLines[poemIndex];

            // Force reflow then slide in
            el.offsetHeight;
            el.classList.remove('slide-in-prep');
        }, 500);
    }, 2000);
}

function stopPoemRotation() {
    if (poemInterval) {
        clearInterval(poemInterval);
        poemInterval = null;
    }
}

// ============================================================================
// Orbital Loading Animation (loading screen instance of orbit-loader.js)
// ============================================================================

let loadingOrbit = null;
let loadingOrbitSize = 0;

/**
 * Show (and start) the loading screen's orbit animation at a given size,
 * recreating the loader only when the size changes.
 */
function showLoadingOrbit(size) {
    if (!loadingOrbit || loadingOrbitSize !== size) {
        if (loadingOrbit) loadingOrbit.destroy();
        loadingOrbit = createOrbitLoader(
            document.getElementById('loading-orbit'), {
                size: size,
                fps: LOADING_ORBIT_FPS,
            });
        loadingOrbitSize = size;
    }
    loadingOrbit.start();
}

function stopLoadingOrbit() {
    if (loadingOrbit) loadingOrbit.stop();
}

// ============================================================================
// Record Screen (numbered board for screenshots)
// ============================================================================

const AI_DISPLAY_NAMES = { dial: 'Dial', cello: 'Cello', curtain: 'Curtain', melody: 'Melody' };

/**
 * Show the record screen with a numbered board.
 */
function showRecordScreen() {
    const aiName = AI_DISPLAY_NAMES[gameState.modelManager.selectedModel] || 'AI';

    // Populate player labels
    const blackLabel = gameState.playerColor === Player.BLACK ? t('player') : aiName;
    const whiteLabel = gameState.playerColor === Player.WHITE ? t('player') : aiName;
    const blackDot = '<span class="record-piece record-piece-black"></span>';
    const whiteDot = '<span class="record-piece record-piece-white"></span>';
    document.getElementById('record-black').innerHTML = blackDot + ' ' + t('black_label') + blackLabel;
    document.getElementById('record-white').innerHTML = whiteDot + ' ' + t('white_label') + whiteLabel;

    // Regret (only if > 0)
    if (gameState.undoCount > 0) {
        document.getElementById('record-regret').textContent = t('undo_count') + gameState.undoCount;
    } else {
        document.getElementById('record-regret').textContent = '';
    }

    // Game length
    document.getElementById('record-length').textContent = t('game_length') + gameState.history.length;

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

    // Size the canvas (square, capped at 600px CSS) so the whole column —
    // title, info rows, board, footer — fits the viewport. The non-canvas
    // height is measured live (the screen is already visible when this
    // runs), plus 10px breathing room on each side. Below 300px the board
    // is not worth shrinking further (move numbers become unreadable):
    // keep 300px and let the overlay scroll instead.
    const vw = window.innerWidth;
    const vh = window.innerHeight;
    let chromeHeight = 0;
    for (const sel of ['.record-title', '.record-info', '.record-footer']) {
        const el = document.querySelector(sel);
        const style = getComputedStyle(el);
        chromeHeight += el.offsetHeight
            + parseFloat(style.marginTop) + parseFloat(style.marginBottom);
    }
    const cssSize = Math.max(Math.min(vw - 40, vh - chromeHeight - 20, 600), 300);
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
