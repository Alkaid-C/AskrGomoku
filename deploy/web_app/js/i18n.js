/**
 * Internationalization (i18n) Module
 *
 * Manages translations and language switching for the Gomoku web app.
 */

const I18N = {
    zh: {
        title: 'Askr五子棋',
        loading_model: '正在加载模型...',
        your_pieces: '你的执子：',
        black_first: '⚫ 黑棋（先手）',
        white_second: '⚪ 白棋（后手）',
        your_opponent: '你的对手：',
        junior: '初级',
        classic_arch: '经典模型架构',
        intermediate: '中级',
        advanced_arch: '高级模型架构',
        advanced: '高级',
        deep_think_desc: '深度思考',
        start_game: '开始游戏',
        privacy_note: '所有运算均在本地完成，我们不会收集任何使用数据。',
        undo: '悔棋',
        your_turn: '你的回合',
        new_game: '新游戏',
        confirm_move: '确认落子',
        cancel: '取消',
        game_over: '游戏结束',
        generate_record: '生成棋谱',
        play_again: '再玩一局',
        new_setup: '新设置',
        you_won: '你赢了！',
        you_lost: '你输了！',
        draw: '平局',
        downloading_model: '正在下载模型 {percent}%...',
        probe_running: '高阶算法需要更长运行时间，正在测试最快运行方式（通常约 10 秒，最长约 1 分钟）...',
        probe_done: '测试结束：预计每步需要约 {seconds} 秒。',
        probe_choice_ok: '我知道了',
        probe_choice_retest: '重新测试',
        probe_choice_repick: '选择其他难度',
        ai_thinking: 'AI思考中...',
        deep_thinking: '深度思考中...',
        ai_error: 'AI出错，请重新开始',
        model_load_failed: '加载模型失败，请刷新页面重试。',
        player: '玩家',
        move_unit: '手',
        sec_per_move: 's/手',
        undo_label: '悔棋',
        times: '次',
        black_label: '黑棋：',
        white_label: '白棋：',
        undo_count: '悔棋次数：',
        game_length: '棋局长度：',
        record_title: 'Askr五子棋'
    },
    en: {
        title: 'Askr Gomoku',
        loading_model: 'Loading model...',
        your_pieces: 'Your color:',
        black_first: '⚫ Black (first)',
        white_second: '⚪ White (second)',
        your_opponent: 'Your opponent:',
        junior: 'Easy',
        classic_arch: 'Classic model',
        intermediate: 'Medium',
        advanced_arch: 'Advanced model',
        advanced: 'Hard',
        deep_think_desc: 'Deep thinking',
        start_game: 'Start Game',
        privacy_note: 'All computation is done locally. We do not collect any data.',
        undo: 'Undo',
        your_turn: 'Your turn',
        new_game: 'New Game',
        confirm_move: 'Confirm',
        cancel: 'Cancel',
        game_over: 'Game Over',
        generate_record: 'Game Record',
        play_again: 'Play Again',
        new_setup: 'New Setup',
        you_won: 'You won!',
        you_lost: 'You lost!',
        draw: 'Draw',
        downloading_model: 'Downloading model {percent}%...',
        probe_running: 'The advanced engine needs more compute per move. Testing the fastest way to run it (usually ~10 seconds, up to a minute)...',
        probe_done: 'Test complete: expect about {seconds}s of thinking per move.',
        probe_choice_ok: 'Got it',
        probe_choice_retest: 'Re-run test',
        probe_choice_repick: 'Choose another difficulty',
        ai_thinking: 'AI thinking...',
        deep_thinking: 'Deep thinking...',
        ai_error: 'AI error, please restart',
        model_load_failed: 'Failed to load model. Please refresh and try again.',
        player: 'Player',
        move_unit: ' moves',
        sec_per_move: 's/move',
        undo_label: 'Undo',
        times: 'x',
        black_label: 'Black: ',
        white_label: 'White: ',
        undo_count: 'Undos: ',
        game_length: 'Game length: ',
        record_title: 'Askr Gomoku'
    }
};

let currentLang = 'zh';

/**
 * Get translated string by key.
 * @param {string} key - Translation key
 * @returns {string} Translated string
 */
function t(key) {
    return (I18N[currentLang] && I18N[currentLang][key]) || (I18N.zh[key]) || key;
}

/**
 * Get translated string with {placeholder} substitution.
 * @param {string} key - Translation key
 * @param {Object} params - Placeholder values, e.g. {seconds: '2.5'}
 * @returns {string} Translated string with placeholders filled in
 */
function tFormat(key, params) {
    let s = t(key);
    for (const [name, value] of Object.entries(params)) {
        s = s.replaceAll('{' + name + '}', String(value));
    }
    return s;
}

/**
 * Apply translations to all elements with data-i18n attributes.
 */
function applyTranslations() {
    document.querySelectorAll('[data-i18n]').forEach(el => {
        const key = el.getAttribute('data-i18n');
        el.textContent = t(key);
    });
    document.documentElement.lang = currentLang === 'zh' ? 'zh-CN' : 'en';
    document.title = t('title');
}

/**
 * Set language and update all UI text.
 * @param {string} lang - Language code ('zh' or 'en')
 */
function setLang(lang) {
    currentLang = lang;
    localStorage.setItem('gomoku-lang', lang);

    // Update switcher button states
    document.querySelectorAll('.lang-btn').forEach(btn => {
        btn.classList.toggle('active', btn.getAttribute('data-lang') === lang);
    });

    applyTranslations();
}

/**
 * Initialize i18n: detect language from localStorage or browser.
 */
function initI18n() {
    const saved = localStorage.getItem('gomoku-lang');
    if (saved && I18N[saved]) {
        currentLang = saved;
    } else {
        const browserLang = (navigator.language || navigator.userLanguage || 'zh').toLowerCase();
        currentLang = browserLang.startsWith('zh') ? 'zh' : 'en';
    }

    // Set up language switcher click handlers
    document.querySelectorAll('.lang-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            setLang(btn.getAttribute('data-lang'));
        });
    });

    // Apply initial language
    setLang(currentLang);
}

// Initialize when DOM is ready
initI18n();
