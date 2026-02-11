/**
 * Internationalization (i18n) Module
 *
 * Manages translations and language switching for the Gomoku web app.
 */

const I18N = {
    zh: {
        title: 'Askr\u4e94\u5b50\u68cb',
        loading_model: '\u6b63\u5728\u52a0\u8f7d\u6a21\u578b...',
        your_pieces: '\u4f60\u7684\u6267\u5b50\uff1a',
        black_first: '\u26ab \u9ed1\u68cb\uff08\u5148\u624b\uff09',
        white_second: '\u26aa \u767d\u68cb\uff08\u540e\u624b\uff09',
        your_opponent: '\u4f60\u7684\u5bf9\u624b\uff1a',
        junior: '\u521d\u7ea7',
        classic_arch: '\u7ecf\u5178\u6a21\u578b\u67b6\u6784',
        intermediate: '\u4e2d\u7ea7',
        advanced_arch: '\u9ad8\u7ea7\u6a21\u578b\u67b6\u6784',
        advanced: '\u9ad8\u7ea7',
        deep_think_desc: '\u6df1\u5ea6\u601d\u8003',
        start_game: '\u5f00\u59cb\u6e38\u620f',
        privacy_note: '\u6240\u6709\u8fd0\u7b97\u5747\u5728\u672c\u5730\u5b8c\u6210\uff0c\u6211\u4eec\u4e0d\u4f1a\u6536\u96c6\u4efb\u4f55\u4f7f\u7528\u6570\u636e\u3002',
        undo: '\u608d\u68cb',
        your_turn: '\u4f60\u7684\u56de\u5408',
        new_game: '\u65b0\u6e38\u620f',
        confirm_move: '\u786e\u8ba4\u843d\u5b50',
        cancel: '\u53d6\u6d88',
        game_over: '\u6e38\u620f\u7ed3\u675f',
        generate_record: '\u751f\u6210\u68cb\u8c31',
        play_again: '\u518d\u73a9\u4e00\u5c40',
        new_setup: '\u65b0\u8bbe\u7f6e',
        you_won: '\u4f60\u8d62\u4e86\uff01',
        you_lost: '\u4f60\u8f93\u4e86\uff01',
        draw: '\u5e73\u5c40',
        ai_thinking: 'AI\u601d\u8003\u4e2d...',
        deep_thinking: '\u6df1\u5ea6\u601d\u8003\u4e2d...',
        ai_error: 'AI\u51fa\u9519\uff0c\u8bf7\u91cd\u65b0\u5f00\u59cb',
        model_load_failed: '\u52a0\u8f7d\u6a21\u578b\u5931\u8d25\uff0c\u8bf7\u5237\u65b0\u9875\u9762\u91cd\u8bd5\u3002',
        player: '\u73a9\u5bb6',
        move_unit: '\u624b',
        sec_per_move: 's/\u624b',
        undo_label: '\u608d\u68cb',
        times: '\u6b21',
        black_label: '\u9ed1\u68cb\uff1a',
        white_label: '\u767d\u68cb\uff1a',
        undo_count: '\u608d\u68cb\u6b21\u6570\uff1a',
        game_length: '\u68cb\u5c40\u957f\u5ea6\uff1a',
        record_title: 'Askr\u4e94\u5b50\u68cb'
    },
    en: {
        title: 'Askr Gomoku',
        loading_model: 'Loading model...',
        your_pieces: 'Your color:',
        black_first: '\u26ab Black (first)',
        white_second: '\u26aa White (second)',
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
