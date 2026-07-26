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
        level_dial: '初级',
        level_cello: '中级',
        level_curtain: '高级',
        level_melody: '大师',
        desc_dial: '经典模型',
        desc_cello: '进阶模型',
        desc_curtain: '后训练',
        desc_melody: '深度思考',
        start_game: '开始游戏',
        privacy_note: '所有运算均在本地完成，我们不会收集任何使用数据。',
        undo: '悔棋',
        your_turn: '你的回合',
        new_game: '新游戏',
        confirm_move: '确认落子',
        cancel: '取消',
        game_over: '游戏结束',
        generate_record: '生成对局截图',
        play_again: '再玩一局',
        new_setup: '新设置',
        you_won: '你赢了！',
        you_lost: '你输了！',
        draw: '平局',
        downloading_model: '正在下载模型 {percent}%...',
        probe_phase_wasm_setup: '准备 CPU 推理...',
        probe_phase_wasm_timing: '测量 CPU 推理速度...',
        probe_phase_webgpu_setup: '准备 GPU 推理...',
        probe_phase_webgpu_timing: '测量 GPU 推理速度...',
        probe_result_heading: '设备性能预估',
        probe_result_time: '约 {seconds} 秒/步',
        probe_done: '<em class="probe-model-name">Melody</em>（大师难度）会进行更深入的思考，以寻找最佳落子。根据我们刚才进行的性能测试，预计每步约 {seconds} 秒。若希望更快落子，可改用 <em class="probe-model-name">Curtain</em>（高级难度）。',
        probe_failed: '性能测试未能完成，您的设备可能无法运行深度思考模式。',
        download_failed: '模型下载失败，请检查网络后重试。',
        probe_choice_ok: '开始对局',
        probe_choice_curtain: '改用高级难度',
        probe_choice_curtain_timed: '改用 <em class="probe-model-name">Curtain</em>（高级，{seconds} 秒/步）',
        probe_choice_back: '返回设置',
        probe_choice_retest: '重新测试',
        ai_thinking: 'AI思考中...',
        deep_thinking: '思考中...',
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
        level_dial: 'Easy',
        level_cello: 'Medium',
        level_curtain: 'Hard',
        level_melody: 'Master',
        desc_dial: 'Classic Model',
        desc_cello: 'Advanced Model',
        desc_curtain: 'Post-Trained',
        desc_melody: 'Deep Think',
        start_game: 'Start Game',
        privacy_note: 'All computation is done locally. We do not collect any data.',
        undo: 'Undo',
        your_turn: 'Your turn',
        new_game: 'New Game',
        confirm_move: 'Confirm',
        cancel: 'Cancel',
        game_over: 'Game Over',
        generate_record: 'Game Screenshot',
        play_again: 'Play Again',
        new_setup: 'New Setup',
        you_won: 'You won!',
        you_lost: 'You lost!',
        draw: 'Draw',
        downloading_model: 'Downloading model {percent}%...',
        probe_phase_wasm_setup: 'Preparing CPU inference...',
        probe_phase_wasm_timing: 'Benchmarking CPU inference...',
        probe_phase_webgpu_setup: 'Preparing GPU inference...',
        probe_phase_webgpu_timing: 'Benchmarking GPU inference...',
        probe_result_heading: 'Estimated Device Performance',
        probe_result_time: 'About {seconds} seconds per move',
        probe_done: '<em class="probe-model-name">Melody</em> (Master difficulty) thinks deeper to find the best move. Based on the performance test we just ran, expect about {seconds} seconds per move. For faster play, switch to <em class="probe-model-name">Curtain</em> (Hard difficulty).',
        probe_failed: 'The performance test could not complete. This device may not be able to run Deep Think.',
        download_failed: 'Model download failed. Please check your network and try again.',
        probe_choice_ok: 'Start Game',
        probe_choice_curtain: 'Switch to Hard',
        probe_choice_curtain_timed: 'Switch to <em class="probe-model-name">Curtain</em> (Hard, {seconds} sec/move)',
        probe_choice_back: 'Back to Setup',
        probe_choice_retest: 'Re-run test',
        ai_thinking: 'AI thinking...',
        deep_thinking: 'Thinking...',
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

    // Dynamic texts (loading phases, probe dialog) are not covered by
    // data-i18n; their owners listen for this event and re-render.
    document.dispatchEvent(new CustomEvent('gomoku-langchange'));
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
