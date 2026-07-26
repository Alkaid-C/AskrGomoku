/**
 * Melody evaluation-cache file format and loader.
 *
 * The cache stores exact black/white positions and raw network outputs. It is
 * tied to the exact ONNX bytes by SHA-256; any download, format, or model
 * mismatch is handled by ModelManager as an optional-optimization failure.
 */

const EVAL_CACHE_MAGIC = 'GMKECACH';
const EVAL_CACHE_FORMAT_VERSION = 1;
const EVAL_CACHE_HEADER_SIZE = 64;
const EVAL_CACHE_BOARD_SIZE = 15;
const EVAL_CACHE_POLICY_SIZE = 225;
const EVAL_CACHE_RECORD_SIZE = 964;
const EVAL_CACHE_MODEL_HASH_SIZE = 32;

const EVAL_CACHE_HEADER_MODEL_HASH_OFFSET = 0x18;
const EVAL_CACHE_RECORD_BLACK_OFFSET = 0x000;
const EVAL_CACHE_RECORD_WHITE_OFFSET = 0x01e;
const EVAL_CACHE_RECORD_LOGITS_OFFSET = 0x03c;
const EVAL_CACHE_RECORD_VALUE_OFFSET = 0x3c0;

/**
 * Download a cache file. Kept separate from parsing so ModelManager can start
 * this small optional download in parallel with the larger ONNX download.
 * @param {string} url
 * @returns {Promise<ArrayBuffer>}
 */
async function evalCacheFetchFile(url) {
    const response = await fetch(url);
    if (!response.ok) {
        throw new Error(`HTTP ${response.status} for ${url}`);
    }
    return response.arrayBuffer();
}

/**
 * Calculate the SHA-256 of the exact ONNX file bytes.
 * @param {Uint8Array} modelBytes
 * @returns {Promise<Uint8Array>}
 */
async function evalCacheModelSha256(modelBytes) {
    if (!globalThis.crypto || !globalThis.crypto.subtle) {
        throw new Error('Web Crypto SHA-256 is unavailable');
    }
    const digest = await globalThis.crypto.subtle.digest('SHA-256', modelBytes);
    return new Uint8Array(digest);
}

/**
 * Parse and validate a v1 cache file.
 *
 * Records are returned in LRU order, oldest first. No partially valid file is
 * accepted: malformed metadata, positions, outputs, duplicates, or a model
 * mismatch reject the entire seed cache.
 *
 * @param {ArrayBuffer} fileBuffer
 * @param {Uint8Array} expectedModelHash - SHA-256 of the loaded ONNX bytes
 * @param {number} maxEntries
 * @returns {Array<Object>} [{key, evaluation: {logits, value}}, ...]
 */
function evalCacheParseFile(fileBuffer, expectedModelHash, maxEntries) {
    if (!(fileBuffer instanceof ArrayBuffer)) {
        throw new Error('Eval cache must be an ArrayBuffer');
    }
    if (!(expectedModelHash instanceof Uint8Array)
            || expectedModelHash.length !== EVAL_CACHE_MODEL_HASH_SIZE) {
        throw new Error('Expected model hash must be a 32-byte Uint8Array');
    }
    if (!Number.isInteger(maxEntries) || maxEntries < 0) {
        throw new Error(`Invalid eval cache capacity: ${maxEntries}`);
    }
    if (fileBuffer.byteLength < EVAL_CACHE_HEADER_SIZE) {
        throw new Error(`Eval cache is truncated (${fileBuffer.byteLength} bytes)`);
    }

    const view = new DataView(fileBuffer);
    for (let i = 0; i < EVAL_CACHE_MAGIC.length; i++) {
        if (view.getUint8(i) !== EVAL_CACHE_MAGIC.charCodeAt(i)) {
            throw new Error('Invalid eval cache magic');
        }
    }

    const version = view.getUint16(0x08, true);
    const boardSize = view.getUint16(0x0a, true);
    const policySize = view.getUint16(0x0c, true);
    const entryCount = view.getUint16(0x0e, true);
    const recordSize = view.getUint32(0x10, true);
    const flags = view.getUint32(0x14, true);

    if (version !== EVAL_CACHE_FORMAT_VERSION) {
        throw new Error(`Unsupported eval cache version: ${version}`);
    }
    if (boardSize !== EVAL_CACHE_BOARD_SIZE
            || policySize !== EVAL_CACHE_POLICY_SIZE
            || recordSize !== EVAL_CACHE_RECORD_SIZE) {
        throw new Error('Eval cache dimensions or record size do not match this app');
    }
    if (flags !== 0) {
        throw new Error(`Unsupported eval cache flags: ${flags}`);
    }
    if (entryCount > maxEntries) {
        throw new Error(`Eval cache has ${entryCount} entries; capacity is ${maxEntries}`);
    }

    const expectedSize =
        EVAL_CACHE_HEADER_SIZE + entryCount * EVAL_CACHE_RECORD_SIZE;
    if (fileBuffer.byteLength !== expectedSize) {
        throw new Error(
            `Eval cache size mismatch: expected ${expectedSize}, got ${fileBuffer.byteLength}`);
    }

    for (let i = 0; i < EVAL_CACHE_MODEL_HASH_SIZE; i++) {
        const actual = view.getUint8(EVAL_CACHE_HEADER_MODEL_HASH_OFFSET + i);
        if (actual !== expectedModelHash[i]) {
            throw new Error('Eval cache was generated for a different ONNX model');
        }
    }
    for (let i = 0x38; i < EVAL_CACHE_HEADER_SIZE; i++) {
        if (view.getUint8(i) !== 0) {
            throw new Error('Eval cache reserved header bytes must be zero');
        }
    }

    const entries = new Array(entryCount);
    const seenKeys = new Set();
    for (let entryIndex = 0; entryIndex < entryCount; entryIndex++) {
        const base =
            EVAL_CACHE_HEADER_SIZE + entryIndex * EVAL_CACHE_RECORD_SIZE;
        const rowMasks = new Array(2 * EVAL_CACHE_BOARD_SIZE);
        let blackCount = 0;
        let whiteCount = 0;

        for (let row = 0; row < EVAL_CACHE_BOARD_SIZE; row++) {
            const blackMask = view.getUint16(
                base + EVAL_CACHE_RECORD_BLACK_OFFSET + 2 * row, true);
            const whiteMask = view.getUint16(
                base + EVAL_CACHE_RECORD_WHITE_OFFSET + 2 * row, true);
            if ((blackMask & 0x8000) !== 0 || (whiteMask & 0x8000) !== 0) {
                throw new Error(`Entry ${entryIndex} has stones outside the board`);
            }
            if ((blackMask & whiteMask) !== 0) {
                throw new Error(`Entry ${entryIndex} has overlapping stones`);
            }
            rowMasks[row] = blackMask;
            rowMasks[EVAL_CACHE_BOARD_SIZE + row] = whiteMask;
            blackCount += _evalCachePopcount15(blackMask);
            whiteCount += _evalCachePopcount15(whiteMask);
        }
        if (blackCount !== whiteCount && blackCount !== whiteCount + 1) {
            throw new Error(`Entry ${entryIndex} has unreachable stone counts`);
        }

        const key = String.fromCharCode(...rowMasks);
        if (seenKeys.has(key)) {
            throw new Error(`Entry ${entryIndex} duplicates an earlier position`);
        }
        seenKeys.add(key);

        const logits = new Float32Array(EVAL_CACHE_POLICY_SIZE);
        for (let action = 0; action < EVAL_CACHE_POLICY_SIZE; action++) {
            const logit = view.getFloat32(
                base + EVAL_CACHE_RECORD_LOGITS_OFFSET + 4 * action, true);
            if (!Number.isFinite(logit)) {
                throw new Error(`Entry ${entryIndex} has a non-finite policy logit`);
            }
            logits[action] = logit;
        }
        const value = view.getFloat32(
            base + EVAL_CACHE_RECORD_VALUE_OFFSET, true);
        if (!Number.isFinite(value) || value < -1 || value > 1) {
            throw new Error(`Entry ${entryIndex} has an invalid value`);
        }

        entries[entryIndex] = { key, evaluation: { logits, value } };
    }
    return entries;
}

/**
 * @param {number} value - unsigned 15-bit integer
 * @returns {number}
 */
function _evalCachePopcount15(value) {
    let count = 0;
    while (value !== 0) {
        value &= value - 1;
        count++;
    }
    return count;
}
