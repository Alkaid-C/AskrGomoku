/**
 * Reusable Keplerian orbital animation.
 *
 * The preferred path transfers a canvas to one shared worker, which solves
 * and draws discrete frames without touching the page's main thread. A
 * main-thread canvas fallback preserves functionality where OffscreenCanvas
 * transfer is unavailable.
 */

let _orbitLoaderCounter = 0;
let _orbitWorker = null;
let _orbitWorkerFailed = false;
const _orbitInstances = new Map();
const _sharedOrbitSystem = OrbitPhysics.createRandomSystem();

function createOrbitCanvas(container, size) {
    container.innerHTML = '';
    const canvas = document.createElement('canvas');
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    canvas.className = 'orbit-canvas';
    canvas.width = Math.max(1, Math.round(size * dpr));
    canvas.height = Math.max(1, Math.round(size * dpr));
    canvas.style.width = size + 'px';
    canvas.style.height = size + 'px';
    canvas.setAttribute('aria-hidden', 'true');
    container.appendChild(canvas);
    return canvas;
}

function createMainThreadFallback(container, { size, fps, system }) {
    const canvas = createOrbitCanvas(container, size);
    const ctx = canvas.getContext('2d');
    let running = false;
    let timerId = null;
    let startTime = 0;
    let lastFrameIndex = -1;

    function tick() {
        if (!running) return;
        const interval = 1000 / fps;
        const now = performance.now();
        const frameIndex = Math.floor((now - startTime) / interval);
        if (frameIndex !== lastFrameIndex) {
            OrbitPhysics.drawFrame(
                ctx, canvas.width, canvas.height, size,
                system, frameIndex / fps);
            lastFrameIndex = frameIndex;
        }
        const nextDue = startTime + (frameIndex + 1) * interval;
        timerId = setTimeout(tick, Math.max(0, nextDue - performance.now()));
    }

    function start() {
        if (running) return;
        running = true;
        startTime = performance.now();
        lastFrameIndex = -1;
        tick();
    }

    function stop() {
        running = false;
        clearTimeout(timerId);
        timerId = null;
    }

    function destroy() {
        stop();
        container.innerHTML = '';
    }

    return { start, stop, destroy };
}

function fallBackInstance(state, reason) {
    if (state.fallback || state.destroyed) return;
    console.warn('Orbit worker unavailable; using main-thread fallback:', reason);
    state.fallback = createMainThreadFallback(state.container, {
        size: state.size,
        fps: state.fps,
        system: state.system,
    });
    if (state.running) state.fallback.start();
}

function failOrbitWorker(reason) {
    if (_orbitWorker) _orbitWorker.terminate();
    _orbitWorker = null;
    _orbitWorkerFailed = true;
    for (const state of _orbitInstances.values()) {
        fallBackInstance(state, reason);
    }
}

function getOrbitWorker() {
    if (_orbitWorkerFailed) return null;
    if (_orbitWorker) return _orbitWorker;

    try {
        _orbitWorker = new Worker(
            gomokuAssetUrl('js/orbit-loader-worker.js'));
        _orbitWorker.onmessage = (event) => {
            const message = event.data;
            const state = _orbitInstances.get(message.id);
            if (message.type === 'error' && state) {
                fallBackInstance(state, message.message);
            }
        };
        _orbitWorker.onerror = (event) => {
            failOrbitWorker(event.message || 'worker startup failed');
        };
        return _orbitWorker;
    } catch (error) {
        failOrbitWorker(error);
        return null;
    }
}

/**
 * Create an orbital loader inside a container.
 * @param {Element} container
 * @param {Object} [options]
 * @param {number} [options.size] - Rendered width/height in CSS pixels
 * @param {number} [options.fps] - Discrete frames per second
 * @returns {Object} {start, stop, destroy}
 */
function createOrbitLoader(container, { size = 200, fps = 60 } = {}) {
    const id = ++_orbitLoaderCounter;
    const workerSupported = typeof Worker !== 'undefined'
        && typeof HTMLCanvasElement !== 'undefined'
        && 'transferControlToOffscreen' in HTMLCanvasElement.prototype;

    if (!workerSupported || _orbitWorkerFailed) {
        return createMainThreadFallback(container, {
            size: size,
            fps: fps,
            system: _sharedOrbitSystem,
        });
    }

    const canvas = createOrbitCanvas(container, size);
    const state = {
        id: id,
        container: container,
        size: size,
        fps: fps,
        system: _sharedOrbitSystem,
        running: false,
        destroyed: false,
        fallback: null,
    };
    _orbitInstances.set(id, state);

    const worker = getOrbitWorker();
    if (!worker) {
        _orbitInstances.delete(id);
        return createMainThreadFallback(container, {
            size: size,
            fps: fps,
            system: _sharedOrbitSystem,
        });
    }

    try {
        const offscreen = canvas.transferControlToOffscreen();
        worker.postMessage({
            type: 'create',
            id: id,
            canvas: offscreen,
            cssSize: size,
            fps: fps,
            system: state.system,
        }, [offscreen]);
    } catch (error) {
        _orbitInstances.delete(id);
        return createMainThreadFallback(container, {
            size: size,
            fps: fps,
            system: _sharedOrbitSystem,
        });
    }

    function start() {
        if (state.running || state.destroyed) return;
        state.running = true;
        if (state.fallback) {
            state.fallback.start();
        } else {
            worker.postMessage({ type: 'start', id: id });
        }
    }

    function stop() {
        if (!state.running || state.destroyed) return;
        state.running = false;
        if (state.fallback) {
            state.fallback.stop();
        } else {
            worker.postMessage({ type: 'stop', id: id });
        }
    }

    function destroy() {
        if (state.destroyed) return;
        state.destroyed = true;
        state.running = false;
        if (state.fallback) {
            state.fallback.destroy();
        } else {
            worker.postMessage({ type: 'destroy', id: id });
            container.innerHTML = '';
        }
        _orbitInstances.delete(id);
    }

    return { start, stop, destroy };
}
