/**
 * Shared worker for all active orbital loaders. Each instance owns one
 * transferred OffscreenCanvas and advances in discrete frames at its
 * requested rate. No per-frame work runs on the page's main thread.
 */

importScripts('orbit-physics.js?v=20260726-orbit-retune');

const instances = new Map();
let timerId = null;

function draw(instance, frameIndex) {
    const elapsedSeconds = frameIndex / instance.fps;
    OrbitPhysics.drawFrame(
        instance.ctx,
        instance.canvas.width,
        instance.canvas.height,
        instance.cssSize,
        instance.system,
        elapsedSeconds);
}

function schedule() {
    if (timerId !== null) return;
    timerId = setTimeout(tick, 0);
}

function tick() {
    timerId = null;
    const now = performance.now();
    let nextDue = Infinity;

    for (const instance of instances.values()) {
        if (!instance.running) continue;

        const interval = 1000 / instance.fps;
        const frameIndex = Math.floor((now - instance.startTime) / interval);
        if (frameIndex !== instance.lastFrameIndex) {
            draw(instance, frameIndex);
            instance.lastFrameIndex = frameIndex;
        }
        nextDue = Math.min(
            nextDue,
            instance.startTime + (frameIndex + 1) * interval);
    }

    if (nextDue !== Infinity) {
        timerId = setTimeout(tick, Math.max(0, nextDue - performance.now()));
    }
}

self.onmessage = (event) => {
    const message = event.data;
    try {
        if (message.type === 'create') {
            const ctx = message.canvas.getContext('2d', {
                alpha: true,
                desynchronized: true,
            });
            if (!ctx) throw new Error('OffscreenCanvas 2D context unavailable');
            instances.set(message.id, {
                id: message.id,
                canvas: message.canvas,
                ctx: ctx,
                cssSize: message.cssSize,
                fps: message.fps,
                system: message.system,
                running: false,
                startTime: 0,
                lastFrameIndex: -1,
            });
            return;
        }

        const instance = instances.get(message.id);
        if (!instance) return;

        if (message.type === 'start') {
            instance.running = true;
            instance.startTime = performance.now();
            instance.lastFrameIndex = -1;
            schedule();
        } else if (message.type === 'stop') {
            instance.running = false;
        } else if (message.type === 'destroy') {
            instances.delete(message.id);
        }
    } catch (error) {
        self.postMessage({
            type: 'error',
            id: message.id,
            message: String((error && error.message) || error),
        });
    }
};
