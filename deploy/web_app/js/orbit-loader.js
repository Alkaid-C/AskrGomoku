/**
 * Orbital Loading Animation (reusable)
 *
 * Three planets on elliptical orbits, connected by lines redrawn every
 * frame. Instantiate with a container element and a pixel size; the SVG
 * keeps a fixed 200x200 internal coordinate system and scales to the
 * requested size, so multiple instances (loading screen, in-game progress)
 * can coexist at different sizes.
 */

let _orbitLoaderCounter = 0;

/**
 * Create an orbit loader inside a container element.
 * @param {Element} container - Element to render into (content is replaced)
 * @param {Object} [options]
 * @param {number} [options.size] - Rendered width/height in CSS pixels
 * @returns {Object} {start, stop, destroy}
 */
function createOrbitLoader(container, { size = 200 } = {}) {
    // Unique per-instance IDs: animateMotion mpath references its orbit
    // path by id, which must not collide across instances.
    const uid = 'orbit-loader-' + (++_orbitLoaderCounter);

    container.innerHTML = `
        <svg width="${size}" height="${size}" viewBox="0 0 200 200">
            <g class="orbit-lines"></g>
            <g class="orbit orbit-1">
                <circle class="planet planet-1" r="4" fill="#666">
                    <animateMotion dur="3s" repeatCount="indefinite">
                        <mpath href="#${uid}-1"/>
                    </animateMotion>
                </circle>
            </g>
            <g class="orbit orbit-2">
                <circle class="planet planet-2" r="6" fill="#555">
                    <animateMotion dur="5s" repeatCount="indefinite">
                        <mpath href="#${uid}-2"/>
                    </animateMotion>
                </circle>
            </g>
            <g class="orbit orbit-3">
                <circle class="planet planet-3" r="8" fill="#444">
                    <animateMotion dur="7s" repeatCount="indefinite">
                        <mpath href="#${uid}-3"/>
                    </animateMotion>
                </circle>
            </g>
            <defs>
                <path id="${uid}-1" d="M 100,80 a 30,20 0 1,1 0,40 a 30,20 0 1,1 0,-40 z"/>
                <path id="${uid}-2" d="M 100,65 a 50,35 0 1,1 0,70 a 50,35 0 1,1 0,-70 z"/>
                <path id="${uid}-3" d="M 100,50 a 70,50 0 1,1 0,100 a 70,50 0 1,1 0,-100 z"/>
            </defs>
        </svg>`;

    const svg = container.firstElementChild;
    const linesGroup = svg.querySelector('.orbit-lines');
    const planets = [
        svg.querySelector('.planet-1'),
        svg.querySelector('.planet-2'),
        svg.querySelector('.planet-3'),
    ];

    let running = false;
    let rafId = null;
    let lines = [];

    // getCTM maps user space to viewport pixels; divide the scale back out
    // so line endpoints (drawn in user space) land on the planets when the
    // SVG is rendered at a size other than its 200x200 viewBox.
    function planetPosition(planet) {
        const ctm = planet.getCTM();
        return { x: ctm.e / ctm.a, y: ctm.f / ctm.d };
    }

    function animate() {
        if (!running) return;
        const pos = planets.map(planetPosition);
        for (let i = 0; i < 3; i++) {
            const a = pos[i];
            const b = pos[(i + 1) % 3];
            lines[i].setAttribute('x1', a.x);
            lines[i].setAttribute('y1', a.y);
            lines[i].setAttribute('x2', b.x);
            lines[i].setAttribute('y2', b.y);
        }
        rafId = requestAnimationFrame(animate);
    }

    function start() {
        if (running) return;
        running = true;
        for (let i = 0; i < 3; i++) {
            const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
            line.setAttribute('stroke', '#999');
            line.setAttribute('stroke-width', '1');
            linesGroup.appendChild(line);
            lines.push(line);
        }
        animate();
    }

    function stop() {
        running = false;
        cancelAnimationFrame(rafId);
        rafId = null;
        lines.forEach(line => line.remove());
        lines = [];
    }

    function destroy() {
        stop();
        container.innerHTML = '';
    }

    return { start, stop, destroy };
}
