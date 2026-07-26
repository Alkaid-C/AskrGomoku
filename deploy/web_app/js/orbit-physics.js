/**
 * Keplerian orbit math and canvas drawing shared by the main-thread fallback
 * and the OffscreenCanvas worker.
 */

globalThis.OrbitPhysics = (() => {
    const TAU = 2 * Math.PI;
    const INTERNAL_SIZE = 200;
    const FOCUS = INTERNAL_SIZE / 2;
    const MAX_APOAPSIS = 72;
    const NOMINAL_PERIODS = [3, 5, 7];
    const OUTER_SEMIMAJOR = 55;
    const ECCENTRICITY_MEAN = 0.19;
    const ECCENTRICITY_STDDEV = 0.055;
    const ECCENTRICITY_MIN = 0.07;
    const ECCENTRICITY_MAX = 0.32;
    const ENERGY_RELATIVE_STDDEV = 0.04;
    const SEMIMAJOR_JITTER = 0.08;
    const PLANET_RADII = [4, 6, 8];
    const PLANET_COLORS = ['#666', '#555', '#444'];
    const LINE_COLOR = '#999';
    const LINE_CSS_WIDTH = 0.8;

    function uniform(random, min, max) {
        return min + (max - min) * random();
    }

    function standardNormal(random) {
        let u = 0;
        let v = 0;
        while (u === 0) u = random();
        while (v === 0) v = random();
        return Math.sqrt(-2 * Math.log(u)) * Math.cos(TAU * v);
    }

    function truncatedNormal(random, mean, stddev, min, max) {
        for (let attempt = 0; attempt < 64; attempt++) {
            const value = mean + stddev * standardNormal(random);
            if (value >= min && value <= max) return value;
        }
        return Math.min(max, Math.max(min, mean));
    }

    /**
     * Generate one three-planet system. The planets share one focus and
     * gravitational parameter. Eccentricity and specific orbital energy use
     * bounded normal distributions; orientation and phase are uniform.
     */
    function createRandomSystem(random = Math.random) {
        const outerPeriod = NOMINAL_PERIODS[NOMINAL_PERIODS.length - 1];
        const mu = 4 * Math.PI ** 2 * OUTER_SEMIMAJOR ** 3 / outerPeriod ** 2;
        const orbits = [];

        for (const nominalPeriod of NOMINAL_PERIODS) {
            const nominalA = Math.cbrt(
                mu * (nominalPeriod / TAU) ** 2);

            let orbit = null;
            for (let attempt = 0; attempt < 128 && orbit === null; attempt++) {
                const eccentricity = truncatedNormal(
                    random,
                    ECCENTRICITY_MEAN,
                    ECCENTRICITY_STDDEV,
                    ECCENTRICITY_MIN,
                    ECCENTRICITY_MAX);

                const minA = nominalA * (1 - SEMIMAJOR_JITTER);
                const maxA = Math.min(
                    nominalA * (1 + SEMIMAJOR_JITTER),
                    MAX_APOAPSIS / (1 + eccentricity));
                if (maxA <= minA) continue;

                const meanEnergy = -mu / (2 * nominalA);
                const minEnergy = -mu / (2 * minA);
                const maxEnergy = -mu / (2 * maxA);
                const energy = truncatedNormal(
                    random,
                    meanEnergy,
                    Math.abs(meanEnergy) * ENERGY_RELATIVE_STDDEV,
                    minEnergy,
                    maxEnergy);
                const semiMajor = -mu / (2 * energy);

                orbit = {
                    eccentricity: eccentricity,
                    specificEnergy: energy,
                    semiMajor: semiMajor,
                    period: TAU * Math.sqrt(semiMajor ** 3 / mu),
                    periapsisAngle: uniform(random, 0, TAU),
                    meanAnomalyAtEpoch: uniform(random, 0, TAU),
                };
            }

            if (orbit === null) {
                throw new Error('Unable to sample a bounded Kepler orbit');
            }
            orbits.push(orbit);
        }

        return {
            mu: mu,
            focusX: FOCUS,
            focusY: FOCUS,
            orbits: orbits,
        };
    }

    function normalizeAngle(angle) {
        const normalized = angle % TAU;
        return normalized < 0 ? normalized + TAU : normalized;
    }

    /**
     * Solve M = E - e sin(E) for the eccentric anomaly.
     */
    function solveEccentricAnomaly(meanAnomaly, eccentricity) {
        const mean = normalizeAngle(meanAnomaly);
        let eccentric = mean;
        for (let i = 0; i < 7; i++) {
            const residual = eccentric
                - eccentricity * Math.sin(eccentric) - mean;
            eccentric -= residual
                / (1 - eccentricity * Math.cos(eccentric));
        }
        return eccentric;
    }

    /**
     * Analytic position at an absolute elapsed time. Screen y grows downward,
     * so increasing mean anomaly produces clockwise motion.
     */
    function positionAt(system, orbit, elapsedSeconds) {
        const meanAnomaly = orbit.meanAnomalyAtEpoch
            + TAU * (elapsedSeconds % orbit.period) / orbit.period;
        const eccentricAnomaly = solveEccentricAnomaly(
            meanAnomaly, orbit.eccentricity);
        const localX = orbit.semiMajor
            * (Math.cos(eccentricAnomaly) - orbit.eccentricity);
        const localY = orbit.semiMajor
            * Math.sqrt(1 - orbit.eccentricity ** 2)
            * Math.sin(eccentricAnomaly);
        const cosAngle = Math.cos(orbit.periapsisAngle);
        const sinAngle = Math.sin(orbit.periapsisAngle);
        return {
            x: system.focusX + localX * cosAngle - localY * sinAngle,
            y: system.focusY + localX * sinAngle + localY * cosAngle,
        };
    }

    function positionsAt(system, elapsedSeconds) {
        return system.orbits.map(orbit =>
            positionAt(system, orbit, elapsedSeconds));
    }

    /**
     * Draw one discrete animation frame. All geometry uses fixed internal
     * coordinates and is scaled to the canvas backing store.
     */
    function drawFrame(ctx, canvasWidth, canvasHeight, cssSize,
                       system, elapsedSeconds) {
        ctx.setTransform(1, 0, 0, 1, 0, 0);
        ctx.clearRect(0, 0, canvasWidth, canvasHeight);
        ctx.setTransform(
            canvasWidth / INTERNAL_SIZE, 0,
            0, canvasHeight / INTERNAL_SIZE, 0, 0);

        const positions = positionsAt(system, elapsedSeconds);
        ctx.beginPath();
        ctx.moveTo(positions[0].x, positions[0].y);
        ctx.lineTo(positions[1].x, positions[1].y);
        ctx.lineTo(positions[2].x, positions[2].y);
        ctx.closePath();
        ctx.strokeStyle = LINE_COLOR;
        ctx.lineWidth = LINE_CSS_WIDTH * INTERNAL_SIZE / cssSize;
        ctx.stroke();

        for (let i = 0; i < positions.length; i++) {
            ctx.beginPath();
            ctx.arc(
                positions[i].x,
                positions[i].y,
                PLANET_RADII[i],
                0,
                TAU);
            ctx.fillStyle = PLANET_COLORS[i];
            ctx.fill();
        }

        return positions;
    }

    return {
        INTERNAL_SIZE,
        createRandomSystem,
        solveEccentricAnomaly,
        positionAt,
        positionsAt,
        drawFrame,
    };
})();
