export interface DataPoint {
    x: number;
    y: number;
    label: number;
}

export type DatasetName = "circle" | "xor" | "spiral" | "gaussian";

function gaussianRandom(mean = 0, std = 1): number {
    let u = 0, v = 0;
    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();
    return mean + std * Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}

export function shuffleData(data: DataPoint[]): DataPoint[] {
    const arr = [...data];
    for (let i = arr.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [arr[i], arr[j]] = [arr[j], arr[i]];
    }
    return arr;
}

export function splitData(data: DataPoint[], trainRatio = 0.7): {
    train: DataPoint[];
    test: DataPoint[];
} {
    const cutoff = Math.floor(data.length * trainRatio);
    return {
        train: data.slice(0, cutoff),
        test: data.slice(cutoff),
    };
}

export function generateData(
    dataset: DatasetName,
    numSamples: number,
    noise: number
): DataPoint[] {
    switch (dataset) {
        case "gaussian": return shuffleData(generateGaussian(numSamples, noise));
        case "circle": return shuffleData(generateCircle(numSamples, noise));
        case "xor": return shuffleData(generateXOR(numSamples, noise));
        case "spiral": return shuffleData(generateSpiral(numSamples, noise));
        default: {
            const _exhaustive: never = dataset;
            throw new Error(`Unknown dataset: ${_exhaustive}`);
        }
    }
}

function generateGaussian(n: number, noise: number): DataPoint[] {
    const points: DataPoint[] = [];
    for (let i = 0; i < n; i++) {
        const label = i < n / 2 ? 0 : 1;
        const cx = label === 0 ? -0.5 : 0.5;
        points.push({
            x: gaussianRandom(cx, 0.2 + noise * 0.3),
            y: gaussianRandom(0, 0.2 + noise * 0.3),
            label,
        });
    }
    return points;
}

function generateCircle(n: number, noise: number): DataPoint[] {
    const points: DataPoint[] = [];
    for (let i = 0; i < n; i++) {
        const label = i < n / 2 ? 0 : 1;
        const radius = label === 0 ? 0.3 : 0.7;
        const angle = Math.random() * 2 * Math.PI;
        points.push({
            x: radius * Math.cos(angle) + gaussianRandom(0, noise * 0.15),
            y: radius * Math.sin(angle) + gaussianRandom(0, noise * 0.15),
            label,
        });
    }
    return points;
}

function generateXOR(n: number, noise: number): DataPoint[] {
    const points: DataPoint[] = [];
    for (let i = 0; i < n; i++) {
        const x = Math.random() * 2 - 1;
        const y = Math.random() * 2 - 1;
        const label = x * y > 0 ? 1 : 0;
        points.push({
            x: x + gaussianRandom(0, noise * 0.2),
            y: y + gaussianRandom(0, noise * 0.2),
            label,
        });
    }
    return points;
}

function generateSpiral(n: number, noise: number): DataPoint[] {
    const points: DataPoint[] = [];
    const half = Math.floor(n / 2);

    for (let label = 0; label < 2; label++) {
        const count = label === 0 ? half : n - half;
        for (let i = 0; i < count; i++) {
            const t = (i / count) * 3 * Math.PI;
            const r = t / (3 * Math.PI);
            const offset = label === 0 ? 0 : Math.PI;
            points.push({
                x: r * Math.cos(t + offset) + gaussianRandom(0, noise * 0.15),
                y: r * Math.sin(t + offset) + gaussianRandom(0, noise * 0.15),
                label,
            });
        }
    }
    return points;
}



