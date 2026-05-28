import { type Activation, type ActivationName, getActivation } from "./activations";

export interface Layer {
    weights: number[][];
    biases: number[];
    activation: Activation;
    activationName: ActivationName;

    inputs: number[];
    preAct: number[];
    outputs: number[];
}

export interface Network {
    layers: Layer[];
    layerSizes: number[];
}

function gaussianRandom(std: number): number {
    let u = 0, v = 0;
    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();
    return std * Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}

export function createNetwork(
    layerSizes: number[],
    activationName: ActivationName
): Network {
    const layers: Layer[] = [];

    for (let l = 1; l < layerSizes.length; l++) {
        const fanIn = layerSizes[l - 1];
        const fanOut = layerSizes[l];
        const activation = getActivation(
            l === layerSizes.length - 1 ? "sigmoid" : activationName
        );

        const std = Math.sqrt(2 / fanIn);

        const weights: number[][] = Array.from({ length: fanOut }, () =>
            Array.from({ length: fanIn }, () => gaussianRandom(std))
        );

        const biases: number[] = new Array(fanOut).fill(0);

        layers.push({
            weights,
            biases,
            activation,
            activationName: l === layerSizes.length - 1 ? "sigmoid" : activationName,
            inputs: new Array(fanIn).fill(0),
            preAct: new Array(fanOut).fill(0),
            outputs: new Array(fanOut).fill(0),
        });
    }

    return { layers, layerSizes };
}

export function forward(network: Network, inputs: number[]): number[] {
    let current = inputs;

    for (const layer of network.layers) {
        layer.inputs = current;
        const next: number[] = [];

        for (let n = 0; n < layer.weights.length; n++) {
            let sum = layer.biases[n];
            for (let i = 0; i < current.length; i++) {
                sum += layer.weights[n][i] * current[i];
            }
            layer.preAct[n] = sum;
            next.push(layer.activation.fn(sum));
        }

        layer.outputs = next;
        current = next;
    }

    return current;
}

export function predict(network: Network, x: number, y: number): number {
    const output = forward(network, [x, y]);
    return output[0] >= 0.5 ? 1 : 0;
}

export function accuracy(
    network: Network,
    data: { x: number; y: number; label: number }[]
): number {
    const correct = data.filter(
        (d) => predict(network, d.x, d.y) === d.label
    ).length;
    return correct / data.length;
}

export function sparsityRatio(network: Network): number {
    let total = 0;
    let zeros = 0;
    for (const layer of network.layers) {
        for (const row of layer.weights) {
            for (const w of row) {
                total++;
                if (w === 0) zeros++;
            }
        }
    }
    return zeros / total;
}

