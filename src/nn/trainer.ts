import { type Network } from "./network";
import { forward } from "./network";
import { binaryCrossEntropyGrad, meanLoss } from "./loss";
import { type DataPoint } from "../data/datasets";

export interface TrainerConfig {
    learningRate: number;
    batchSize: number;
}

export interface TrainStepResult {
    loss: number;
    predictions: number[];
}

export function trainStep(
    network: Network,
    data: DataPoint[],
    config: TrainerConfig
): TrainStepResult {

    const batch = sampleBatch(data, config.batchSize);

    const predictions: number[] = [];
    for (const point of batch) {
        const output = forward(network, [point.x, point.y]);
        predictions.push(output[0]);
    }

    const labels = batch.map((d) => d.label);
    const loss = meanLoss(predictions, labels);

    const layers = network.layers;

    const weightGrads = layers.map((l) =>
        l.weights.map((row) => new Array(row.length).fill(0))
    );
    const biasGrads = layers.map((l) => new Array(l.biases.length).fill(0));

    for (let b = 0; b < batch.length; b++) {
        // Re-run forward to populate layer caches for this point
        forward(network, [batch[b].x, batch[b].y]);

        // Output layer error signal
        let delta = [binaryCrossEntropyGrad(predictions[b], labels[b])];

        // Backprop through layers in reverse
        for (let l = layers.length - 1; l >= 0; l--) {
            const layer = layers[l];

            // Multiply by activation derivative
            delta = delta.map((d, i) => d * layer.activation.dfn(layer.preAct[i]));

            // Accumulate gradients for weights and biases
            for (let n = 0; n < layer.weights.length; n++) {
                biasGrads[l][n] += delta[n];
                for (let i = 0; i < layer.weights[n].length; i++) {
                    weightGrads[l][n][i] += delta[n] * layer.inputs[i];
                }
            }

            // Propagate delta to previous layer
            if (l > 0) {
                const prevDelta = new Array(layers[l - 1].outputs.length).fill(0);
                for (let n = 0; n < layer.weights.length; n++) {
                    for (let i = 0; i < layer.weights[n].length; i++) {
                        prevDelta[i] += layer.weights[n][i] * delta[n];
                    }
                }
                delta = prevDelta;
            }
        }
    }

    // 5. Update weights — average gradients over batch, apply learning rate
    const scale = config.learningRate / batch.length;
    for (let l = 0; l < layers.length; l++) {
        for (let n = 0; n < layers[l].weights.length; n++) {
            layers[l].biases[n] -= scale * biasGrads[l][n];
            for (let i = 0; i < layers[l].weights[n].length; i++) {
                layers[l].weights[n][i] -= scale * weightGrads[l][n][i];
            }
        }
    }

    return { loss, predictions };
}

/** Random mini-batch sample without replacement */
function sampleBatch(data: DataPoint[], size: number): DataPoint[] {
    const shuffled = [...data].sort(() => Math.random() - 0.5);
    return shuffled.slice(0, Math.min(size, data.length));
}