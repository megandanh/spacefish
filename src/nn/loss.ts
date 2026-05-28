export function binaryCrossEntropy(pred: number, label: number): number {
    const p = Math.max(1e-7, Math.min(1 - 1e-7, pred));
    return -(label * Math.log(p) + (1 - label) * Math.log(1 - p));
}

export function meanLoss(
    preds: number[],
    labels: number[]
): number {
    let total = 0;
    for (let i = 0; i < preds.length; i++) {
        total += binaryCrossEntropy(preds[i], labels[i]);
    }
    return total / preds.length;
}

export function binaryCrossEntropyGrad(pred: number, label: number): number {
    const p = Math.max(1e-7, Math.min(1 - 1e-7, pred));
    return (p - label) / (p * (1 - p));
}