export type ActivationName = "relu" | "sigmoid" | "tanh" | "linear";

export interface Activation {
    fn:  (x: number) => number;
    dfn:  (x: number) => number;
}

const relu: Activation = {
    fn:  (x) => Math.max(0, x),
    dfn:  (x) => x > 0 ? 1 : 0,
};

const sigmoid: Activation = {
    fn:  (x) => 1 / (1 + Math.exp(-x)),
    dfn:  (x) => {
        const s = 1 / (1 + Math.exp(-x));
        return s * (1 - s);
    },
};

const tanh: Activation = {
    fn:  (x) => Math.tanh(x),
    dfn: (x) => 1 - Math.tanh(x) ** 2,
};

const linear: Activation = {
    fn:  (x) => x,
    dfn: (_) => 1,
};

export const ACTIVATIONS: Record<ActivationName, Activation> = {
    relu,
    sigmoid, 
    tanh,
    linear,
};

export function getActivation(name: ActivationName): Activation {
    return ACTIVATIONS[name];
}