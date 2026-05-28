import { generateData, splitData } from "./data/datasets";
import { createNetwork, forward, predict } from "./nn/network";


const { train, test } = splitData(generateData("spiral", 200, 0.1));
console.log("train:", train.length, "test:", test.length);
console.log("sample point:", train[0]);

const net = createNetwork([2, 4, 4, 1], "relu");
const output = forward(net, [0.5, -0.3]);
console.log("raw output:", output);           // e.g. [0.61]
console.log("prediction:", predict(net, 0.5, -0.3)); // 0 or 1