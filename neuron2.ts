// * ---------------------------------- NEURON TYPES ----------------------------------

const sigmoid = (x: number): number => {
    return 1 / (1 + Math.exp(-x));
};

const relu = (x: number): number => {
    return Math.max(0, x);
};

const leakyrelu = (x: number): number => {
    return x < 0 ? 0.01 * x : x;
};

const tanh = (x: number): number => {
    let top: number = Math.exp(x) - Math.exp(-x);
    let bot: number = Math.exp(x) + Math.exp(-x);
    return top / bot;
};

const linear = (x: number): number => {
    return x;
};

const softmax = (logits: number[]): number[] => {
    const expScores = logits.map(score => Math.exp(score));
    const sumExpScores = expScores.reduce((acc, val) => acc + val, 0);
    return expScores.map(expScore => expScore / sumExpScores);
};

// * ---------------------------------- DERIVATES OF NEURON TYPES ----------------------------------

const derSigmoid = (x: number): number => {
    const sig = sigmoid(x);
    return sig * (1 - sig);
};

const derRelu = (x: number): number => {
    return x > 0 ? 1 : 0;
};

const derLeakyrelu = (x: number): number => {
    return x > 0 ? 1 : 0.01;
};

const derTanh = (x: number): number => {
    return 1 - Math.pow(tanh(x), 2);
};

const derLinear = (x: number): number => {
    return 1;
};

// * ---------------------------------- MODEL STRUCTURE ----------------------------------

const createModel = (layers: any[][], initialWeightValue: number = 1) => {
    let model: any = {};

    model.inputs = [];
    model.inputsAmount = layers[0][1];

    for (let i = 1; i < layers.length; i++) {
        let lastLayerAmount: number;
        let layerConnections: number[][] = [];

        let propName = `layer${i}`;
        model[propName] = [];
        model[`${propName}Type`] = layers[i][0];
        model[`${propName}Amount`] = layers[i][1];
        model[`${propName}Connections`] = [];

        if (i != 1) {
            lastLayerAmount = model[`layer${i - 1}Amount`];
            model[`${propName}LastLayer`] = `layer${i - 1}`;
        } else {
            lastLayerAmount = model.inputsAmount;
            model[`${propName}LastLayer`] = "inputs";
        }

        for (let n = 0; n < layers[i][1]; n++) {
            layerConnections[n] = [];
            for (let l = 0; l < lastLayerAmount; l++) {
                let weight = Math.random() * (2 * initialWeightValue) - initialWeightValue;
                layerConnections[n][l] = weight;
            }
        }

        model[`${propName}Connections`] = layerConnections;
    }

    model.outputLayer = `layer${layers.length - 1}`;

    console.log("Model initialized");

    return model;
};

// * ---------------------------------- MODEL CALCULATIONS ----------------------------------

const calculateNeurons = (model: any, inputs: number[], showData = false) => {
    if (!model.inputsAmount || !model.outputLayer || model.inputsAmount != inputs.length) {
        return false;
    }

    let gModel = JSON.parse(JSON.stringify(model));
    gModel.inputs = inputs;

    let totalLayers = Number(model.outputLayer.slice(5));

    for (let layers = 1; layers <= totalLayers; layers++) {
        let layerNeurons = gModel[`layer${layers}Amount`];
        let lastLayerNeurons: string;
        let layerTotalInputs: number[][] = [];

        if (layers != 1) {
            lastLayerNeurons = `layer${layers - 1}`;
        } else {
            lastLayerNeurons = "inputs";
        }

        if (showData) console.log("\n-------- NEW LAYER --------\n");
        if (showData) console.log("Layer neurons: ", layerNeurons);

        for (let neuron = 0; neuron < layerNeurons; neuron++) {
            if (showData) console.log("\nNeuron number ", neuron);
            let totalNeuronInput: number = 0;  // Changed to a single number

            let neuronConnections = gModel[`layer${layers}Connections`][neuron];
            if (showData) console.log("Neuron connections: ", neuronConnections);
            if (showData) console.log("LastLayerNeurons", lastLayerNeurons);

            for (let lln = 0; lln < gModel[`${lastLayerNeurons}Amount`]; lln++) {
                if (showData) console.log("Input value: ", gModel[`${lastLayerNeurons}`][lln]);
                if (showData) console.log("Input weight: ", neuronConnections[lln]);
                totalNeuronInput += gModel[`${lastLayerNeurons}`][lln] * neuronConnections[lln]; // Sum of products
            }

            layerTotalInputs.push(totalNeuronInput);

            if (showData) console.log("Neuron total input: ", totalNeuronInput);
        }

        if (showData) console.log("Layer total inputs: ", layerTotalInputs);

        for (let tiv = 0; tiv < layerTotalInputs.length; tiv++) {
            gModel[`layer${layers}`].push(calculateNeuronValue(gModel[`layer${layers}Type`], layerTotalInputs[tiv]));
        }
    }

    return gModel;
};

// * ---------------------------------- OPTIMIZER ----------------------------------

const sgdOptimizer = (model: any, obtained: number[], expected: number[], learningRate: number) => {
    let newModel = JSON.parse(JSON.stringify(model));

    let outputLayer = model.outputLayer;
    let totalLayers = Number(outputLayer.slice(5));

    let errors: number[] = [];
    for (let i = 0; i < obtained.length; i++) {
        errors.push(expected[i] - obtained[i]);
    }

    let mseGradient = derMeanSquaredError(obtained, expected);

    for (let neuron = 0; neuron < model[`${outputLayer}Amount`]; neuron++) {
        for (let conn = 0; conn < model[`${outputLayer}Connections`][neuron].length; conn++) {
            let currentWeight = model[`${outputLayer}Connections`][neuron][conn];
            let newWeight = currentWeight - learningRate * errors[neuron] * mseGradient[neuron];
            newModel[`${outputLayer}Connections`][neuron][conn] = newWeight;
        }
    }

    for (let layer = totalLayers - 1; layer >= 1; layer--) {
        let layerType = model[`layer${layer}Type`];
        let lastLayer = model[`layer${layer}LastLayer`];
        let layerConnections = model[`layer${layer}Connections`];

        for (let neuron = 0; neuron < model[`layer${layer}Amount`]; neuron++) {
            for (let conn = 0; conn < layerConnections[neuron].length; conn++) {
                let grad = errors[neuron] * calculateDerivateValue(layerType, [model[lastLayer][conn]]);
                let newWeight = model[`layer${layer}Connections`][neuron][conn] - learningRate * grad;
                newModel[`${layer}${neuron}`][conn] = newWeight;
            }
        }
    }

    return newModel;
};

// * ---------------------------------- TRAINING ----------------------------------

const epochs = (amount: number, model: any, dataset: number[][], results: number[][], optimizer: any, learningRate: number) => {
    let actualModel = JSON.parse(JSON.stringify(model));

    for (let i = 0; i < amount; i++) {
        for (let x = 0; x < dataset.length; x++) {
            let predicted = calculateNeurons(actualModel, dataset[x])[actualModel.outputLayer];
            actualModel = optimizer(actualModel, predicted, results[x], learningRate);
        }

        if (i % 1000 === 0) {
            let epochError = 0;
            for (let x = 0; x < dataset.length; x++) {
                let predicted = calculateNeurons(actualModel, dataset[x])[actualModel.outputLayer];
                epochError += meanSquaredError(predicted, results[x]);
            }
            console.log(`Epoch ${i}: Error ${epochError / dataset.length}`);
        }
    }

    return actualModel;
};

// * ---------------------------------- MODEL PREDICTIONS ----------------------------------

const calculateNeuronValue = (type: string, inputs: number): number => {
    let value = inputs;

    switch (type) {
        case "sigmoid":
            return sigmoid(value);
        case "relu":
            return relu(value);
        case "leakyrelu":
            return leakyrelu(value);
        case "tanh":
            return tanh(value);
        case "linear":
            return linear(value);
        default:
            return value;
    }
};

const calculateDerivateValue = (type: string, inputs: number): number => {
    let value = inputs;

    switch (type) {
        case "sigmoid":
            return derSigmoid(value);
        case "relu":
            return derRelu(value);
        case "leakyrelu":
            return derLeakyrelu(value);
        case "tanh":
            return derTanh(value);
        case "linear":
            return derLinear(value);
        default:
            return value;
    }
};

// * ---------------------------------- LOSS FUNCTION ----------------------------------

const meanSquaredError = (obtained: number[], expected: number[]): number => {
    let errorSum = 0;

    for (let i = 0; i < obtained.length; i++) {
        errorSum += Math.pow(expected[i] - obtained[i], 2);
    }

    return errorSum / obtained.length;
};

const derMeanSquaredError = (obtained: number[], expected: number[]): number[] => {
    let gradients: number[] = [];

    for (let i = 0; i < obtained.length; i++) {
        gradients.push(2 * (obtained[i] - expected[i]) / obtained.length);
    }

    return gradients;
};

// * ---------------------------------- EXAMPLE EXECUTION ----------------------------------

let model = createModel([
    ["", 2],
    ["sigmoid", 2],
    ["sigmoid", 1]
], 0.5);

let dataset = [[0, 0], [0, 1], [1, 0], [1, 1]];
let results = [[0], [1], [1], [0]];

model = epochs(10000, model, dataset, results, sgdOptimizer, 0.1);

console.log("Predictions after training:");
dataset.forEach(data => {
    console.log(`Input: ${data}`);
    console.log(`Output: ${calculateNeurons(model, data)[model.outputLayer]}`);
});
