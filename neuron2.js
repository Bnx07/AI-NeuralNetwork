// * ---------------------------------- NEURON TYPES ----------------------------------
var sigmoid = function (x) {
    return 1 / (1 + Math.exp(-x));
};
var relu = function (x) {
    return Math.max(0, x);
};
var leakyrelu = function (x) {
    return x < 0 ? 0.01 * x : x;
};
var tanh = function (x) {
    var top = Math.exp(x) - Math.exp(-x);
    var bot = Math.exp(x) + Math.exp(-x);
    return top / bot;
};
var linear = function (x) {
    return x;
};
var softmax = function (logits) {
    var expScores = logits.map(function (score) { return Math.exp(score); });
    var sumExpScores = expScores.reduce(function (acc, val) { return acc + val; }, 0);
    return expScores.map(function (expScore) { return expScore / sumExpScores; });
};
// * ---------------------------------- DERIVATES OF NEURON TYPES ----------------------------------
var derSigmoid = function (x) {
    var sig = sigmoid(x);
    return sig * (1 - sig);
};
var derRelu = function (x) {
    return x > 0 ? 1 : 0;
};
var derLeakyrelu = function (x) {
    return x > 0 ? 1 : 0.01;
};
var derTanh = function (x) {
    return 1 - Math.pow(tanh(x), 2);
};
var derLinear = function (x) {
    return 1;
};
// * ---------------------------------- MODEL STRUCTURE ----------------------------------
var createModel = function (layers, initialWeightValue) {
    if (initialWeightValue === void 0) { initialWeightValue = 1; }
    var model = {};
    model.inputs = [];
    model.inputsAmount = layers[0][1];
    for (var i = 1; i < layers.length; i++) {
        var lastLayerAmount = void 0;
        var layerConnections = [];
        var propName = "layer".concat(i);
        model[propName] = [];
        model["".concat(propName, "Type")] = layers[i][0];
        model["".concat(propName, "Amount")] = layers[i][1];
        model["".concat(propName, "Connections")] = [];
        if (i != 1) {
            lastLayerAmount = model["layer".concat(i - 1, "Amount")];
            model["".concat(propName, "LastLayer")] = "layer".concat(i - 1);
        }
        else {
            lastLayerAmount = model.inputsAmount;
            model["".concat(propName, "LastLayer")] = "inputs";
        }
        for (var n = 0; n < layers[i][1]; n++) {
            layerConnections[n] = [];
            for (var l = 0; l < lastLayerAmount; l++) {
                var weight = Math.random() * (2 * initialWeightValue) - initialWeightValue;
                layerConnections[n][l] = weight;
            }
        }
        model["".concat(propName, "Connections")] = layerConnections;
    }
    model.outputLayer = "layer".concat(layers.length - 1);
    console.log("Model initialized");
    return model;
};
// * ---------------------------------- MODEL CALCULATIONS ----------------------------------
var calculateNeurons = function (model, inputs, showData) {
    if (showData === void 0) { showData = false; }
    if (!model.inputsAmount || !model.outputLayer || model.inputsAmount != inputs.length) {
        return false;
    }
    var gModel = JSON.parse(JSON.stringify(model));
    gModel.inputs = inputs;
    var totalLayers = Number(model.outputLayer.slice(5));
    for (var layers = 1; layers <= totalLayers; layers++) {
        var layerNeurons = gModel["layer".concat(layers, "Amount")];
        var lastLayerNeurons = void 0;
        var layerTotalInputs = [];
        if (layers != 1) {
            lastLayerNeurons = "layer".concat(layers - 1);
        }
        else {
            lastLayerNeurons = "inputs";
        }
        if (showData)
            console.log("\n-------- NEW LAYER --------\n");
        if (showData)
            console.log("Layer neurons: ", layerNeurons);
        for (var neuron = 0; neuron < layerNeurons; neuron++) {
            if (showData)
                console.log("\nNeuron number ", neuron);
            var totalNeuronInput = 0; // Changed to a single number
            var neuronConnections = gModel["layer".concat(layers, "Connections")][neuron];
            if (showData)
                console.log("Neuron connections: ", neuronConnections);
            if (showData)
                console.log("LastLayerNeurons", lastLayerNeurons);
            for (var lln = 0; lln < gModel["".concat(lastLayerNeurons, "Amount")]; lln++) {
                if (showData)
                    console.log("Input value: ", gModel["".concat(lastLayerNeurons)][lln]);
                if (showData)
                    console.log("Input weight: ", neuronConnections[lln]);
                totalNeuronInput += gModel["".concat(lastLayerNeurons)][lln] * neuronConnections[lln]; // Sum of products
            }
            layerTotalInputs.push(totalNeuronInput);
            if (showData)
                console.log("Neuron total input: ", totalNeuronInput);
        }
        if (showData)
            console.log("Layer total inputs: ", layerTotalInputs);
        for (var tiv = 0; tiv < layerTotalInputs.length; tiv++) {
            gModel["layer".concat(layers)].push(calculateNeuronValue(gModel["layer".concat(layers, "Type")], layerTotalInputs[tiv]));
        }
    }
    return gModel;
};
// * ---------------------------------- OPTIMIZER ----------------------------------
var sgdOptimizer = function (model, obtained, expected, learningRate) {
    var newModel = JSON.parse(JSON.stringify(model));
    var outputLayer = model.outputLayer;
    var totalLayers = Number(outputLayer.slice(5));
    var errors = [];
    for (var i = 0; i < obtained.length; i++) {
        errors.push(expected[i] - obtained[i]);
    }
    var mseGradient = derMeanSquaredError(obtained, expected);
    for (var neuron = 0; neuron < model["".concat(outputLayer, "Amount")]; neuron++) {
        for (var conn = 0; conn < model["".concat(outputLayer, "Connections")][neuron].length; conn++) {
            var currentWeight = model["".concat(outputLayer, "Connections")][neuron][conn];
            var newWeight = currentWeight - learningRate * errors[neuron] * mseGradient[neuron];
            newModel["".concat(outputLayer, "Connections")][neuron][conn] = newWeight;
        }
    }
    for (var layer = totalLayers - 1; layer >= 1; layer--) {
        var layerType = model["layer".concat(layer, "Type")];
        var lastLayer = model["layer".concat(layer, "LastLayer")];
        var layerConnections = model["layer".concat(layer, "Connections")];
        for (var neuron = 0; neuron < model["layer".concat(layer, "Amount")]; neuron++) {
            for (var conn = 0; conn < layerConnections[neuron].length; conn++) {
                var grad = errors[neuron] * calculateDerivateValue(layerType, [model[lastLayer][conn]]);
                var newWeight = model["layer".concat(layer, "Connections")][neuron][conn] - learningRate * grad;
                newModel["".concat(layer).concat(neuron)][conn] = newWeight;
            }
        }
    }
    return newModel;
};
// * ---------------------------------- TRAINING ----------------------------------
var epochs = function (amount, model, dataset, results, optimizer, learningRate) {
    var actualModel = JSON.parse(JSON.stringify(model));
    for (var i = 0; i < amount; i++) {
        for (var x = 0; x < dataset.length; x++) {
            var predicted = calculateNeurons(actualModel, dataset[x])[actualModel.outputLayer];
            actualModel = optimizer(actualModel, predicted, results[x], learningRate);
        }
        if (i % 1000 === 0) {
            var epochError = 0;
            for (var x = 0; x < dataset.length; x++) {
                var predicted = calculateNeurons(actualModel, dataset[x])[actualModel.outputLayer];
                epochError += meanSquaredError(predicted, results[x]);
            }
            console.log("Epoch ".concat(i, ": Error ").concat(epochError / dataset.length));
        }
    }
    return actualModel;
};
// * ---------------------------------- MODEL PREDICTIONS ----------------------------------
var calculateNeuronValue = function (type, inputs) {
    var value = inputs;
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
var calculateDerivateValue = function (type, inputs) {
    var value = inputs;
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
var meanSquaredError = function (obtained, expected) {
    var errorSum = 0;
    for (var i = 0; i < obtained.length; i++) {
        errorSum += Math.pow(expected[i] - obtained[i], 2);
    }
    return errorSum / obtained.length;
};
var derMeanSquaredError = function (obtained, expected) {
    var gradients = [];
    for (var i = 0; i < obtained.length; i++) {
        gradients.push(2 * (obtained[i] - expected[i]) / obtained.length);
    }
    return gradients;
};
// * ---------------------------------- EXAMPLE EXECUTION ----------------------------------
var model = createModel([
    ["", 2],
    ["sigmoid", 2],
    ["sigmoid", 1]
], 0.5);
var dataset = [[0, 0], [0, 1], [1, 0], [1, 1]];
var results = [[0], [1], [1], [0]];
model = epochs(10000, model, dataset, results, sgdOptimizer, 0.1);
console.log("Predictions after training:");
dataset.forEach(function (data) {
    console.log("Input: ".concat(data));
    console.log("Output: ".concat(calculateNeurons(model, data)[model.outputLayer]));
});
