// // Tipos de neuronas: Bool, int (-256;255)
// // Tipos de relacion: NAND OR XOR AND IFVALOR> IFVALOR< SUMA RESTA MULTIPLICACION
// * ---------------------------------- NEURON TYPES ----------------------------------

// ? Neurons

const sigmoid = (x: number) :number => {
    return 1 / (1 + Math.exp(-x));
}

const relu = (x: number) :number => {
    return Math.max(0, x);
}

const leakyrelu = (x: number) :number => {
    return (x < 0) ? 0.01 * x : x;
}

const tanh = (x: number) :number => {
    let top :number = (Math.exp(x) - Math.exp(-x));
    let bot :number = (Math.exp(x) + Math.exp(-x));
    return top/bot;
}

const linear = (x:number) :number => {
    return x;
}

const softmax = (logits :number[]) :number[] => { // ? This is e elevated to xi divided by the sumatory of e**xn
    const expScores = logits.map(score => Math.exp(score));
    const sumExpScores = expScores.reduce((acc, val) => acc + val, 0);
    return expScores.map(expScore => expScore / sumExpScores);
}

// * ---------------------------------- DERIVATES OF NEURON TYPES ----------------------------------

const derSigmoid = (x: number) => {
    let top = -Math.exp(-x);
    let bottom = Math.pow(1+Math.exp(-x), 2);
    return top/bottom;
}

const derRelu = (x: number) => {
    if (x > 0) return 1;
    else return 0;
}

const derLeakyrelu = (x: number) => {
    if (x > 0) return 1;
    else return 0.01;
}

const derTanh = (x: number) => {
    let top = 4 * Math.exp(2*x);
    let bottom = Math.pow(Math.exp(2*x) + 1, 2);
    return top/bottom;
}

const derLinear = (x: number) => {
    return 1;
}

// TODO: SOFTMAX DERIVATE

const derSoftmax = (logits: number[]): number[] => {
    return logits
}

// * ---------------------------------- WEIGHTS ----------------------------------
// ? In case a neuron recieves values from multiple neurons, the output would be calculated the following way
// ? output = sigmoid( (x1*w1) + (x2*w2) ) being w1 and w2 the weights and x1 and x2 the values from the other neurons

// * ---------------------------------- BRAIN MODELING ----------------------------------
// ? Each neuron needs to have information about the neurons it gets the values from and their weights

// ? In an example brain with this structure

// ?    x1 - sig2
// ?       \
// ?         sig1
// ?       /
// ?    x2

// ? The sig1 neuron would have information such as [[x1, w1], [x2, w2])
// ? The sig2 neuron would have information such as [[x1, w1]]

// ? In both cases, the w1 is different, sig1's w1 could be 0.4 and sig2's w1 could be 0.25

// * ---------------------------------- OPTIMIZERS ----------------------------------
// ! Because of complexity, I'm using SGD (Stochastic Gradient Descent)

// ' How it works
// ? Optimizers will modify the weights and biases in order to make the AI more efficient

// ' How it works detailed
// ? SGD will use MSE (Mean Squared Error) to calculate the error between the expected values and the obtained ones, this is the loss function
// ? The derivate of the loss function will tell the model in which direction and which magnitude the weights must be modified
// ? Weigths will be updated in the direction of the negative gradient (The negative gradient will tell us the best direction to minimize error fast)
// ? The weight will be modified doing newWeight = oldWeight - learningRate * gradient

// ' Hyperparameters
// • Learning Rate: How much the weights are modified, a learning rate of 0.1 means the weight will be modified up to a 10% 
// • Batch Size: Specifies the amount of samples it analyzes before modifying the weights, should be lower than the total samples of the epoch
// ! For simplicity, the batch size will be 1 output

// ' Important Concepts
// • Gradients: Represent the magnitude and direction of the changes needed in order to minimize the loss (Incorrect cases)
// ? Can also be defined as the slope of the loss function with respect to model parameters. They guide the optimizer
// ? Each epoch, SGD will divide all the cases in batches, and will analyze them to see if the modifications were productive or not, in order to reduce loss
// ? The biases are all those metadata things, such as weights 

// * ---------------------------------- CONDITIONS NEEDED FOR THE PROGRAM ----------------------------------

// ' Design the structure in array
// ? For example [["input", 4], ["sigmoid", 3], ["sigmoid", 1]]
// ? That example would be:
// ? Input layer:       4
// ? Hidden layer 1:    3
// ? Output layer:      1

// ' Indicate the epochs
// ? When creating the AI, it would be necessary to specify the epochs or iterations

// ' Give tagged information
// ? When creating the AI, it would be necessary to indicate the inputs values and the expected outputs
// ? The structure could be: 
// ? model.inputdata = [[1, 1, 1, 1], [2,2,2,2]]
// ? model.outputdata = [[4], [8]]
// ? This would be translated as the case with four 1 as inputs has an output that is 4, the case with four 2 as inputs has an output that is 8

// ! ---------------------------------- ACTUAL PROGRAMMING ----------------------------------

// * ---------------------------------- AUXILIAR FUNCTIONS ----------------------------------

const unifyValues = (values: number[]) => {
    let value = 0;

    for (let i = 0; i < values.length; i++) {
        value += values[i];
    }

    return value;
}

const calculateNeuronValue = (activation: string, values: number[]) :number | number[] => {
    // FIXME: console.log(activation)
    // FIXME: console.log(values)
    let result;
    switch (activation) {
        case "relu":
            result = relu(unifyValues(values));
            break;
        case "leakyrelu":
            result = leakyrelu(unifyValues(values))
            break;
        case "sigmoid":
            result = sigmoid(unifyValues(values));
            break;
        case "softmax": // ! I dont know how to do with this, softmax recieves an array instead of a value
            // TODO: Softmax isnt done correctly, maybe if the outputLayer is softmax, I could do a hidden linear layer and then softmax
            // • That would seem like [["input", 2], ["softmax", 2]] but in reality is [["input", 2], ["linear", 2], ["softmax", 1]]
            result = softmax(values);
            break;
        case "tanh":
            result = tanh(unifyValues(values))
            break;
        case "linear":
            result = linear(unifyValues(values))
            break;
    }
    // FIXME: console.log("Neuron result: ", result)
    return result;
}

const calculateDerivateValue = (activation: string, values: number[]) :number => {
    let result;
    switch (activation) {
        case "relu":
            result = derRelu(unifyValues(values));
            break;
        case "leakyrelu":
            result = derLeakyrelu(unifyValues(values));
            break;
        case "sigmoid":
            result = derSigmoid(unifyValues(values));
            break;
        case "softmax": // ! I dont know how to do with this, softmax recieves an array instead of a value
            //  TODO: result = derSoftmax(values);
            break;
        case "tanh":
            result = derTanh(unifyValues(values))
            break;
        case "linear":
            result = derLinear(unifyValues(values))
            break;
    }
    // FIXME: console.log("Neuron result: ", result)
    return result;
}

// * ---------------------------------- MODEL STRUCTURE ----------------------------------

// ? Array de ejemplo: [["input", 4], ["sigmoid", 3], ["sigmoid", 1]]
let createModel = (layers :any[][], initialWeightValue :number) => {
    // ! The initialWeightValue is used to create a range between initialWeightValue and -initialWeightValue
    // FIXME: The model hasnt any weights incorporated, nor relations about from where to take the values
    let model :any = {}

    // ? Handling input layer
    model.inputs = [];
    model.inputsAmount = layers[0][1];

    model.lastLayerNeurons = 0;

    // ? Handling hidden layers
    for (let i = 1; i < layers.length; i++) { // ? Iterate over layers except the input one (Thats why it starts as 1)
        
        let lastLayerAmount :number;
        let layerConnections :number[][] = []; // ? This are the connections with last layer, will be pushed at the end
        // FIXME: layerConnections MIGHT need to be 2D
        
        let propName = `layer${i}`; // ? The layer 0 is the input one
        model[propName] = [];
        model[`${propName}Type`] = layers[i][0];
        model[`${propName}Amount`] = layers[i][1];
        model[`${propName}Connections`] = [];
        
        // ' This if is for calculating how many neurons are in the last layer (Respectively to this)

        if (i != 1) { // ? If it isnt the first hidden layer, take the last hidden layer as inputs
            lastLayerAmount = model[`layer${i-1}Amount`]; // ? This is the amount of neurons

            model[`${propName}LastLayer`] = `layer${i - 1}` // ? This is the name of the last layer
        } else { // ? If it is the first hidden layer, take the inputs

            lastLayerAmount = model.inputsAmount;// ? This is the amount of inputs
            
            model[`${propName}LastLayer`] = "inputs"; // ? This is the name of the last layer
        }

        // ' Creating connections dynamically with initial weights

        // ? For creating the initial weights its necessary to have two things
        // • How many neurons are there in the current layer
        // • How many neurons are in the last layer
        // ? After having those two things, its necessary to map each neuron with ALL the neurons in the last layer

        for (let n = 0; n < layers[i][1]; n++) { // ? N because of neuron
            layerConnections[n] = [];
            for (let l = 0; l < lastLayerAmount; l++) { // ? L for LastLayerNeuron

  	            let weight = Math.random() * (2 * initialWeightValue) - initialWeightValue;
                layerConnections[n][l] = weight;
                // ? This makes in the neuron n, a connection with the last layer neuron l and initializes the weight as initialWeight
            }
        }

        model[`${propName}Connections`] = layerConnections;
    }

    // ? Specifying which is the outputLayer

    model.outputLayer = `layer${layers.length - 1}`

    console.log("Model initialized");

    return model;
}

// * ---------------------------------- MODEL CALCULATIONS ---------------------------------- Pensar qué nombre poner, es que lea las distintas cosas para funcionar

const calculateNeurons = (model :any, inputs :number[]) => {
    if (!model.inputsAmount || !model.outputLayer || model.inputsAmount != inputs.length) { // ? Basic verifications
        return false;
    }

    let gModel = JSON.parse(JSON.stringify(model)) // ? This is for not modifying the model itself

    gModel.inputs = inputs;

    let totalLayers = Number(model.outputLayer.slice(5))

    for (let layers = 1; layers <= totalLayers; layers ++) { // ? Iterates as many times as layers exist
        let layerNeurons = gModel[`layer${layers}Amount`];
        let lastLayerNeurons;
        let layerTotalInputs :number[][] = []; // ? This variable is used for storing the total neuron input of all neurons 

        if (layers != 1) { // ? If this is not the first hidden layer, the last layer amount of neurons is defined as
            lastLayerNeurons = `layer${layers - 1}`;
        } else { // ? If it is the first hidden layer, the amount is gModel.inputsAmount
            lastLayerNeurons = "inputs";
        }

        // FIXME: console.log("\n-------- NEW LAYER --------\n")

        // FIXME: console.log("Layer neurons: ", layerNeurons);
        for (let neuron = 0; neuron < layerNeurons; neuron ++) { // ? For each neuron in the layer
            // FIXME: console.log("\nNeuron number ", neuron)
            let totalNeuronInput :number[]= []; // ? Total values of the neuron, they are stored as an array because softmax requires an array, not a number

            let neuronConnections = gModel[`layer${layers}Connections`][neuron]
            // FIXME: console.log("Neuron connections: ", neuronConnections);
            // FIXME: console.log("LastLayerNeurons", lastLayerNeurons);
            
            // ? lln stands for last layer neuron
            for (let lln = 0; lln <= Number(gModel[`${lastLayerNeurons}Amount`]) - 1; lln ++) { // ? For each last layer neuron
                // FIXME: console.log("Input value: ", gModel[`${lastLayerNeurons}`][lln])
                // FIXME: console.log("Input weight: ", neuronConnections[lln]);
                totalNeuronInput.push(gModel[`${lastLayerNeurons}`][lln] * neuronConnections[lln]);
                // ? Adds to the neuron total input the lln value multiplied by the weight
            }

            // ? At this point, the neuron has its total input

            layerTotalInputs.push(totalNeuronInput);

            // FIXME: console.log("Neuron total input: ", totalNeuronInput);
        }

        // FIXME: console.log("Layer total inputs: ", layerTotalInputs);

        for (let tiv = 0; tiv < layerTotalInputs.length; tiv++) {
            gModel[`layer${layers}`].push(calculateNeuronValue(gModel[`layer${layers}Type`], layerTotalInputs[tiv]))
        }

        // ? At this point, the layer has all the input values and is ready for utilizing its activation function

        // ! Do everything else
    }
    // * The process should be like:
    // • Create a duplicated model, from now refered as gModel
    // • Set the gModel.inputs as the function inputs
    // • Excecute a while (there are more layers) {
        // • For each layerNeuron {
            // • let totalSum = 0
            // • For each neuronConnection {
                // • totalSum += lastLayerNeurons[number] * weight
            // • }
            // ? Once all the weights for the neuron were calculated, do neuronFunction(totalSum)
            // ? Then push that result to the neuronValue
            // • 
        // • }
    // • }

    return gModel;
}

// * ---------------------------------- WEIGHTS VARIATIONS (Random) ----------------------------------

let mutateBrain = (model: any, randomRate: number, expectedResults = [], obtainedResults = []) => {
    // ? Clones the model
    let mutatedModel = JSON.parse(JSON.stringify(model));

    // ? Iterates each layer
    for (let i = 1; i <= Number(mutatedModel.outputLayer.slice(5)); i++) {
        let layerConnections = mutatedModel[`layer${i}Connections`];
        
        // ? Iterates each neuron
        for (let neuron = 0; neuron < layerConnections.length; neuron++) {
            // ? Iterates each connection
            for (let conn = 0; conn < layerConnections[neuron].length; conn++) {
                // ? Mutates the weight
                let weight = layerConnections[neuron][conn];
                let mutation = (Math.random() * 2 * randomRate) - randomRate;
                layerConnections[neuron][conn] = weight + mutation;
            }
        }
    }

    return mutatedModel;
};

let dummyMutate = (model: any, randomRate: any, expectedResults = [], obtainedResults = []) => {
    return model
}

// * ---------------------------------- OPTIMIZER ----------------------------------

const meanSquaredError = (obtainedResults: number[], expectedResults: number[]) => {
    // ? Sumatory of (expected - obtained) squared and divided by the amount of values
    let total = 0;
    for (let i = 0; i < obtainedResults.length; i++) {
        total += Math.pow((expectedResults[i] - obtainedResults[i]), 2);
    }
    return total/expectedResults.length;
}

const derMeanSquaredError = (obtainedResults: number[], expectedResults: number[]) => {
    let gradient: number[] = [];
    for (let i = 0; i < obtainedResults.length; i++) {
        gradient.push(2 * (obtainedResults[i] - expectedResults[i]) / obtainedResults.length);
    }
    return gradient;
}

const sgdOptimizer = (model: any, obtained: number[], expected: number[], learningRate) => {
    // ? Copies the model
    let newModel = JSON.parse(JSON.stringify(model));

    // ? Obtains the output layer
    let outputLayer = model.outputLayer;
    let totalLayers = Number(outputLayer.slice(5));

    // ? Calculates the error and the derivate of the MSE
    let errors: number[] = [];
    for (let i = 0; i < obtained.length; i++) {
        errors.push(expected[i] - obtained[i]);
    }    
    let mseGradient = derMeanSquaredError(obtained, expected);

    // ? Output layer adjustment

    for (let neuron = 0; neuron < model[`${outputLayer}Amount`]; neuron++) { // ? For each neuron
        for (let conn = 0; conn < model[`${outputLayer}}Connections`].length; conn++) { // ? For each connection
            let currentWeight = model[`${outputLayer}Connections`][conn];

            let newWeight = currentWeight - errors[conn] * mseGradient[conn];

            newModel[`${outputLayer}Connections`][conn] = newWeight;
        }
    }

    // ? Hidden layers adjustment
    for (let layer = totalLayers - 1; layer >= 1; layer--) { // ? Will do all hidden layers except input
        let layerType = model[`layer${layer}Type`];
        let lastLayer = model[`layer${layer}LastLayer`];
        let layerConnections = model[`layer${layer}Connections`];
        console.log("Layer type: ", layerType)
        console.log("Last layer: ", lastLayer)

        for (let neuron = 0; neuron < model[`layer${layer}Amount`]; neuron++) { // ? For each neuron
            for (let conn = 0; conn < layerConnections[neuron].length; conn++) { // ? For each connection
                
                // ? Calculates gradient
                let grad = errors[neuron] * calculateDerivateValue(layerType, [model[lastLayer][conn]])

                // ? Calculates new weight
                let newWeight = model[`layer${layer}Connections`][neuron][conn] - learningRate * grad;

                // ? Saves new weight
                newModel[`layer${layer}Connections`][neuron][conn] = newWeight;
            }
        }

    }

    return newModel
}

// * ---------------------------------- EPOCHS ----------------------------------

let epochs = (epochsAmount: number, model: object, trainData: number[][], trainResults: number[][], optimizer, mutationRate: number = 0.1, learningRate: number = 0.01) => {
    let actualModel = model;
    
    for (let i = 1; i < epochsAmount + 1; i++) {
        let outputs: number[][] = [];
        
        for (let j = 0; j < trainData.length; j++) {
            
            let trainModel = JSON.parse(JSON.stringify(actualModel));
            trainModel.inputs = trainData[j];

            let gModel = calculateNeurons(trainModel, trainData[j]);

            outputs.push(gModel[gModel.outputLayer]);
        }

        console.log(`Epoch ${i}: `);
        console.log(outputs)

        for (let out = 0; out < outputs.length; out++) {
            actualModel = JSON.parse(JSON.stringify(sgdOptimizer(actualModel, outputs[out], trainResults[out], learningRate)));
        }
        // actualModel = JSON.parse(JSON.stringify(dummyMutate(actualModel, mutationRate)));
    }
}


// * ---------------------------------- DECLARATIONS ----------------------------------

// let brainTest = createModel([["input", 2], ["relu", 2], ["sigmoid", 1]], 1)

// let result = calculateNeurons(brainTest, [1, 0]);

// let mutatedBrainTest = mutateBrain(brainTest, 0.1);

// let trainData = [[1, 1], [1, 0], [0, 1], [0, 0]]
// let trainResults = [[0], [0], [0], [1]]

// epochs(5, brainTest, trainData, trainResults, mutateBrain, 300)

let brainTest = createModel([["input", 2], ["relu", 2], ["sigmoid", 1]], 1);

console.log(brainTest)

let trainData = [[1, 1], [1, 0], [0, 1], [0, 0]];
let trainResults = [[1], [1], [1], [0]];

// epochs(70, brainTest, trainData, trainResults, sgdOptimizer, 0.1, 0.01);