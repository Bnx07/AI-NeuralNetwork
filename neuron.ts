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

// ' What they do
// ? Optimizers will modify the weights and biases in order to make the AI more efficient

// ' How it works detailed
// ? SGD will use MSE (Mean Squared Error) to calculate the error between the expected values and the obtained ones, this is the loss function
// ? The derivate of the loss function will tell the model in which direction and which magnitude the weights must be modified
// ? Weigths will be updated in the direction of the negative gradient (The negative gradient will tell us the best direction to minimize error fast)
// ? The weight will be modified doing newWeight = oldWeight - learningRate * gradient

// ? For the rest of the network, the weights of the hidden layers will be updated based their derivates
// ? For example, sigmoid neurons will use the sigmoid derivate for calculating the gradient, and will be modified based on the next layer error
// ? In other words, the sigmoid neuron will be modified doing newWeight = oldWeight - learningRate * gradient * nextLayerNeuronError
// TODO: Search with more details to see how this works, but this is the basic idea

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

const calculateNeuronValue = (activation: string, values: number[], showResult = false) :number | number[] => {
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
    if (showResult) console.log("Neuron result: ", result)
    return result;
}

const calculateDerivateValue = (activation: string, value: number, showResult = false) :number => {

    // ! THE FUNCTION ISNT WORKING PROPERLY

    let result; // FIXME: This is supposed to be result: number | number[] but it crashes I dont know why
    switch (activation) {
        case "relu":
            result = derRelu(value);
            console.log(result)
            break;
        case "leakyrelu":
            result = derLeakyrelu(value);
            break;
        case "sigmoid":
            result = derSigmoid(value);
            console.log(result)
            break;
        case "softmax": // ! I dont know how to do with this, softmax recieves an array instead of a value
            //  TODO: result = derSoftmax(values);
            break;
        case "tanh":
            result = derTanh(value)
            break;
        case "linear":
            result = derLinear(value)
            break;
        default:
            result = 0;
    }
    if (showResult) console.log("Neuron result: ", result)
    console.log("Derivate result: ", result)
    return result;
}

// * ---------------------------------- MODEL STRUCTURE ----------------------------------

// ? Array de ejemplo: [["input", 4], ["sigmoid", 3], ["sigmoid", 1]]
let createModel = (layers :any[][], initialWeightValue :number = 1) => {
    // ! The initialWeightValue is used to create a range between initialWeightValue and -initialWeightValue
    let model :any = {}

    // ? Handling input layer
    model.inputs = [];
    model.inputsAmount = layers[0][1];

    // ? Handling hidden layers
    for (let i = 1; i < layers.length; i++) { // ? Iterate over layers except the input one (Thats why it starts as 1)
        
        let lastLayerAmount :number;
        let layerConnections :number[][] = []; // ? This are the connections with last layer, will be pushed at the end
        
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

// * ---------------------------------- MODEL CALCULATIONS ----------------------------------

const calculateModel = (model :any, inputs :number[], showData = false) => {
    if (!model.inputsAmount || !model.outputLayer || model.inputsAmount != inputs.length) { // ? Basic verifications
        return false;
    }

    let gModel = JSON.parse(JSON.stringify(model)) // ? This is for not modifying the model itself

    gModel.inputs = inputs;

    let totalLayers = Number(model.outputLayer.slice(5))

    for (let layers = 1; layers <= totalLayers; layers ++) { // ? Iterates as many times as layers exist
        let layerNeurons = gModel[`layer${layers}Amount`];
        let lastLayerNeurons: string;
        let layerTotalInputs :number[][] = []; // ? This variable is used for storing the total neuron input of all neurons 

        if (layers != 1) { // ? If this is not the first hidden layer, the last layer amount of neurons is defined as
            lastLayerNeurons = `layer${layers - 1}`;
        } else { // ? If it is the first hidden layer, the amount is gModel.inputsAmount
            lastLayerNeurons = "inputs";
        }
        
        if (showData) console.log("\n-------- NEW LAYER --------\n")

        if (showData) console.log("Layer neurons: ", layerNeurons);
        for (let neuron = 0; neuron < layerNeurons; neuron ++) { // ? For each neuron in the layer
            if (showData) console.log("\nNeuron number ", neuron)
            let totalNeuronInput :number[]= []; // ? Total values of the neuron, they are stored as an array because softmax requires an array, not a number

            let neuronConnections = gModel[`layer${layers}Connections`][neuron]
            if (showData) console.log("Neuron connections: ", neuronConnections);
            if (showData) console.log("LastLayerNeurons", lastLayerNeurons);
            
            // ? lln stands for last layer neuron
            for (let lln = 0; lln <= Number(gModel[`${lastLayerNeurons}Amount`]) - 1; lln ++) { // ? For each last layer neuron
                if (showData) console.log("Input value: ", gModel[`${lastLayerNeurons}`][lln])
                if (showData) console.log("Input weight: ", neuronConnections[lln]);
                totalNeuronInput.push(gModel[`${lastLayerNeurons}`][lln] * neuronConnections[lln]);
                // ? Adds to the neuron total input the lln value multiplied by the weight
            }

            // ? At this point, the neuron has its total input

            layerTotalInputs.push(totalNeuronInput);

            if (showData) console.log("Neuron total input: ", totalNeuronInput);
        }

        if (showData) console.log("Layer total inputs: ", layerTotalInputs);

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

// * ---------------------------------- ERROR FUNCTIONS ----------------------------------

const meanSquaredError = (obtainedResults: number[], expectedResults: number[]) => {
    // ? Sumatory of (expected - obtained) squared and divided by the amount of values
    let total = 0;
    for (let i = 0; i < obtainedResults.length; i++) {
        total += Math.pow((expectedResults[i] - obtainedResults[i]), 2);
    }
    return total/expectedResults.length;
}

const derMeanSquaredError = (obtainedResults: number[], expectedResults: number[]): number[] => {
    let gradient: number[] = [];
    let total = 0;

    for (let i = 0; i < obtainedResults.length; i++) {

        // ! Cases
        // • No gradient push is 0.5 for both 1 and 0 expected
        // • gradient.push(2 * obtained - expected) is 0.5 for both 1 and 0 expected
        // • gradient.push(2 * expected - obtained) is 0 for both 1 and 0 expected
        // • gradient.push(2 * obtained - expected / obtained.length) is 1 for both 1 and 0 expected

        // if (obtainedResults[i] > expectedResults[i]) gradient.push(2 * (obtainedResults[i] - expectedResults[i]))
        // else gradient.push(2 * (expectedResults[i] - obtainedResults[i]))

        // ' Theoretically it can also be obtained[i] - expected[i]

        if (expectedResults[i] > obtainedResults[i]) {
            gradient.push(2 * (obtainedResults[i] - expectedResults[i]) / obtainedResults.length)
        } else {
            gradient.push(2 * (expectedResults[i] - obtainedResults[i]))
        }
        // FIXME: Si obtainedResult es más chico, funciona 2 * (obtainedResults[i] - expectedResults[i])
        // FIXME: Mientras que si obtainedResult es más grande, funciona (expectedResults[i] - obtainedResults[i])
        // gradient.push(2 * (obtainedResults[i] - expectedResults[i]) / obtainedResults.length);
        // gradient.push(2 * (obtainedResults[i] - expectedResults[i]))
        // gradient.push(2 * (expectedResults[i] - obtainedResults[i]))
    }
    return gradient;
}

// * ---------------------------------- WEIGHTS VARIATIONS (Random) ----------------------------------

let mutateBrain = (model: any, randomRate: number, expectedResults = [], obtainedResults = []) => {
    let mutatedModel = JSON.parse(JSON.stringify(model)); // ? Clones the model

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

let dummyOptimizer = (model: any, randomRate: any, expectedResults = [], obtainedResults = []) => {
    return model
}

// * ---------------------------------- OPTIMIZER ----------------------------------

// const sgdOptimizer = (model: any, obtained: number[], expected: number[], learningRate: number) => {
//     // ? Copies the model
//     let newModel = JSON.parse(JSON.stringify(model));

//     // ? Obtains the output layer
//     let outputLayer = model.outputLayer;
//     let totalLayers = Number(outputLayer.slice(5));

//     // ? Calculates the error and the derivate of the MSE
//     let outputErrors: number[] = [];
//     for (let i = 0; i < obtained.length; i++) {
//         outputErrors.push(expected[i] - obtained[i]);
//     }

//     let mseGradient = derMeanSquaredError(obtained, expected);

//     // ? Output layer adjustment
    
//     for (let neuron = 0; neuron < model[`${outputLayer}Amount`]; neuron++) { // ? For each neuron
//         for (let conn = 0; conn < model[`${outputLayer}Connections`].length; conn++) { // ? For each connection
//             // ! let currentWeight = model[`${outputLayer}Connections`][conn];
//             let currentWeight = model[`${outputLayer}Connections`][neuron][conn] // ? Obtains the weight

//             // ? Original code
//             // // let newWeight = currentWeight - learningRate * errors[neuron] * mseGradient[conn]; 

//             // ? Recommended code
//             // // let newWeight = currentWeight - learningRate * outputErrors[neuron] * obtained[neuron];

//             // ? Used code
//             let newWeight = currentWeight - learningRate * outputErrors[neuron] * mseGradient[conn];

//             newModel[`${outputLayer}Connections`][neuron][conn] = newWeight;
//         }
//     }

//     let errors = outputErrors;
        
//     // ? Hidden layers adjustment
//     for (let layer = totalLayers - 1; layer >= 1; layer--) { // ? Will do all hidden layers except input and output
        
//         // * A lot of variables I will need eventually
//         let layerType = model[`layer${layer}Type`];
//         let lastLayer = model[`layer${layer}LastLayer`];
//         let layerConnections = model[`layer${layer}Connections`];
//         let nextLayer = model[`layer${layer + 1}`];
//         let nextLayerConnections = model[`layer${layer + 1}Connections`];
//         let nextLayerType = model[`layer${layer + 1}Type`];

//         // ? Initialize layer errors array with zeros
//         let layerErrors: number[] = [];
//         for (let i = 0; i < model[`layer${layer}Amount`]; i++) {
//             layerErrors.push(0);
//         }

//         // ? Calculate errors for the current hidden layer
//         for (let neuron = 0; neuron < model[`layer${layer + 1}Amount`]; neuron++) {
//             for (let conn = 0; conn < nextLayerConnections[neuron].length; conn++) {
//                 console.log(nextLayer)
//                 // FIXME: This is a possible error, the derivateValue is with the nextLayerConnection of a neuron
//                 // TODO: Maybe understanding better what is this supposed to do, its easier to fix
//                 layerErrors[conn] += errors[neuron] * nextLayer[neuron] * calculateDerivateValue(nextLayerType, nextLayerConnections[neuron]);
//                 let currentConnIndex = nextLayerConnections[neuron][conn];
//                 // FIXME: Now is with the nextLayerConnection of a neuron's connection
//                 // TODO: Maybe understanding better what is this supposed to do, its easier to fix
//                 let errorContribution = errors[neuron] * calculateDerivateValue(nextLayerType, nextLayerConnections[neuron][conn]);
//                 layerErrors[currentConnIndex] += errorContribution;
//             }
//         }

//         // ? Calculates new weight considering errors
//         for (let neuron = 0; neuron < model[`layer${layer}Amount`]; neuron++) {
//             // FIXME: Everything related to this code returns null, is completely broken
//             for (let conn = 0; conn < layerConnections[neuron].length; conn++) {
                
//                 console.log(`Layer ${layer}; LayerType ${layerType}; Neuron ${neuron}; Connection ${conn}`)
//                 let currentWeight = model[`layer${layer}Connections`][neuron][conn];

//                 // FIXME: At this point, neuron error is already null
//                 console.log(`Error ${layerErrors[neuron]}`)
//                 //console.log(`Last layer connection? ${model[`layer${layer - 1}`][conn]}`)
//                 // FIXME: calculates using its initial connections and the layer error of the neuron
//                 let newWeight = currentWeight - learningRate * layerErrors[neuron] * model[`layer${layer}Connections`][neuron][conn];

//                 console.log(`Current weight ${currentWeight}`)
//                 console.log(`New weigth ${newWeight}`)

//                 newModel[`layer${layer}Connections`][neuron][conn] = newWeight;
//             }
//         }

//         errors = layerErrors;  // Update errors for the next layer backpropagation
//     }

//     return newModel
// }

const sgdOptimizer = (model: any, obtained: number[], expected: number[], learningRate: number) => {
    let newModel = JSON.parse(JSON.stringify(model));

    // ? Obtains the output layer
    let outputLayer = model.outputLayer;
    let totalLayers = Number(outputLayer.slice(5));

    // ? Calculates the error and the derivate of the MSE
    let outputErrors: number[] = [];
    for (let i = 0; i < obtained.length; i++) {
        outputErrors.push(expected[i] - obtained[i]);
    }

    let mseGradient = derMeanSquaredError(obtained, expected);

    // ' Output layer weigths adjustment
    let outputLastLayer = model[`${outputLayer}LastLayer`]
    
    for (var neuron = 0; neuron < model[`${outputLayer}Amount`]; neuron++) { // ? For each neuron
        for (var conn = 0; conn < model[`${outputLastLayer}Amount`]; conn++) {

            let oldWeight = model[`${outputLayer}Connections`][neuron][conn]; // ? Obtains the weight the connection used to have
            let neuronGradient = mseGradient[neuron]; // ? Obtains the gradient with the derivate of MSE in relation to the neuron
            let activationDerivate = calculateDerivateValue(model[`${outputLayer}Type`], model[`${outputLayer}`][neuron]); // ? Calculates the derivate of the neuron

            let newWeight = oldWeight - learningRate * neuronGradient * activationDerivate;

            newModel[`${outputLayer}Connections`][neuron][conn] = newWeight;
        }
    }

    // ' Hidden layers

    let lastLayerErrors: number[] = outputErrors;

    for (var layer = 1; layer < totalLayers; layer++) {
        let lastLayerAmount: number = model[`layer${layer + 1}Amount`];

        let layerValues: number[] = model[`layer${layer}`];
        let layerAmount: number = model[`layer${layer}Amount`];
        let layerConnec: number[] = model[`layer${layer}Connections`];
        let layerType: string = model[`layer${layer}Type`]

        let layerErrors: number[] = []; // ? Will modify lastLayerErrors for backpropagation

        for (var neuron = 0; neuron < layerAmount; neuron++) {
            for (var conn = 0; conn < lastLayerAmount; conn++) {
                let oldWeight = layerConnec[neuron][conn];
                let neuronGradient = lastLayerErrors[conn];
                let activationDerivate = calculateDerivateValue(layerType, layerValues[neuron]);

                // TODO: ESTA INCOMPLETO PERO YA NO ME DA LA CABEZA

                let newWeight = oldWeight - neuronGradient * activationDerivate * oldWeight * layerValues[neuron]
            }
        }
    }
}

// * ---------------------------------- EPOCHS ----------------------------------

let epochs = (epochsAmount: number, model: object, trainData: number[][], trainResults: number[][], optimizer, learningRate: number = 0.01, showEpochs: boolean = false, showDetailedEpochs: boolean = true) => {
    let actualModel = model;
    
    // ? For each epoch
    for (let i = 1; i < epochsAmount + 1; i++) {
        let outputs: number[][] = [];
        
        // ? For each train data
        for (let j = 0; j < trainData.length; j++) { 
            
            let trainModel = JSON.parse(JSON.stringify(actualModel)); // ? Creates a copy of the model
            trainModel.inputs = trainData[j];

            let gModel = calculateModel(trainModel, trainData[j]); // ? Calculates the outputs based on the inputs

            outputs.push(gModel[gModel.outputLayer]); // ? Pushes the outputs to a variable for further analysis later
        }

        if (showEpochs) {
            if (showDetailedEpochs) console.log("Epoch ", i, ": ", outputs);
            else {
                // TODO: Make it calculate percentage
                // console.log(`Epoch ${i}: ${percentage}`);
                console.log(`Epoch ${i}: ${outputs}`);
            }
        } else if (!showEpochs && i == epochsAmount) console.log(`Epoch ${i}: ${outputs}`);

        // ? For each output obtained, use the optimizer to train it
        for (let out = 0; out < outputs.length; out++) { 
            actualModel = JSON.parse(JSON.stringify(optimizer(actualModel, outputs[out], trainResults[out], learningRate)));
        }
        // actualModel = JSON.parse(JSON.stringify(dummyMutate(actualModel, mutationRate)));
    }

    return actualModel
}

// * ---------------------------------- DECLARATIONS AND TESTING ----------------------------------

let brainTest = createModel([["input", 2], ["relu", 2], ["sigmoid", 1]], 1)

let simpleTrainData = [[1,1]];
let complexTrainData = [[1,1], [1,0], [0,1], [0,0]];

let simpleOutData = [[1]];
let complexOutData = [[1],[0], [0], [1]];

console.log(brainTest)

epochs(10, brainTest, simpleTrainData, simpleOutData, mutateBrain, 1, false)

epochs(10, brainTest, complexTrainData, complexOutData, mutateBrain, 1, true)
