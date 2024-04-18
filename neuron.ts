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

// ' Hyperparameters
// • Learning Rate: How much the weights are modified, a learning rate of 0.1 means the weight will be modified up to a 10% 
// • Batch Size: Specifies the amount of samples it analyzes before modifying the weights, should be lower than the total samples of the epoch

// ' Important Concepts
// • Gradients: Represent the magnitude and direction of the changes needed in order to minimize the loss (Incorrect cases)
// Can also be defined as the slope of the loss function with respect to model parameters. They guide the optimizer
// ? Each epoch, SGD will divide all the cases in batches, and will analyze them to see if the modifications were productive or not, in order to reduce loss
// ? The biases are all those metadata things, such as weights 

// * ---------------------------------- CONDITIONS NEEDED FOR THE PROGRAM ----------------------------------

// ' Design the structure in array
// ? For example [4, ["sigmoid", 3], ["sigmoid", 1]]
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

// * ---------------------------------- ACTUAL PROGRAMMING ----------------------------------

// ! Model structurating

// TODO: Hacer que cree el modelo según un array de datos
// ? Array de ejemplo: [["input", 4], ["sigmoid", 3], ["sigmoid", 1]]
let createModel = (layers :any[][], initialWeights: number) => {
    // FIXME: The model hasnt any weights incorporated, nor relations about from where to take the values
    let model :any = {}

    // ? Handling input layer
    model.inputs = [];
    model.inputsAmount = layers[0][1];

    model.lastLayerNeurons = 0;

    // ? Handling hidden layers
    for (let i = 1; i < layers.length - 1; i++) { // Iterate over layers except the output layer
        
        let lastLayerAmount :number;
        let layerConnections :string[] = []; // ? This are the connections with last layer, will be pushed at the end
        
        let propName = `layer${i + 1}`;
        model[propName] = [];
        model[`${propName}Type`] = layers[i][0];
        model[`${propName}Amount`] = layers[i][1];
        
        // ' This if is for calculating how many neurons are in the last layer (Respectively to this)

        if (i != 1) { // ? If it isnt the first hidden layer, the last layers neurons will be this
            lastLayerAmount = model[`layer${i}Amount`];
        } else { // ? If it is the first hidden layer, take the inputs
            lastLayerAmount = model.inputsAmount;
        }

        // ' Creating connections dynamically

        for (let j = 0; j < layers[i][1]; j++) {
            // ? Do links to all the neurons from the last layer
            
            // ? This should be something like:
            // ? For this layer, take the values model.lastLayer[0] to model.lastLayer[j]
            
            // ? The connections are directly called lastLayer[number]
            layerConnections.push(`layer${i}[${j}]`); // ? This tells which layer it is (i) and which neuron it reads (j)
        }

        model[`${propName}Connections`] = layerConnections;

        // TODO: The comments below

        // ? Could make here something that if i == 1, and for example, the layer 1 has 2 neurons, and the input 2 neurons
        // ? Then initialize the connections like [ [[inputs[0], inputs[1], [initialWeight, initialWeight]], [[inputs[0], inputs[1], [initialWeight, initialWeight]] ]

        // ? For the layers different than 1, it should be `layer${i-1}` instead of inputs, and it would look like model[`layer${i-1}`]
    }


    // ? Handling output layer
    let outputLayer = layers[layers.length - 1];
    model.output = [];
    model.outputType = outputLayer[0];
    model.outputAmount = outputLayer[1];

    return model;
}

// let initialBrain = [
//     [0, 0],
//     [0]
// ];

// const calculateNeuron = (activation: string, values: number[]) => {
//     switch (activation) {
//         case "ReLU":
//             break;
//         case "Sigmoid":
//             break;
//         case "Softmax":
//             break;
//         case "Tanh":
//             break;
//         case "Linear":
//             break;
        
//     }
// }

// let connections = [[0, 1, "multiply", "value"], [0, 2, "and", "value (NULL)"], []]

const logits = [2, 1.5, 1];
const probabilities = softmax(logits);
console.log(probabilities); // Print the resulting probabilities