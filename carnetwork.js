// function sigmoid(x) {
//     return 1 / (1 + Math.exp(-x));
// }

// class NeuralNetwork {
//     constructor() {
//         // Definir el tamaño de las capas
//         this.inputSize = 10; // Neuronas de entrada
//         this.hidden1Size = 3; // Neuronas de la capa oculta 1
//         this.hidden2Size = 2; // Neuronas de la capa oculta 2
//         this.outputSize = 2; // Neuronas de salida

//         // Inicializar pesos y sesgos
//         this.weights1 = []; // Pesos entre capa de entrada y capa oculta 1
//         this.bias1 = []; // Sesgos de la capa oculta 1
//         this.weights2 = []; // Pesos entre capa oculta 1 y capa oculta 2
//         this.bias2 = []; // Sesgos de la capa oculta 2
//         this.weightsOut = []; // Pesos entre capa oculta 2 y capa de salida
//         this.biasOut = []; // Sesgos de la capa de salida

//         // Inicializar pesos y sesgos aleatorios
//         this.initializeWeightsAndBiases();
//     }

//     initializeWeightsAndBiases() {
//         // Inicializar pesos y sesgos aleatorios para cada conexión
//         // Aquí deberías implementar tu lógica para inicializar los pesos y sesgos aleatorios
//         // Por simplicidad, inicializaremos con valores aleatorios entre -1 y 1
//         this.weights1 = [...Array(this.hidden1Size)].map(() => Array.from({ length: this.inputSize }, () => Math.random() * 2 - 1));
//         this.bias1 = Array.from({ length: this.hidden1Size }, () => Math.random() * 2 - 1);
//         this.weights2 = [...Array(this.hidden2Size)].map(() => Array.from({ length: this.hidden1Size }, () => Math.random() * 2 - 1));
//         this.bias2 = Array.from({ length: this.hidden2Size }, () => Math.random() * 2 - 1);
//         this.weightsOut = [...Array(this.outputSize)].map(() => Array.from({ length: this.hidden2Size }, () => Math.random() * 2 - 1));
//         this.biasOut = Array.from({ length: this.outputSize }, () => Math.random() * 2 - 1);
//     }

//     feedforward(inputData) {
//         // Capa oculta 1
//         const hidden1 = this.computeLayer(inputData, this.weights1, this.bias1);

//         // Capa oculta 2
//         const hidden2 = this.computeLayer(hidden1, this.weights2, this.bias2);

//         // Capa de salida
//         const output = this.computeLayer(hidden2, this.weightsOut, this.biasOut);

//         return output;
//     }

//     computeLayer(inputData, weights, biases) {
//         // Calcular la salida de una capa dada la entrada, pesos y sesgos
//         return weights.map((neuronWeights, i) => {
//             const weightedSum = neuronWeights.reduce((sum, weight, j) => sum + weight * inputData[j], 0);
//             return sigmoid(weightedSum + biases[i]);
//         });
//     }
// }

// // Crear instancia de la red neuronal

// // Ejemplo de datos de entrada
// const inputData = [50, 55, 60, 100, 120, 121, 110, 37, 25, 20];

// // Obtener predicciones
// for (var i = 0; i < 100; i++) {
//     const neuralNet = new NeuralNetwork();
//     const outputPredicted = neuralNet.feedforward(inputData);
//     console.log("Predicción de salida:", outputPredicted);

//     // Umbral para decisión de doblar
//     const thresholdTurn = 0.5; // Umbral para considerar la decisión de giro

//     // Umbral para decisión de acelerar
//     const thresholdMove = 0.5; // Umbral para considerar la decisión de acelerar
    
//     // Traducir los valores de salida en decisiones de giro y aceleración
//     let decisionTurn = "No girar"; // Por defecto, la decisión es no girar
//     let decisionMove = "Detener"; // Por defecto, la decisión es detener
    
//     if (outputPredicted[0] > thresholdTurn) {
//         if (outputPredicted[1] > thresholdMove) {
//             decisionMove = "Avanzar";
//         }
//         decisionTurn = "Doblar a la izquierda";
//     } else if (outputPredicted[1] > thresholdTurn) {
//         if (outputPredicted[1] > thresholdMove) {
//             decisionMove = "Avanzar";
//         }
//         decisionTurn = "Doblar a la derecha";
//     } else if (outputPredicted[1] > thresholdMove) {
//         decisionMove = "Avanzar";
//     }
    
//     console.log("Decisión de giro:", decisionTurn);
//     console.log("Decisión de aceleración:", decisionMove);
// }