# * Neuron functions

sigmoid(x) = return 1 / (1 + exp(-x))

relu(x) = x > 0 ? x : 0 # ? Es lo mismo que if (x > 0) return x else return 0 end

leakyRelu(x) = x > 0 ? x : 0.01 * x

# function mytanh(x)
#     top = exp(x) - exp(-x)
#     bot = exp(x) + exp(-x)
#     return top/bot
# end

linear(x) = return x

function softmax(logits)
    # expScores = Array{Float64}(undef, length(logits)) # ? Julia needs to know the exact size of the array (Or matrix) when initialized
    expScores = exp.(logits)
    totalSum = sum(expScores)
    finalScores = expScores ./ totalSum
    
    # ! Just revision code
    # totalPercentage = 0
    # for i=1:length(logits)
    #     totalPercentage += finalScores[i]
    # end
    # println(totalPercentage)

    return finalScores
end

# * Auxiliar functions

function unifyValues(values)
    total = 0

    for i=1:length(values)
        total += values[i]
    end

    return total
end

function calculateNeuron(activation, values)
    # println(activation)
    result = 0
    arrayResult = Array{Float64}(undef, length(values))

    # ? If doesnt work, try global result
    if (activation == "sigmoid")
        result = sigmoid(unifyValues(values))
    elseif (activation == "relu")
        result = relu(unifyValues(values))
    elseif (activation == "leakyRelu")
        result = leakyRelu(unifyValues(values))
    elseif (activation == "tanh")
        result = tanh(unifyValues(values))
    elseif (activation == "linear")
        result = linear(unifyValues(values))
    elseif (activation == "softmax")
        arrayResult = softmax(values)
    end

    if (activation == "softmax")
        return arrayResult
    else
        return result
    end
end

# TODO: calculateDerivate is equal to calculateNeuron but with the derivates of the functions

# * Model structure (PANIK)

# ? Example creation string -> [["input", 2], ["sigmoid", 1]]

# creationString = [["input", 2], ["sigmoid", 1]] # ' Vector is any, I dont want a type ANY because is not safe

# ? Second option 
    # ? model = new Model(2) Being that 2 the inputs amount
    # ? model.addLayer("sigmoid", 1)

# ? Third option
    # ? layersType = ["sigmoid", "relu"]
    # ? layersAmount = [2, 2]

mutable struct Model
    inputs::Array{Float64, 1} # ? Array of 1 element, of type any
    inputsAmount::Int
    layersType::Dict{String, String} # ? Dictionary of keys type string and values type string
    layersAmount::Dict{String, Int64} # ? Dictionary of keys type string and values type integer
    layersConnections::Dict{String, Array{Float64, 2}} # ? In Julia, the arrays are declared Array{T, N} being T the type and N the dimensions
    layersValues::Dict{String, Array{Float64}} # ? Dictionary of keys type string and values type Float array
    outputLayer::String
end

function createModel(inputsAmount, layersType, layersAmount, initialWeight) 
    if (length(layersType) != length(layersAmount)) 
        throw(ArgumentError("The amount of layers doesn't match :c")) # ! The face is very important so I understand this is an error I created
    end

    model = Model([], inputsAmount, Dict(), Dict(), Dict(), Dict(), "")

    for i=1:length(layersType)
        if (layersType[i] != "softmax")
            model.layersAmount["layer$(i)"] = layersAmount[i]
            model.layersType["layer$(i)"] = layersType[i]
        else
            model.layersAmount["layer$(i)"] = layersAmount[i]
            model.layersType["layer$(i)"] = "linear"
            model.layersAmount["layer$(i+1)"] = 1
            model.layersType["layer$(i+1)"] = layersType[i]
        end
    end

    for layer in 1:length(layersType) # ? It covers all layers as each layer has a layer before
        if (layersType[layer] == "softmax" && layer != length(layersType)) throw(ArgumentError("The softmax layer should be the last layer >:c")) end # ? If the layer is softmax and it is not the last one
        if (layer == 1)
            weights = Array{Float64}(undef, layersAmount[layer], inputsAmount) # ? Array of size (layersAmount[layer], inputsAmount)
            # weights = Array{Float64, inputsAmount}
            for neuron=1:layersAmount[layer]
                for conn=1:inputsAmount
                    weights[neuron, conn] = rand() * initialWeight # TODO: Check some way of making it in a way such that if you give it -1 the range will be [-1, 1] and with 1, [0, 1]
                end
            end
            # println(weights)
            model.layersConnections["layer$(layer)"] = weights
        else
            weights = Array{Float64}(undef, layersAmount[layer], layersAmount[layer - 1]) # ? Array of size (layersAmount[layer], layersAmount[layer - 1])
            # weights = Array{Float64, layersAmount[layer - 1]}
            for neuron=1:layersAmount[layer]
                for conn=1:layersAmount[layer - 1]
                    weights[neuron, conn] = rand() * initialWeight
                end
            end
            # println(weights)
            model.layersConnections["layer$(layer)"] = weights
            
            if (layersType[layer] == "softmax")
                weights = Array{Float64}(undef, 1, layersAmount[layer]) # ? Array of size (1, layersAmount[layer])
                # weights = Array{Float64, layersAmount[layer]}
                for neuron=1:1
                    for conn=1:layersAmount[layer]
                        weights[neuron, conn] = 1
                    end
                end
                # println(weights)
                model.layersConnections["layer$(layer+1)"] = weights
            end
        end
    end

    if (layersType[length(layersType)] == "softmax")
        model.outputLayer = "layer$(length(layersAmount) + 1)"
    else
        model.outputLayer = "layer$(length(layersAmount))"
    end

    return model
end

function calculateModel(inputsValues, model)
    if length(inputsValues) != model.inputsAmount
        throw(ArgumentError("The amount of inputs doesn't match >:c"))
    end
    
    cModel = deepcopy(model) # ? Realizes a deep copy of the model

    cModel.inputs = inputsValues

    for layer = 1:length(cModel.layersAmount) # ? It covers all layers as each layer has a layer before
        allNeuronsValues = Array{Float64}(undef, cModel.layersAmount["layer$(layer)"]) # ? Array of size (layersAmount[layer])
        for neuron = 1:cModel.layersAmount["layer$(layer)"] # ? For each neuron
            # println("Neuron conns: ", cModel.layersConnections["layer$(layer)"][neuron, :])
            neuronValues = Array{Float64}(undef, length(cModel.layersConnections["layer$(layer)"][neuron, :])) # ? The two dots specify that it is needed the whole row of the matrix
            if layer == 1
                for conn = 1:cModel.inputsAmount
                    neuronValues[conn] = cModel.inputs[conn] * cModel.layersConnections["layer$(layer)"][neuron, conn] # ? In Julia, you access to an element of the matrix by doing [col, row]
                end
            else 
                for conn = 1:cModel.layersAmount["layer$(layer - 1)"]
                    neuronValues[conn] = cModel.layersValues["layer$(layer - 1)"][conn] * cModel.layersConnections["layer$(layer)"][neuron, conn]
                end
            end

            if (cModel.layersType["layer$(layer)"] != "softmax")
                allNeuronsValues[neuron] = calculateNeuron(cModel.layersType["layer$(layer)"], neuronValues)
            else
                allNeuronsValues = calculateNeuron(cModel.layersType["layer$(layer)"], neuronValues)
            end
        end

        cModel.layersValues["layer$(layer)"] = allNeuronsValues
    end

    return cModel
end

# ? println(model.layersType["layer1"]) This will print "sigmoid" or the type of the layer1 in the dictionary layersType

# // Ignore this # ? I'm too confused by the sintax, continue the chat with ChatGPT https://chatgpt.com/c/c20b50a1-9722-4a91-b730-918bcc8baac6

# // Ignore this too # • I actually dont understand anything, so that's a future me problem https://stackoverflow.com/questions/56257781/how-do-we-do-classes-in-julia

# * Genetic algorithms

dummyOptimizer(model, variation, expected, obtained) = return model

function dummyMutate(oldModel, variation, expected, obtained)
    totalLayers = parse(Int, SubString(oldModel.outputLayer, 6))

    model = deepcopy(oldModel)

    for layer = 1:totalLayers
        for neuron = 1:model.layersAmount["layer$layer"]
            for conn = 1:length(model.layersConnections["layer$layer"][neuron, :])
                range = model.layersConnections["layer$layer"][neuron, conn] * variation
                actualValue = model.layersConnections["layer$layer"][neuron, conn]

                model.layersConnections["layer$layer"][neuron, conn] = rand() * 2range + (actualValue - range)
            end
        end
    end

    for layer = 1:totalLayers
        # println("Layer: ", layer)
        # println(model.layersValues["layer$layer"])
        for value = 1:model.layersAmount["layer$layer"]
            model.layersValues["layer$layer"][value] = 0
        end
        # model.layersValues["layer$layer"] = Array{Float64}(undef, model.layersAmount["layer$layer"])
        # println(model.layersValues["layer$layer"])
    end

    # println("Old model")
    # println(oldModel.layersValues)

    # println("New model")
    # println(model.layersValues)

    return model
end

# ! DISCARDED FOR THE MOMENT
# // function sgdOptimizer(oldModel, variation, expected, obtained)
    # USE MSE
    # newWeight = oldWeight * learningRate * lossFunction (MSE)
# // end

# ' Suggested optimizers
# • ADAM
# • RMSProp
# • AdaGrad
# • Momentum

# ' Best optimizer to start with: ADAM

# * Epochs

function epochs(epochsAmount, model, optimizer, variation, trainData, trainExpected)
    gModel = deepcopy(model)
    for epoch = 1:epochsAmount
        for batch = 1:length(trainData)
            excecuted = calculateModel(trainData[batch], gModel)
            obtained = excecuted.layersValues[excecuted.outputLayer]
            gModel = optimizer(gModel, variation, trainExpected[batch], obtained)
            println("Epoch $epoch, Batch $batch, Output: $obtained") # TODO: Todavia queda hacer el mutate con el optimizer
        end
    end
end

# * Testing

# model = createModel(2, ["leakyRelu", "tanh"], [2, 1], 1)
# model = createModel(2, ["sigmoid"], [2], 1)
model = createModel(2, ["sigmoid", "softmax"], [2, 2], 1)
model = calculateModel([1, 1], model)

println(model.layersValues[model.outputLayer])

# epochs(10, model, dummyMutate, 0.2, [[1,1]], [1])

# copied_model = deepcopy(model)
# println(copied_model)

# println(tanh(17))

# ? Verifies if the deep copy worked as intended
# @assert copied_model != model
# @assert copied_model.inputs == model.inputs
# @assert copied_model.inputsAmount == model.inputsAmount
# @assert copied_model.layersAmount == model.layersAmount
# @assert copied_model.layersValues == model.layersValues
# @assert copied_model.layersConnections == model.layersConnections
# @assert copied_model.layersType == model.layersType
# @assert copied_model.outputLayer == model.outputLayer

# TODO: 1) Complete softmax function integration to model
# TODO: 2) Complete model testing
# TODO: 3) Complete SGD Optimizer