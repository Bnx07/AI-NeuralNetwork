include("neuron.jl")

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
        throw(ArgumentError("The amount of layers doesn't match :c")) # ! The face is very important so I understand this is an error I created and not a sintax error or something like that
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

            for neuron=1:layersAmount[layer]
                for conn=1:inputsAmount
                    weights[neuron, conn] = rand() * initialWeight # TODO: Check some way of making it in a way such that if you give it -1 the range will be [-1, 1] and with 1, [0, 1]
                end
            end
            
            model.layersConnections["layer$(layer)"] = weights
        else
            weights = Array{Float64}(undef, layersAmount[layer], layersAmount[layer - 1]) # ? Array of size (layersAmount[layer], layersAmount[layer - 1])

            for neuron=1:layersAmount[layer]
                for conn=1:layersAmount[layer - 1]
                    weights[neuron, conn] = rand() * initialWeight
                end
            end

            model.layersConnections["layer$(layer)"] = weights
            
            if (layersType[layer] == "softmax")
                weights = Array{Float64}(undef, 1, layersAmount[layer]) # ? Array of size (1, layersAmount[layer])

                for neuron=1:1
                    for conn=1:layersAmount[layer]
                        weights[neuron, conn] = 1
                    end
                end

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

export createModel, calculateModel

# # * Testing

# model = createModel(2, ["sigmoid", "softmax"], [2, 2], 1)
# model = calculateModel([1, 1], model)

# println(model.layersValues[model.outputLayer])

# TODO: 1) Complete softmax function integration to model
# TODO: 2) Complete model testing
# TODO: 3) Complete SGD Optimizer