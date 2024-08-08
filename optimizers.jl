# TODO: All the code related to the optimizers need to be moved to this file

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

        for value = 1:model.layersAmount["layer$layer"]
            model.layersValues["layer$layer"][value] = 0
        end
    end

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

export dummyOptimizer, dummyMutate