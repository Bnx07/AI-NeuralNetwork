# TODO: All the code related to the epochs need to be moved to this file

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

export epochs