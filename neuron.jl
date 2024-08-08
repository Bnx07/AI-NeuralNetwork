# TODO: Neurons and auxiliar functions like calculateNeuron and unifyValues need to be moved to this file

# * Neuron functions

sigmoid(x) = return 1 / (1 + exp(-x))

relu(x) = x > 0 ? x : 0 # ? Es lo mismo que if (x > 0) return x else return 0 end

leakyRelu(x) = x > 0 ? x : 0.01 * x

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

export calculateNeuron