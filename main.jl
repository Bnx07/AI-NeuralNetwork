# TODO: All the code related to excecution and library exports need to be moved to this file
# ? This file will act as the main file of the project, where all the files converge
include("model.jl")
include("epochs.jl")
include("optimizers.jl")

testModel = createModel(2, ["sigmoid", "softmax"], [2, 2], 1)

epochs(10, testModel, dummyOptimizer, 0.1, [[0, 0], [0, 1], [1, 0], [1, 1]], [[0], [1], [1], [0]])

# ? Exporting the code as a library
# ? This will allow other programming languages to use this code
# ? The library will be named "JuliaModel"

# module JuliaModel

# using .JuliaModel.Model
# using .JuliaModel.Epochs
# using .JuliaModel.Optimizers

# export createModel, calculateModel, epochs

# end