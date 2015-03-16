module Regression

export ridge, predict, mse, cross_validate, kfold, Kfold

include("cross_validation.jl")
include("ridge.jl")
include("mse.jl")

end
