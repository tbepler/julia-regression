module Regression

export ridge, cross_validate, kfold, Kfold

include("cross_validation.jl")
include("ridge.jl")

end
