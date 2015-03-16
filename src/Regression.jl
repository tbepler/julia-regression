module Regression
include("cross_validation.jl")
include("ridge.jl")

export ridge
export cross_validate
export kfold
export Kfold

end
