immutable Kfold
    indices::Vector{Int}
    k::Int
    
    function Kfold( n::Int, k::Int )
        new( randperm(n), k )
    end

end
    
function partition( xval::Kfold, i::Int )
    part = xval.indices[ i : xval.k : size(xval.indices,1) ]
    return setdiff( xval.indices, part ), part
end

length( xval::Kfold ) = xval.k
Base.start( xval::Kfold ) = 1
Base.next( xval::Kfold, i::Int ) = partition( xval, i ), i + 1
Base.done( xval::Kfold, i::Int ) = i > xval.k

function cross_validate( train_f, error_f, partitions )
    scores = Float64[]
    for (train,test) in partitions
        model = train_f( train )
        push!( scores, error_f( model, test ) )
    end
    return scores
end

function kfold( train_f, error_f, n, k )
    return cross_validate( train_f, error_f, Kfold( n, k ) )
end
