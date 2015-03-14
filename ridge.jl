include("center.jl")

function ridge( X, y, lambda )

    colmeans_X = center!( X )
    mean_y = center!( y )

    A = X'*X + diagm( fill( lambda, size(X,2) ) )
    w = A\(X'*y)

    broadcast!( +, X, X, colmeans_X )
    broadcast!( +, y, y, mean_y )

    bias = mean_y - colmeans_X * w

    return w, bias

end

function predict( w, bias, X )
    return broadcast( +, X*w, bias )
end

function ridge( X, y, lambda, kernel )

    m = size( X, 1 )
    K = Array( eltype(y), m, m )
    for i = 1:m
        for j = i:m
            K[i,j] = kernel( X[i,:], X[j,:] )
            K[j,i] = K[i,j]
        end
    end

    colmeans_K = center!( K )
    mean_y = center!( y )

    K += diagm( fill( lambda, m ) )
    w = K \ y

    broadcast!( +, y, y, mean_y )

    bias = mean_y - colmeans_K * w

    return w, bias, kernel, X

end

function predict( w, bias, kernel, X_model, X )
    m = size( X, 1 )
    n = size( X_model, 1 )
    K = Array( eltype(w), m, n )
    for i = 1:m
        for j = 1:n
            K[i,j] = kernel( X[i,:], X_model[j,:] )
        end
    end
    return broadcast( +, K*w, bias )
end

function predict( model, X )
    predict( model..., X )
end



