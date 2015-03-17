include("center.jl")
include("cross_validation.jl")
include("mse.jl")

function ridge( X, y, lambda::Number )

    colmeans_X = center!( X )
    mean_y = center!( y )

    #A = X'*X + diagm( fill( lambda, size(X,2) ) )
    #w = A\(X'*y)

    A = [ X ; fill( sqrt( lambda ), ( 1, size(X,2) ) ) ]
    B = [ y ; fill( 0, ( 1, size(y,2) ) ) ]
    w = A\B

    broadcast!( +, X, X, colmeans_X )
    broadcast!( +, y, y, mean_y )

    bias = mean_y - colmeans_X * w

    return w, bias

end

function ridge( X, y, ls, gen = Kfold( size(X,1), 5 ) )
    if Base.length( ls ) == 1
        return ridge( X, y, ls[1] )
    end
    l_errs = Array( Float64, size(ls,1) )
    for i in 1:size(ls,1)
        l = ls[i]
        train_f(is) = ridge( X[is,:], y[is,:], l )
        error_f(model,is) = mse( predict( model, X[is,:] ), y[is,:] )
        l_errs[i] = mean( cross_validate( train_f, error_f, gen ) )
    end
    l = ls[ indmin( l_errs ) ]
    return ridge( X, y, l ), l, l_errs
end

function predict( w, bias, X )
    return broadcast( +, X*w, bias )
end

function ridgeKernel( K, y, lambda )
    m = size( K, 1 )

    colmeans_K = center!( K )
    mean_y = center!( y )
    
    for i = 1:m
        K[i,i] += lambda
    end
    w = K \ y
    for i = 1:m
        K[i,i] -= lambda
    end
    
    broadcast!( +, y, y, mean_y )
    broadcast!( +, K, K, colmeans_K )

    bias = mean_y - colmeans_K * w

    return w, bias
end

function ridgeKernel!( K, y, lambda )
    m = size( K, 1 )

    colmeans_K = center!( K )
    mean_y = center!( y )
   
    for i = 1:m
        K[i,i] += lambda
    end
    w = K \ y
    
    broadcast!( +, y, y, mean_y )

    bias = mean_y - colmeans_K * w

    return w, bias
end

function predictKernel( w, bias, K )
    return broadcast( +, K*w, bias )
end

function ridge( kernel::Function, X, y, lambda::Number )

    m = size( X, 1 )
    K = Array( eltype(y), m, m )
    for i = 1:m
        for j = i:m
            K[i,j] = kernel( X[i,:], X[j,:] )
            K[j,i] = K[i,j]
        end
    end

    w,bias = ridgeKernel!( K, y, lambda )

    return w, bias, kernel, X

end

function ridge( kernel::Function, X, y, ls, gen = Kfold( size(X,1), 5 ) )
    
    if Base.length( ls ) == 1
        return ridge( X, y, ls[1], kernel )
    end

    m = size( X, 1 )
    K = Array( eltype(y), m, m )
    for i = 1:m
        for j = i:m
            K[i,j] = kernel( X[i,:], X[j,:] )
            K[j,i] = K[i,j]
        end
    end

    l_errs = Array( Float64, size(ls,1) )
    for i in 1:size(ls,1)
        l = ls[i]
        train_f(is) = ridgeKernel( K[is,is], y[is,:], l ), is
        
        function error_f(v,is)
            model, train_is = v
            yh = predictKernel( model..., K[is,train_is] )
            return mse( yh, y[is,:] )
        end

        l_errs[i] = mean( cross_validate( train_f, error_f, gen ) )
    end
    l = ls[ indmin( l_errs ) ]
    w,bias = ridgeKernel!( K, y, l )
    return (w, bias, kernel, X), l, l_errs
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
    return predictKernel( w, bias, K )
end

function predict( model, X )
    predict( model..., X )
end



