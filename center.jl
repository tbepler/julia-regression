function center!( A )
    colmeans = mean( A, 1 )
    broadcast!( -, A, A, colmeans )
    return colmeans
end

function center( A )
    return broadcast( -, A, mean( A, 1 ) )
end
