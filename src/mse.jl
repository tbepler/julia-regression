function mse( x, y, dim )
    return mean( ( x - y ).^ 2, dim )
end

function mse( x, y )
    return mean( ( x - y ).^ 2 )
end
