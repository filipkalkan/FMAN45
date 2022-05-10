function dldX = relu_backward(X, dldY)
    dldX = heaviside(X) .* dldY;
end
