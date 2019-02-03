function g = tanhGradient(z)
%TANHGRADIENT returns the gradient of the tanh function
%evaluated at z
%   g = TANHGRADIENT(z) computes the gradient of the tanh function
%   evaluated at z. 

g = 1 - tanh(z).^2;



end
