function W = randInitializeWeights(L_in, L_out, w_init)
%RANDINITIALIZEWEIGHTS Randomly initialize the weights of a layer with L_in
%incoming connections and L_out outgoing connections
%   W = RANDINITIALIZEWEIGHTS(L_in, L_out) randomly initializes the weights 
%   of a layer with L_in incoming connections and L_out outgoing 
%   connections. 
%
%   Note that W should be set to a matrix of size(L_out, 1 + L_in) as
%   the first column of W handles the "bias" terms
%

% You need to return the following variables correctly 
W = zeros(L_out, 1 + L_in);

% Randomly initialize the weights to small values
if w_init == 0% Using epsilon values according to the expression in the ex4 of ML course
    epsilon_init = sqrt(6) / sqrt(L_in + L_out);
    W = rand(L_out, 1 + L_in) * 2 * epsilon_init - epsilon_init;
elseif w_init == 1 % Using Andew Ng's DL initialization
    W = randn(L_out, 1 + L_in) * 0.01;
elseif w_init == 2 % heuristic for ReLU units
    W = randn(L_out, 1 + L_in) * sqrt(2/L_in);
elseif w_init == 3 % Xavier - heuristic for tanh
    W = randn(L_out, 1 + L_in) * sqrt(1/(L_in + L_out));
end








% =========================================================================

end
