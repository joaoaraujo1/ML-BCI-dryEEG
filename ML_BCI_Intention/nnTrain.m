function [model_nn] = nnTrain(Data,hidden_layer_size,num_labels,options,lambda)
%NNTRAIN - Train a Neural Network
% 
%

input_layer_size = size(Data.UP,2);
Xtrain = [Data.UP;Data.DOWN];
ytrain = [ones(size(Data.UP,1),1);2*ones(size(Data.DOWN,1),1)];

% Initialize weights
initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

% Unroll parameters into a vector to fit in the minimization function
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];

% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, ...
    input_layer_size, ...
    hidden_layer_size, ...
    num_labels, Xtrain, ytrain, lambda);

% Minimization function
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

% Obtain Theta1 and Theta2 with the correct shape back from nn_params
model_nn.Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
    hidden_layer_size, (input_layer_size + 1));
model_nn.Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
    num_labels, (hidden_layer_size + 1));

end