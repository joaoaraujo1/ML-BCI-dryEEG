function p = nnPredict(model_nn, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(model_nn.Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

h1 = sigmoid([ones(m, 1) X] * model_nn.Theta1');
h2 = sigmoid([ones(m, 1) h1] * model_nn.Theta2');
[~, p] = max(h2, [], 2);

% =========================================================================


end
