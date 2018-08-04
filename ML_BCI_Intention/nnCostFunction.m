function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
% Change y to a matrix of 0's and 1's
eye_matrix = eye(num_labels);
y_matrix = eye_matrix(y,:);

%FORWARD PREDICTION CALCULATION
a1 = [ones(m, 1) X];
z2 = a1 * Theta1';
a2 = sigmoid(z2); 
a2 = [ones(size(a2,1),1), a2];
z3 = a2 * Theta2';
a3 = sigmoid(z3);

%neste caso a nossa hipotese e igual a activa?ao no 3? nodulo (a3)
hx = a3;

% Compute cost according to the ML_9 equation
J = (1/m) * sum(sum(-y_matrix .* log(hx) - (1-y_matrix) .* log(1 - hx)));

%REGULARIZATION PARAMETER CALCULATION
%Remove bias unit
Theta1_m = Theta1(:,2:end);
Theta2_m = Theta2(:,2:end);

%Parameter estimation without the bias unit according to ML_9 formula
reg = (lambda/(2 * m)) * (sum(sum(Theta1_m .^ 2)) + sum(sum(Theta2_m .^ 2)));

%Cost function update
J = J + reg;

%BACKPROPAGATION
%save the error and compute the error for the output and hidden layers 
%excluding the bias unit
d3 = (a3 - y_matrix);
d2 = d3 * Theta2(:,2:end) .* sigmoidGradient(z2);

%Compute final Deltas using ML_9 equation
Delta1 = d2' * a1;
Delta2 = d3' * a2;

%Calculate the gradients from the deltas without the bias unit. We can
%replace the first column of the Theta weights with 0 to exclude the unit
%without loss of this dimension
Theta1(:,1) = 0;
Theta2(:,1) = 0;
Theta1_grad = ((1/m) * Delta1) + ((lambda/m) * Theta1);
Theta2_grad = ((1/m) * Delta2) + ((lambda/m) * Theta2);
    


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
