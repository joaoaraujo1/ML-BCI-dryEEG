function [model_nn] = nnTrain(Data,hidden_layer_size,num_labels,options,lambda,act_fun,w_init,min_fun,learn_rate,N,K)
%NNTRAIN - Train a Neural Network
% 
%

input_layer_size = size(Data.UP,2);
Xtrain = [Data.UP;Data.DOWN];
ytrain = [ones(size(Data.UP,1),1);2*ones(size(Data.DOWN,1),1)];

%% Optimization: K-fold shuffled cross-validation across multiple weight initializations

m = floor(size(Xtrain,1) / K);                             % Number of samples used for validation
Xval = nan(size(Xtrain,1) - m, size(Xtrain,2));            % Initialize validation training set
yval = nan(size(ytrain,1) - m,1);                          % Initialize validation training set labels
Xval_test = nan(m,size(Xtrain,2));                         % Initialize validation test set
yval_test = nan(m,1);                                      % Initialize validation test labels
X_shuffled = nan(size(Xtrain));                            % Initialize a shuffled Xtrain matrix
y_shuffled = nan(size(ytrain));                            % Initialize shuffled training labels
best_acc = 0;                                              % Initialize best accuracy
nn_params_final = nan((input_layer_size+1) * hidden_layer_size + ...
                           (hidden_layer_size+1) * num_labels,1); % Optimized nn params array(w+b)
training_final = 0; % signals if it is time to train the model with the best initial weights
nn_p_relu = nan;

weight_initialization = 1;
while weight_initialization <= N 
    
    new_idx = randperm(size(Xtrain,1)); % Randomize training indexes
    k_acc = nan(K,1); % Intiialize accuracies array
    param_tmp = nan; % Initialize each iteration parameters
    
    if training_final == 0
    
        nn_params = nan(size(nn_params_final)); % Initialize nn params array

        % Initialize weights
        initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size,w_init);
        initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels, w_init);

        % Unroll parameters into a vector to fit in the minimization function
        initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];
        
        for s = 1:size(new_idx,2)
            X_shuffled(s,:) = Xtrain(new_idx(s),:);
            y_shuffled(s,1) = ytrain(new_idx(s),1);
        end
        
    else
        
        initial_nn_params = nn_params_final;
        m = 0;

    end
    
    if training_final == 1
        k = 1:1;
    else
        k = 1:K;
    end
    
    for k = k % k-fold x-val
        
        % Fill in data for validation_train and validation_test matrices
        Xval_test = X_shuffled( ((k-1)*m) + 1  : m + ((k-1)*m),: );
        yval_test = y_shuffled( ((k-1)*m) + 1  : m + ((k-1)*m),: );

        Xval = X_shuffled( 1:((k-1)*m),: );
        Xval = [Xval; X_shuffled( m + 1 + ((k-1)*m):end,:)];
        yval = y_shuffled( 1:((k-1)*m),1 );
        yval = [yval; y_shuffled( m + 1 + ((k-1)*m):end,1)];

        % Initialize bias units
        b1 = zeros(hidden_layer_size,1);
        b2 = zeros(num_labels,1);
        
        nn_params = initial_nn_params;

        if min_fun == 0 % conjugate gradient minimization

            % Create "short hand" for the cost function to be minimized
            costFunction = @(p) nnCostFunction(p, ...
                input_layer_size, ...
                hidden_layer_size, ...
                num_labels, Xval, yval, lambda, act_fun,b1,b2,0);

            % Minimization function
            [nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

        elseif min_fun == 1 % gradient descent minimization

            Theta1 = initial_Theta1(:,2:end);
            Theta2 = initial_Theta2(:,2:end);

            for i = 1:options.MaxIter

                %Forward propagation, cost computation and backProp
                [J,~,grads] = nnCostFunction(nn_params, input_layer_size, hidden_layer_size,num_labels, Xval, yval, lambda, act_fun,b1,b2,1);

                %Sanity checks
                if isnan(J) %Nan exception
                    fprintf('Training stopped due to NaN exception in iteration %4i\r',i);
                    break
                end

                %Update parameters
                Theta1 = Theta1 - learn_rate * grads{1};
                Theta2 = Theta2 - learn_rate * grads{2};
                b1 = b1 - learn_rate * grads{3};
                b2 = b2 - learn_rate * grads{4};

                %Formatting operations
                params1 = [b1 Theta1]; params2 = [b2 Theta2];
                nn_params = [params1(:); params2(:)];

                %Print cost
                if rem(i,options.PrintInterval) == 0
                    fprintf('Iteration %4i | Cost: %4.6e\r', i, J);
                end

            end
        end
        
        if training_final == 0
            % Validation accuracy estimation
            val_model.Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                hidden_layer_size, (input_layer_size + 1));
            val_model.Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                num_labels, (hidden_layer_size + 1));

            pred = nnPredict(val_model,Xval_test);
            k_acc(k) = mean(double(pred == yval_test)); 
            params{k} = nn_params;
        end
    end
    
    % If we have a better mean accuracy in x-val, update our final nn
    % params
    if training_final == 0
        if mean(k_acc) > best_acc
            best_acc = mean(k_acc);
            nn_params_final = initial_nn_params;
            if act_fun >= 0 && act_fun <=5
                [~,idx] = max(k_acc);
                nn_p_relu = params{idx};
            end
            fprintf('New best initial weights - validation acc: %.2f\n', best_acc);
        end
        
    end
    
    % If we ended our validation train model with all data
    if act_fun < 0
        if weight_initialization == N && training_final == 0
            training_final = 1;
            weight_initialization = N-1;        
        end
    end
    
    weight_initialization = weight_initialization + 1;
  
end

if act_fun >= 0 && act_fun <=5
    nn_params = nn_p_relu;
end

% Obtain Theta1 and Theta2 with the correct shape back from nn_params
model_nn.Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
    hidden_layer_size, (input_layer_size + 1));
model_nn.Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
    num_labels, (hidden_layer_size + 1));

fprintf('NN training completed!\n\n');


end