clear ; close all; clc

%% Initialization
fprintf('Loading data and preferences...\n');

%Load subject's data
load('BMI_S1_data');
trialsUp       = [trialsUp1;trialsUp2;trialsUp3;trialsUp4;trialsUp5];
trialsDown     = [trialsDown1;trialsDown2;trialsDown3;trialsDown4;trialsDown5];
trialsUp_len   = [trialsUp1_len;trialsUp2_len;trialsUp3_len;trialsUp4_len;trialsUp5_len];
trialsDown_len = [trialsDown1_len;trialsDown2_len;trialsDown3_len;trialsDown4_len;trialsDown5_len];

%Basic variables
preferences.Fs = 500;                        % Cognionics sampling frequency
preferences.ensemble_len = size(trialsUp,2); % number of channels we have available on our dataset

%Feature extraction variables
preferences.normalize_epochs = 0; % change to one if you want to scale each epoch from 0-1 before filtering
preferences.center_epochs = 0;    % take the mean value out of each epoch before analysing (center in 0)
preferences.detrend_epochs = 0;   % detrend each epoch before analysis
preferences.epoch_size = 2500;    % size of epochs to extract (number of datapoints @ Fs 500Hz)
preferences.window_step = 125;    % number of samples the window is shifted from the previous epoch

%DWT variables
preferences.dwt_level = 7; % number of levels of analysis of the wavelet: corresponds to number of detail and approximation coefficients
dwtmode('per','nodisplay')
preferences.wname ='db4';  % type of wavelet used in the transform. See every available wavelet type using the command 'doc wfilters'
preferences.dwt_psd = 0;   % Choose PSD features for the DWT
preferences.dwt_var = 0;   % Choose variance features for the DWT
preferences.dwt_raw = 0;   % Choose the coefficients as final features

%CSP variables
preferences.nof = 5;          % number of filters to calculate for the CSP for each condition
preferences.K = 10;           % number of most discriminant filters to choose for the FBCSP
preferences.lambda_CSP = 0.0; % penalty for non-stationarity of CSP filters
preferences.bandsize = 10;    % size of each band (Hz) for the CSP methods

%Heuristic variables
preferences.corr_threshold = .75;      % Correlation strength threshold for filter bandwidth in heuristic 1
preferences.min_bandsize = 0;          % Minimum bandwidth for filters
preferences.heur_normalize_epochs = 0; % normalize epochs when calculating heuristic
preferences.heur_shuffles = 300;       % number of shuffles for calculating heuristic 2
preferences.threshold_std = 0.1;       % -/+ threshold difference value to continue increasing filter bandwidth in heuristic 2

%LDA variables
lambda_LDA = 0.00; % regularization term to avoid overfitting

%SVM variables
C_linear = .5; % Constant C for SVM with linear kernel
C_rbf = 1;     % Constant C for SVM with radial basis function
sigma = .5;    % std of the gaussian kernel in the SVM-RBF

%NN variables
hidden_layer_size = 5;                % Number of units in the hidden layer
options = optimset('MaxIter', 200);   % Define number of iterations for the minimization function
lambda_NN = 0.02;                     % lambda regularization parameter
num_labels = 2;                       % 2 Labels: up / down


%Feature engineering methods to use
preferences.do_dwt = 0;       % Perform discrete wavelet transform for denoise and filtering
preferences.do_csp = 1;       % perform Multi-band CSP for discriminability enhancement
preferences.do_fbcsp = 0;     % perform filterbank CSP for discriminability enhancement
preferences.do_ratiovar = 0;  % Use log var ratio features like in Ramoser, 2000
preferences.do_logvar = 0;    % Simple log variance variables (use when not using CSP methods)
preferences.do_heuristic = 2; % 1- Heuristic based on Blankertz et al., 2008 | 2- Custom heuristic

%Classifiers to train
preferences.do_svmrbf = 1; % SVM with radial basis function kernel
preferences.do_svml = 1;   % SVM with linear kernel
preferences.do_lda = 1;    % LDA
preferences.do_nn = 1;     % Artificial Neural Network (1 hidden layer)

%% Data filtering and basic feature extraction

%Training data
[Data,preferences] = extractFeaturesML(preferences,trialsUp,trialsDown,trialsUp_len,trialsDown_len);

%Test data
[Data_test,preferences] = extractFeaturesML(preferences,trialsUp6,trialsDown6,trialsUp6_len,trialsDown6_len);

%% Feature engineering
fprintf('Feature engineering...\n');
if preferences.do_csp == 1
    [CSP,Feature_data] = cspTrain(preferences.epoch_size, preferences.nof, preferences.lambda_CSP, preferences.do_ratiovar, Data);
    
elseif preferences.do_fbcsp == 1
    [FBCSP,final_filterbank,Feature_data] = fbcspTrain(preferences.epoch_size, preferences.nof, preferences.lambda_CSP,Data, preferences.K, preferences.ensemble_len, preferences.do_ratiovar);
    
elseif preferences.do_logvar == 1
    Feature_data.UP = squeeze(log(var(reshape(Data.UP, preferences.epoch_size,[],size(Data.UP,2))))); 
    Feature_data.DOWN = squeeze(log(var(reshape(Data.DOWN, preferences.epoch_size,[],size(Data.DOWN,2)))));
    
elseif preferences.do_dwt == 1
    Feature_data.UP = Data.UP; 
    Feature_data.DOWN = Data.DOWN;
    
end


%% Train classifiers
fprintf('Training classifier(s)...\n\n');
if preferences.do_lda == 1
    model_lda = ldaTrain(Feature_data,lambda_LDA);
end

if preferences.do_svml == 1
    model_svm_linear = svmTrain(Feature_data, C_linear, @linearKernel, 1e-3, 5);
end

if preferences.do_svmrbf == 1
    model_svm_rbf = svmTrain(Feature_data, C_rbf, @(x1, x2) gaussianKernel(x1, x2, sigma));
end

if preferences.do_nn == 1
    model_nn = nnTrain(Feature_data,hidden_layer_size,num_labels,options,lambda_NN);
end


%% Feature engineering for test run

if preferences.do_csp == 1
    Feature_data_test = cspFeatures(Data_test,CSP,preferences.nof,preferences.epoch_size, preferences.do_ratiovar);
    
elseif preferences.do_fbcsp == 1
    Feature_data_test = fbcspFeatures(Data_test,FBCSP,final_filterbank,preferences.nof,preferences.epoch_size, preferences.do_ratiovar);
    
elseif preferences.do_logvar == 1
    Feature_data_test.UP = squeeze(log(var(reshape(Data_test.UP, preferences.epoch_size,[],size(Data_test.UP,2))))); 
    Feature_data_test.DOWN = squeeze(log(var(reshape(Data_test.DOWN, preferences.epoch_size,[],size(Data_test.DOWN,2)))));
    
elseif preferences.do_dwt == 1
    Feature_data_test.UP = Data_test.UP; 
    Feature_data_test.DOWN = Data_test.DOWN;
    
end


%% Classification testing

if preferences.do_lda == 1
    pred_test = ldaPredict(Feature_data_test,model_lda);
    y_test = [ones(size(Feature_data_test.UP,1),1); -1*ones(size(Feature_data_test.DOWN,1),1)];
    acc_test = mean(double(pred_test' == y_test)) * 100;
    fprintf('LDA test session accuracy: %.2f%%.\n',acc_test);
end

if preferences.do_svml == 1
    [pred_test,~] = svmPredict(model_svm_linear, [Feature_data_test.UP;Feature_data_test.DOWN]);
    y_test = [ones(size(Feature_data_test.UP,1),1); zeros(size(Feature_data_test.DOWN,1),1)];
    acc_test = mean(double(pred_test == y_test)) * 100;
    fprintf('SVM-L test session accuracy: %.2f%%\n',acc_test);
end

if preferences.do_svmrbf == 1
    [pred_test,~] = svmPredict(model_svm_rbf, [Feature_data_test.UP;Feature_data_test.DOWN]);
    y_test = [ones(size(Feature_data_test.UP,1),1); zeros(size(Feature_data_test.DOWN,1),1)];
    acc_test = mean(double(pred_test == y_test)) * 100;
    fprintf('SVM-RBF test session accuracy: %.2f%%\n',acc_test);
end

if preferences.do_nn == 1
    pred_test = nnPredict(model_nn,[Feature_data_test.UP;Feature_data_test.DOWN]);
    y_test = [ones(size(Feature_data_test.UP,1),1); 2 * ones(size(Feature_data_test.DOWN,1),1)];
    acc_test = mean(double(pred_test == y_test)) * 100;
    fprintf('Neural Network test session accuracy: %.2f%%\n',acc_test);
end













