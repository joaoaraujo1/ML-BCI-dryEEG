function [CSP,Feature_data] = cspTrain( epoch_size, nof, lambda, do_ratiovar, Data, do_logvar)
%   CSP Stationary Common Spatial Patterns following Christian Koethe's 
%   implementation and the reference:
%
%   Wojciech,W, Vidaurre,C & Kawanabe,M (2011)
%
%   for non-stationarity correction
%
%   João Araújo, 2018
%

%% Train CSP

%   Get equal amount of trials for up and down conditions
X_up_train = Data.UP;
X_down_train = Data.DOWN;

if(size(X_up_train,1) > size(X_down_train,1))
    X_up_train = X_up_train(1:size(X_down_train,1),:);
else
    X_down_train = X_down_train(1:size(X_up_train,1),:);
end


%   Calculate how many data chunks you will use to estimate the
%   non-stationary features
trials_n = size(X_up_train,1)/epoch_size;
divisor_vector =1:trials_n;
divisors = divisor_vector(rem(trials_n,divisor_vector) == 0);
chunk_len = median(divisors); % choose the medium sized divisor
%chunk_len = 1;
chunk_size = epoch_size * chunk_len;


%   Initialize variables for the non-stationarity penalisation
Delta_total = trials_n/chunk_size;
Delta1 = 0;
Delta2 = 0;

%   Calculate Deltas (non-stationarities)
for d = 1:Delta_total
    cov1 = cov(X_up_train((d-1) * chunk_size + 1 : d * chunk_size, :));
    delta1 = P(cov1 - cov(X_up_train));
    Delta1 = Delta1 + delta1;

    cov2 = cov(X_down_train((d-1) * chunk_size + 1 : d * chunk_size, :));
    delta2 = P(cov2 - cov(X_down_train));
    Delta2 = Delta2 + delta2;
end

Delta1 = Delta1 / Delta_total;
Delta2 = Delta2 / Delta_total;


%   Calculate the CSP according to Koethe's and adding the penalisation
%   term weighted by our lambda
[V,~] = eig(cov(X_up_train),cov(X_up_train)+cov(X_down_train)+lambda * (Delta1 + Delta2));
CSP = V(:,[1:nof end-nof+1:end]);

%% Estimate final features

Feature_data = cspFeatures(Data,CSP,nof,epoch_size,do_ratiovar,do_logvar);

end

