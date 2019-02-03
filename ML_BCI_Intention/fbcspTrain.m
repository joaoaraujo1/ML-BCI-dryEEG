function [FBCSP,final_filterbank,Feature_data] = fbcspTrain( epoch_size, nof, lambda,Data, K, ensemble_len, do_ratiovar, do_logvar, showPlots)
%   FBsCSP Filter Bank stationary Common Spatial Patterns. Selects the K
%   best spatial filters. Filter selection is done using LDA on each filter (wrapper 
%   approach). Uses the sCSP approach to correct for non stationarities in
%   the covariance matrices
%   
%   References:
%   Kai Keng Ang, Zheng Yang Chin, Haihong Zhang, and Cuntai Guan (2008)
%   Zheng Yang Chin, Kai Keng Ang, Chuanchu Wang, Cuntai Guan, Haihong Zhang (2009)  
%   
%   Joao Araujo, 2018
%
%

%   Get equal amount of trials for up and down conditions
X_up = Data.UP;
X_down = Data.DOWN;

if(size(X_up,1) > size(X_down,1))
    X_up = X_up(1:size(X_down,1),:);
else
    X_down = X_down(1:size(X_up,1),:);
end


%   Calculate how many data chunks you will use to estimate the
%   non-stationary features
trials_n = size(X_up,1)/epoch_size;
divisor_vector =1:trials_n;
divisors = divisor_vector(rem(trials_n,divisor_vector) == 0);
chunk_len = median(divisors); % choose the medium sized divisor
chunk_size = epoch_size * chunk_len;
Delta_total = trials_n/chunk_size;

nob = size(X_up,2) / ensemble_len; %bands = feature_length / n_channels
band_vec = 1:ensemble_len; % band vector
CSP = []; %Initialize CSP matrix variable


for band_idx= 0:ensemble_len:size(X_up,2)-1
    
    vec = band_vec + band_idx;
    
    %   Get your specific band features from the whole feature matrix
    X_up_train = X_up(:,vec);
    X_down_train = X_down(:,vec);
    

    %   Initialize variables (Deltas) for the non-stationarity penalisation
    Delta1 = 0;
    Delta2 = 0;

    %   Calculate Deltas
    for d = 1:Delta_total
        cov1 = cov(X_up_train((d-1) * chunk_size + 1 : d * chunk_size, :));
        delta1 = P(cov1 - cov(X_up_train));
        Delta1 = Delta1 + delta1;

        cov2 = cov(X_down_train((d-1) * chunk_size + 1 : d * chunk_size, :));
        delta2 = P(cov2 - cov(X_down_train));
        Delta2 = Delta2 + delta2;
    end

    %   final calculation: both Deltas mean
    Delta1 = Delta1 / Delta_total;
    Delta2 = Delta2 / Delta_total;

    %   Calculate the CSP according to Koethe's and adding the penalisation
    %   term weighted by our lambda
    [V,~] = eig(cov(X_up_train),cov(X_up_train)+cov(X_down_train)+lambda * (Delta1 + Delta2));
    CSP = [CSP,V(:,[1:nof end-nof+1:end])];

end


%   Calculate LDA accuracy for up/down trials on each individual
%   CSP feature
I = []; %Initialize accuracies vector
i = 1;
for b = 0:ensemble_len:size(X_up,2)-1
    
    for j = 1:2*nof
        
        %   Get feature vector with up/down trials
        R_num_up = squeeze(log(var(reshape((X_up(:,band_vec + b) * CSP(:,i)), epoch_size,[],1))))';
        R_num_down = squeeze(log(var(reshape((X_down(:,band_vec + b) * CSP(:,i)), epoch_size,[],1))))';
        
        X_i = [R_num_up;R_num_down];
    
        %   Get labels for the feature vectors
        y_i = [-1*ones(size(R_num_up,1),1); ones(size(R_num_down,1),1)];

        % Use LDA to get a measure of how linearly separable the trials are    
        model.w = (mean(R_num_down)-mean(R_num_up))/(cov(R_num_up)+cov(R_num_down));
        model.b = (mean(R_num_up)+mean(R_num_down))*(-model.w/2)';

        for p_idx = 1:length(X_i)
            y_pred_val(p_idx) = sign(X_i(p_idx,:) * model.w' + model.b);
        end
%         
        I = [I;abs(mean(y_pred_val' == y_i))];
                
        i = i + 1;
        
    end
      
end

%   Get the feature vector with the indexes of features with better LDA performance in a descending order
[~,sort_idx] = sort(I,'descend');

%   Select the first K features for the final feature vector based on the
%   features with higher LDA accuracy computed on the full CSP matrix
final_filterbank = [];

for k = 1:K

    final_filterbank = [final_filterbank,sort_idx(k)];
   
end

% Get our final filterbank and chosen vector with sorted filters
final_filterbank = sort(final_filterbank);
FBCSP = CSP(:,final_filterbank);


%% Feature retrieval

Feature_data = fbcspFeatures(Data,FBCSP,final_filterbank,nof,epoch_size,do_ratiovar,do_logvar);

%Plot projected data
if showPlots
    
    for i = 1:K-1

        % Get the bands
        band1 = floor(final_filterbank(i)/(nof*2))+1;
        if final_filterbank(i+1) == (nof*2)*nob
            band2 = nob;
        else
            band2 = floor(final_filterbank(i+1)/(nof*2))+1;
        end

        % Get filter number
        filter1 = rem(final_filterbank(i),nof*2);
        if filter1 == 0
            filter1 = nof*2;
        end
        filter2 = rem(final_filterbank(i+1),nof*2);
        if filter2 == 0
            filter2 = nof*2;
        end    
        
        % Plot FBCSP feature pairs
        figure
        subplot(1,2,1)
        plot(Feature_data.UP(:,i),Feature_data.UP(:,i+1),'b.')
        hold on
        plot(Feature_data.DOWN(:,i),Feature_data.DOWN(:,i+1),'r.')
        hold off
        title('FBCSP features')
        xlabel(['Filter ' num2str(filter1) ' of band ' num2str(band1)]);
        ylabel(['Filter ' num2str(filter2) ' of band ' num2str(band2)]);
        legend('UP epochs','DOWN epochs');
        subplot(1,4,3)
        boxplot([Feature_data.UP(:,i),Feature_data.UP(:,i+1)])
        title(['Feature distribution for UP epochs'])
        xlabel('FBCSP feature')
        ylim([0.2 1.6])
        subplot(1,4,4)
        boxplot([Feature_data.DOWN(:,i),Feature_data.DOWN(:,i+1)])
        title(['Feature distribution for DOWN epochs'])
        xlabel('FBCSP feature')
        ylim([0.2 1.6])
        
        % Plot boxplot of original band1 distributions and variance
        figure
        cols1 = (band1-1)*ensemble_len + 1: (band1-1)*ensemble_len + ensemble_len;
        dataBand1Up = X_up(:,cols1);
        dataBand1Down = X_down(:,cols1);
        featureUp = squeeze(log(var(reshape(dataBand1Up, epoch_size,[],size(dataBand1Up,2))))); 
        featureDown = squeeze(log(var(reshape(dataBand1Down, epoch_size,[],size(dataBand1Down,2)))));
        subplot(2,2,1)
        boxplot(featureUp)
        title(['Band ' num2str(band1) ' log variance for UP epochs'])
        xlabel('Channels')
        ylabel('log var')
        ylim([-28 -23])
        subplot(2,2,2)
        boxplot(featureDown)
        title(['Band ' num2str(band1) ' log variance for DOWN epochs'])
        xlabel('Channels')
        ylabel('log var')
        ylim([-28 -23])
        cols2 = (band2-1)*ensemble_len + 1: (band2-1)*ensemble_len + ensemble_len;
        dataBand2Up = X_up(:,cols2);
        dataBand2Down = X_down(:,cols2);
        featureUp = squeeze(log(var(reshape(dataBand2Up, epoch_size,[],size(dataBand2Up,2))))); 
        featureDown = squeeze(log(var(reshape(dataBand2Down, epoch_size,[],size(dataBand2Down,2)))));
        subplot(2,2,3)
        boxplot(featureUp)
        title(['Band ' num2str(band2) ' log variance for UP epochs'])
        xlabel('Channels')
        ylabel('log var')
        ylim([-28 -23])
        subplot(2,2,4)
        boxplot(featureDown)
        title(['Band ' num2str(band2) ' log variance for DOWN epochs'])
        xlabel('Channels')
        ylabel('log var')
        ylim([-28 -23])
        
        pause
        close all
    end
    
end

