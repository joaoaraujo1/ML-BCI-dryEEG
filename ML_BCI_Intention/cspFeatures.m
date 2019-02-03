function [Feature_data] = cspFeatures(Data,CSP,nof,epoch_size,do_ratiovar,do_logvar)
% CSPFEATURES - Returns Multi-CSP trained features
%
%   Get the final spatial filtered full training data using log-variance
%   features according to Ramoser et al, 2000 paper or simple variance
%   features as to user's preferences
%
%   João Araújo, 2018
%

% Calculate numerator
if do_logvar == 0
    R_num_up = squeeze(var(reshape((Data.UP * CSP), epoch_size,[],2*nof))); 
    R_num_down = squeeze(var(reshape((Data.DOWN * CSP), epoch_size,[],2*nof))); 
else
    R_num_up = squeeze(log(var(reshape((Data.UP * CSP), epoch_size,[],2*nof)))); 
    R_num_down = squeeze(log(var(reshape((Data.DOWN * CSP), epoch_size,[],2*nof)))); 
end

% Calculate denominator
R_den_up = sum(R_num_up,2);
R_den_down = sum(R_num_down,2);

% Final features
Feature_data.UP = [];
Feature_data.DOWN = [];
if do_ratiovar == 1
    
    for p = 1:2*nof
        Feature_data.UP  = [Feature_data.UP, log(R_num_up(:,p)./R_den_up)];
        Feature_data.DOWN  = [Feature_data.DOWN, log(R_num_down(:,p)./R_den_down)];   
    end
    
else
    Feature_data.UP = R_num_up;
    Feature_data.DOWN = R_num_down;   
end



end
