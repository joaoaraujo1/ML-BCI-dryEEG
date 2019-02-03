function [Feature_data] = fbcspFeatures(Data,CSP,final_filterbank,nof,epoch_size,do_ratiovar,do_logvar)
% FBCSPFEATURES - Returns FBCSP trained features
%
%   Get the final spatial filtered full training data using log-variance
%   features according to Ramoser et al, 2000 paper or simple variance
%   features as to user's preferences
%
%   João Araújo, 2018
%

band_vec = 1:5; % band vector
i = 1;
csp_idx = 1;
Feature_data.UP = [];
Feature_data.DOWN = [];

for b=0:5:size(Data.UP,2)-1
    
    for j = 1:2*nof
        
        if(any(final_filterbank == i) == 1)
            
            if do_logvar == 0
                Feature_data.UP   = [Feature_data.UP,squeeze((var(reshape((Data.UP(:,band_vec + b) * CSP(:,csp_idx) ), epoch_size,[],1))))']; 
                Feature_data.DOWN = [Feature_data.DOWN,squeeze((var(reshape((Data.DOWN(:,band_vec + b) * CSP(:,csp_idx) ), epoch_size,[],1))))']; 
            else
                Feature_data.UP   = [Feature_data.UP,squeeze(log(var(reshape((Data.UP(:,band_vec + b) * CSP(:,csp_idx) ), epoch_size,[],1))))']; 
                Feature_data.DOWN = [Feature_data.DOWN,squeeze(log(var(reshape((Data.DOWN(:,band_vec + b) * CSP(:,csp_idx) ), epoch_size,[],1))))']; 
            end
            
            csp_idx = csp_idx + 1;
        end

        i = i + 1;
        
    end
    
end

if do_ratiovar == 1
    
    % Calculate denominator
    R_den_up = sum(Feature_data.UP ,2);
    R_den_down = sum(Feature_data.DOWN,2);
    
    R_num_up = Feature_data.UP;
    R_num_down = Feature_data.DOWN;
    
    Feature_data.UP = [];
    Feature_data.DOWN = [];
    
    for p = 1:size(R_num_up,2)
        Feature_data.UP  = [Feature_data.UP, log(R_num_up(:,p)./R_den_up)];
        Feature_data.DOWN  = [Feature_data.DOWN, log(R_num_down(:,p)./R_den_down)];   
    end
    
end



end
