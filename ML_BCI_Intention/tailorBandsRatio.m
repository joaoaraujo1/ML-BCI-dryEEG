function [bands_psd] = tailorBandsRatio(psd_ratio_all,preferences)
%TAILORBANDSRATIO - Get the final filterbank using a custom heuristic
%   Get the 
%
%

median_psd_ratio = squeeze(mean((psd_ratio_all),3));

std_psd_ratio = squeeze(std((psd_ratio_all),[],3));


[~,psd_bands] = pwelch(zeros(500,1),preferences.Fs,[],[],preferences.Fs); % getting the bands of the fft at 500Hz

%% Get the most discriminative psd ratio difference

lineProp1.col{1} = 'r'; %RED
lineProp2.col{1} = 'b'; %BLUE
lineProp3.col{1} = 'k'; %BLACK
lineProp4.col{1} = 'c'; %CYAN
lineProp5.col{1} = 'y'; %YELLOW
lineProp6.col{1} = 'g'; %GREEN

%%
num_ratio_array = nan(preferences.ensemble_len,40);
den_ratio_array = nan(preferences.ensemble_len,40);

 for ens = 1:preferences.ensemble_len
    close all
    
    % PSD
   
    figure
    title(['Median psd difference between Up/Down epochs for channel ', num2str(ens)]);

    mseb(psd_bands(1:82)',median_psd_ratio(ens,:),std_psd_ratio(ens,:),lineProp3);
    hold on
    plot(psd_bands(1:82),zeros(size(psd_bands(1:82),1),1),'Color','g')
    ylim([-1 1])
    xlim([0 82])
    xlabel('Frequency(Hz)');
    
    %find the numerator for our channel best psd ratio
    psd_ratio_std_w = median_psd_ratio-std_psd_ratio;
    band_index = 1;
    
    while any(psd_ratio_std_w(ens,4:end) > 0) && band_index < 5
        
        % We check the max from the 4th element onwards to remove very low frequencies close to 0. So we have to sum
        % +3 when we track indexes
        [~,max_ratio_index] = max(psd_ratio_std_w(ens,4:end));

        i = max_ratio_index+3;
        max_ratio_band_downRange = max_ratio_index+3;
        max_ratio_band_upRange = max_ratio_index+3;

        while median_psd_ratio(ens,i) > 0 && psd_ratio_std_w(ens,i) > preferences.threshold_std
            max_ratio_band_downRange = i;
            i = i - 1;
            if i == 3
                break
            end
        end

        i = max_ratio_index+3;
        while median_psd_ratio(ens,i) > 0 && psd_ratio_std_w(ens,i) > preferences.threshold_std
            max_ratio_band_upRange = i;
            i = i + 1;
            if i > 82
                break
            end
        end

        plot(ones(size(psd_bands(1:82),1),1)*psd_bands(max_ratio_band_upRange),-(size(psd_bands(1:82),1))/2:(size(psd_bands(1:82),1))/2-1,'Color','r','LineStyle','--')
        plot(ones(size(psd_bands(1:82),1),1)*psd_bands(max_ratio_band_downRange),-(size(psd_bands(1:82),1))/2:(size(psd_bands(1:82),1))/2-1,'Color','r','LineStyle','--')
        
        psd_ratio_std_w(ens,max_ratio_band_downRange:max_ratio_band_upRange) = 0;
        
        num_ratio_array(ens,band_index) = psd_bands(max_ratio_band_downRange);
        num_ratio_array(ens,band_index+1) = psd_bands(max_ratio_band_upRange);
        
        band_index = band_index + 2;

    end

    %find the psd denominator for our channel best psd ratio
    psd_ratio_std_w = median_psd_ratio+std_psd_ratio;
    band_index = 1;
    
    while any(psd_ratio_std_w(ens,4:end) < 0) && band_index < 5
        
        [~,min_ratio_index] = min(psd_ratio_std_w(ens,4:end));

        i = min_ratio_index+3;
        min_ratio_band_downRange = min_ratio_index+3;
        min_ratio_band_upRange = min_ratio_index+3;

        while median_psd_ratio(ens,i) < 0 && psd_ratio_std_w(ens,i) < -preferences.threshold_std
            min_ratio_band_downRange = i;
            i = i - 1;
            if i == 3
                break
            end
        end

        i = min_ratio_index+3;
        while median_psd_ratio(ens,i) < 0 && psd_ratio_std_w(ens,i) < -preferences.threshold_std
            min_ratio_band_upRange = i;
            i = i + 1;
            if i > 82
                break
            end
        end

        plot(ones(size(psd_bands(1:82),1),1)*psd_bands(min_ratio_band_upRange),-(size(psd_bands(1:82),1))/2:(size(psd_bands(1:82),1))/2-1,'Color','b','LineStyle','--')
        plot(ones(size(psd_bands(1:82),1),1)*psd_bands(min_ratio_band_downRange),-(size(psd_bands(1:82),1))/2:(size(psd_bands(1:82),1))/2-1,'Color','b','LineStyle','--')
        
        psd_ratio_std_w(ens,min_ratio_band_downRange:min_ratio_band_upRange) = 10;
        
        den_ratio_array(ens,band_index) = psd_bands(min_ratio_band_downRange);
        den_ratio_array(ens,band_index+1) = psd_bands(min_ratio_band_upRange);
        
        band_index = band_index + 2;
    end

    pause
    
    close all
    
    
end
bands_psd = [num_ratio_array,den_ratio_array];

end