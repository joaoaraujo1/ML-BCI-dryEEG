function [bandpass_filters] = heuristicFilterEstimation(preferences,trialsUp,trialsDown,trialsUp_len,trialsDown_len)
%HEURISTIC FILTER ESTIMATION - Retrieve features or filters that maximize differences for up/down trials
%
% João Araújo, 2018
%

%EEG filter variables
Nyq = preferences.Fs/2;
Wo = 50/Nyq;
BW = Wo/20;
[myButter.b, myButter.a] = butter(4, [1 80]/Nyq,'bandpass');
[myButter.d, myButter.c] = iirnotch(Wo,BW);


%log psd matrices
dB_up = [];
dB_down = [];

highPassDown = [];
highPassUp = [];

lowPassDown = [];
lowPassUp = [];


%Retrieve trial features
bands = zeros(257,1);

for dir = 1:2 %UP and DOWN trials
    
    trial_idx = 0;
    i = 0;
    EEG = nan(preferences.epoch_size, preferences.ensemble_len);
    
    if dir == 1
        Trials = trialsUp;
        TrialsLen = trialsUp_len;
        ii = 1;
    else
        Trials = trialsDown;
        TrialsLen = trialsDown_len;
        ii = 1;
    end

    while i < length(Trials)
    
        EEG = Trials(i  + 1 : i  + preferences.epoch_size,:);
        i = i + preferences.epoch_size;
        trial_idx = trial_idx + 1;
        sample_idx = preferences.epoch_size;

        while sample_idx <= (TrialsLen(trial_idx)) 
            
                
            filt_EEG = filtfilt(myButter.b, myButter.a, EEG);
            filt_EEG = filter(myButter.d, myButter.c, filt_EEG);
            
            filt_EEG = filt_EEG - repmat(mean(filt_EEG),preferences.epoch_size,1);
            
            %epoch re-scaling option
            if(preferences.heur_normalize_epochs == 1)
                for ch = 1:preferences.ensemble_len
                    filt_EEG(:,ch) = (filt_EEG(:,ch) - min(filt_EEG(:,ch)))/(max(filt_EEG(:,ch)) - min(filt_EEG(:,ch)));
                end
            end
            
            psd = zeros(257,preferences.ensemble_len);
            
            for b=1:preferences.ensemble_len
                sig = filt_EEG(:,b);
                sig = detrend(sig);
                [psd(:,b),bands] = pwelch((sig),preferences.Fs,[],[],preferences.Fs); % pwelch using 1sec of data 
                dB(:,b) = (log(psd(:,b)));
            end
            
            if dir == 1
                dB_up(ii,:,:) = dB(1:82,:);
                ii = ii+1;
            else
                dB_down(ii,:,:) = dB(1:82,:);
                ii = ii+1;
            end

            if sample_idx + 125 <= (TrialsLen(trial_idx))
                EEG = circshift(EEG, -preferences.window_step); % moving window
                EEG(end - preferences.window_step  + 1 : end, : ) = Trials(i - preferences.window_step + 1 : i,:); %update EEG buffer window with some new data
                i = i + preferences.window_step;
            end

            sample_idx = sample_idx + preferences.window_step;

        end

    end
end


%% Heuristic 1: Filters estimated by cross-correlation in band PSDs
%
%   Heuristic based on Benjamin Blankertz, Ryota Tomioka, Steven Lemm,
%   Motoaki Kawanabe, and Klaus-Robert Müller (2008) with the following
%   changes:
%   - The correlation score is calculated separately for each channel and
%   the frequency band is much larger (1-80Hz)
%   - Instead of imposing a threshold of .95 of the score value to increase
%   bandwidth, this value is now customizable in the main script for
%   optimization
%

if preferences.do_heuristic == 1
    
    fprintf('\nEstimating the best filters by channel frequency PSD cross-correlation...\n')

    for jj = 1:preferences.ensemble_len

        cor_coef_up(jj,:,:) = corrcoef(squeeze(dB_up(:,:,jj)));
        cor_coef_up(cor_coef_up == 1) = 0;
        cor_sum_up(:,jj) = squeeze(sum(cor_coef_up(jj,:,:),2));
        [valmax_up(jj), argmax_up(jj)] = max(cor_sum_up(:,jj));

        cor_coef_down(jj,:,:) = corrcoef(squeeze(dB_down(:,:,jj)));
        cor_coef_down(cor_coef_down == 1) = 0;
        cor_sum_down(:,jj) = squeeze(sum(cor_coef_down(jj,:,:),2));
        [valmax_down(jj), argmax_down(jj)] = max(cor_sum_down(:,jj));

    end


    if(any(valmax_down) < 0) fprintf('\n\nWARNING!! Some max correlation values for DOWN are < 0!!\n\n');
    end
    if(any(valmax_up) < 0) fprintf('\n\nWARNING!! Some max correlation values for UP are < 0!!\n\n');
    end

    for ch = 1:preferences.ensemble_len

        fmax_up = squeeze(sum(sum(cor_coef_up(ch,:,:),1)))/(preferences.ensemble_len + size(dB_up,2));
        fmax_down = squeeze(sum(sum(cor_coef_down(ch,:,:),1)))/(preferences.ensemble_len + size(dB_down,2));


        [f_star_score_max_up, f_star_max_up] = max(fmax_up);
        [f_star_score_max_down, f_star_max_down] = max(fmax_down);

        f0_up = f_star_max_up;
        f1_up = f_star_max_up;

        f0_down = f_star_max_down;
        f1_down = f_star_max_down;

        while( f0_up - 1 > 0)
            if (fmax_up(f0_up - 1) >= f_star_score_max_up * preferences.corr_threshold)

                f0_up = f0_up - 1;
            else break
            end
        end

        while(f1_up + 1 <= length(fmax_up))
            if (fmax_up(f1_up + 1) >= f_star_score_max_up * preferences.corr_threshold)

                f1_up = f1_up + 1;

            else break
            end
        end

        if(abs(bands(f0_up) - bands(f1_up + 1)) > preferences.min_bandsize)
            highPassUp = bands(f0_up); if(highPassUp == 0) highPassUp = 1; end
            lowPassUp = bands(f1_up +1); %% +1 because we are getting the lower limit of the band with the psd
            else highPassUp = NaN; lowPassUp = NaN;
        end

        %fprintf('\nThe best bandpass filter for UP is between %.2f and %.2f Hz\n',highPassUp,lowPassUp);

        while( f0_down - 1 > 0)
            if (fmax_down(f0_down - 1) >= f_star_score_max_down * preferences.corr_threshold)

                f0_down = f0_down - 1;
            else break
            end
        end

        while(f1_down + 1 <= length(fmax_down))
            if (fmax_down(f1_down + 1) >= f_star_score_max_down * preferences.corr_threshold)

                f1_down = f1_down + 1;

            else break
            end
        end

        if(abs(bands(f0_down) - bands(f1_down + 1)) > preferences.min_bandsize && (f0_down ~= f0_up || f1_down ~= f1_up))
            highPassDown = bands(f0_down); if(highPassDown == 0) highPassDown = 1; end
            lowPassDown = bands(f1_down + 1); %% +1 because we are getting the lower limit of the band with the psd

        else highPassDown = NaN; lowPassDown = NaN;
        end

        %fprintf('The best bandpass filter for DOWN is between %.2f and %.2f Hz\n',highPassDown,lowPassDown);

        bandpass_filters(:,:,ch) = [highPassUp,lowPassUp;highPassDown,lowPassDown];


    end
    
end

%% Heuristic 2: Filters estimated by differences in band PSD of up/down trials
%
%   This heuristic has the main objective of finding which bands hold the
%   maximum PSD difference across conditions and use that information to
%   build a bandpass filter. It goes as follows:
%
%   1) Shuffle epochs and find the median difference between UP/DOWN epochs
%   2) Repeat this process X times and get the overall mean of medians
%   3)

if preferences.do_heuristic == 2

    dB_up = squeeze(reshape(dB_up,[],82,preferences.ensemble_len));
    dB_down = squeeze(reshape(dB_down,[],82,preferences.ensemble_len));

    for m = 1:preferences.heur_shuffles

        dB_up_new = zeros(size(dB_up));
        dB_down_new = zeros(size(dB_down));
        
        %shuffle data to calculate the ratio between up and down band PSDs
        shuffle_up = randperm(size(dB_up,1));
        for s = 1:length(shuffle_up)
            dB_up_new(s,:,:) = dB_up(shuffle_up(s),:,:);
        end

        shuffle_down = randperm(size(dB_down,1));
        for s = 1:length(shuffle_down)
            dB_down_new(s,:,:) = dB_down(shuffle_down(s),:,:);
        end

        % truncate UP if UP > DOWN
        if(size(dB_up,1) > size(dB_down,1))
            dB_up_new = dB_up_new(1:(size(dB_down,1)),:,:);
        end

        % truncate DOWN if DOWN > UP
        if(size(dB_up,1) < size(dB_down,1))
            dB_down_new = dB_down_new(1:(size(dB_up,1)),:,:);
        end

        % Save the median PSD difference across trials
        for ens = 1:preferences.ensemble_len
            psd_ratio(ens,:,m)  = median(squeeze((reshape(dB_up_new(:,:,ens),[],82,1))) - squeeze((reshape(dB_down_new(:,:,ens),[],82,1)))); 
        end
        
    end
    
    bandpass_filters = mean(psd_ratio,3);
    
    %bandpass_filters = tailorBandsRatio(psd_ratio,preferences);
    
end

end

