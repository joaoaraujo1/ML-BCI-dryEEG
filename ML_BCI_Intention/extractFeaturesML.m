function [Data,preferences] = extractFeaturesML(preferences,trialsUp,trialsDown,trialsUp_len,trialsDown_len)
%EXTRACT TRIAL FEATURES - Retrieve features for up/down trials
%
%   João Araújo, 2018
%

%EEG filter variables
Nyq = preferences.Fs/2;
Wo = 50/Nyq;
BW = Wo/20;

% Variables of interest for trials
X_up = [];
X_down = [];

% Heuristic-based filterband estimation
if preferences.do_heuristic ~= 0 && isfield(preferences,'final_filters') == 0
    
    load('BMI_S1_data');
    
    % Heuristic 1 - based on 
    if preferences.do_heuristic == 1
        preferences.final_filters = heuristicFilterEstimation(preferences,trialsUp5,trialsDown5,trialsUp5_len,trialsDown5_len);
    
    elseif preferences.do_heuristic == 2
        psd_ratio_all = heuristicFilterEstimation(preferences,trialsUp5,trialsDown5,trialsUp5_len,trialsDown5_len);
%         filters_S2 = heuristicFilterEstimation(preferences,trialsUp2,trialsDown2,trialsUp2_len,trialsDown2_len);
%         filters_S3 = heuristicFilterEstimation(preferences,trialsUp3,trialsDown3,trialsUp3_len,trialsDown3_len);
%         filters_S4 = heuristicFilterEstimation(preferences,trialsUp4,trialsDown4,trialsUp4_len,trialsDown4_len);
%         filters_S5 = heuristicFilterEstimation(preferences,trialsUp5,trialsDown5,trialsUp5_len,trialsDown5_len);
%         psd_ratio_all = cat(3,filters_S1,filters_S2,filters_S3,filters_S4,filters_S5);
        preferences.final_filters = tailorBandsRatio(psd_ratio_all,preferences);
    end
end

%Retrieve trial features
for dir = 1:2 %UP and DOWN trials
    
    trial_idx = 0;
    i = 0;
    bd = 1;
    X_trial = [];
    EEG = nan(preferences.epoch_size, preferences.ensemble_len);
    
    if dir == 1
        Trials = trialsUp;
        TrialsLen = trialsUp_len;
        fprintf('\nFiltering and extracting EEG features from UP trials... \n')
    else
        Trials = trialsDown;
        TrialsLen = trialsDown_len;
        fprintf('\nFiltering and extracting EEG features from DOWN trials... \n')
    end

    while i < length(Trials)
    
        EEG = Trials(i  + 1 : i  + preferences.epoch_size,:);
        i = i + preferences.epoch_size;
        trial_idx = trial_idx + 1;
        sample_idx = preferences.epoch_size;

        while sample_idx <= (TrialsLen(trial_idx))
            
            if preferences.do_dwt == 1


                [myButter.b, myButter.a] = butter(4, [1 80]/Nyq,'bandpass');
                [myButter.d, myButter.c] = iirnotch(Wo,BW);

                filt_EEG = filtfilt(myButter.b, myButter.a, EEG);
                filt_EEG = filter(myButter.d, myButter.c, filt_EEG);

                if preferences.center_epochs == 1
                    for ch = 1:preferences.ensemble_len
                        filt_EEG(:,ch) = filt_EEG(:,ch) - repmat(mean(filt_EEG(:,ch)),preferences.epoch_size,1);
                    end

                end

                if preferences.detrend_epochs == 1
                    for ch = 1:preferences.ensemble_len
                        filt_EEG(:,ch) = detrend(filt_EEG(:,ch));
                    end
                end

                %epoch re-scaling option
                if(preferences.normalize_epochs == 1)
                    for ch = 1:preferences.ensemble_len
                        filt_EEG(:,ch) = (filt_EEG(:,ch) - min(filt_EEG(:,ch)))/(max(filt_EEG(:,ch)) - min(filt_EEG(:,ch)));
                    end
                end

                for ch=1:preferences.ensemble_len
                    [C, L] = wavedec(filt_EEG(:,ch),preferences.dwt_level,preferences.wname);
                    D = detcoef(C,L,'cells');

                    % we skip the first detail level which is the information
                    % from 125-250 Hz
                    for level = 2:preferences.dwt_level
                        if preferences.dwt_psd == 1
                            [psd,~] = pwelch(D{level},[],[],[],preferences.Fs/(level+2));
                            X_trial = [X_trial, log(mean(psd))];
                            
                        end
                            
                        if preferences.dwt_var == 1
                            X_trial = [X_trial, log(var(D{level}))];
                            
                        end
                        
                        if preferences.dwt_raw == 1
                            X_trial = [X_trial, D{level}'];
                        end
                            
                            
                    end

                    % We add the approximation coefficients information
                    A = appcoef(C,L,preferences.wname);
                    if preferences.dwt_psd == 1
                        [psd,~] = pwelch(A,[],[],[],preferences.Fs/(level+2));
                        X_trial = [X_trial, log(mean(psd))];
                        
                    end

                    if preferences.dwt_var == 1
                        X_trial = [X_trial, log(var(A))];
                    
                    end
                    
                    if preferences.dwt_raw == 1
                        X_trial = [X_trial, A'];
                    end

                end

            end
            
            % for standard CSP retrieve filtered signal in consecutive x-sized bands
            if(preferences.do_csp == 1 || preferences.do_fbcsp == 1) && preferences.do_heuristic == 0
                
                while(bd + preferences.bandsize-1 <= 80)

                    [myButter.b, myButter.a] = butter(4, [bd (bd + preferences.bandsize)]/Nyq,'bandpass');
                    [myButter.d, myButter.c] = iirnotch(Wo,BW);

                    filt_EEG = filtfilt(myButter.b, myButter.a, EEG);
                    filt_EEG = filter(myButter.d, myButter.c, filt_EEG);

                    if preferences.center_epochs == 1
                        for ch = 1:preferences.ensemble_len
                            filt_EEG(:,ch) = filt_EEG(:,ch) - repmat(mean(filt_EEG(:,ch)),preferences.epoch_size,1);
                        end

                    end

                    if preferences.detrend_epochs == 1
                        for ch = 1:preferences.ensemble_len
                            filt_EEG(:,ch) = detrend(filt_EEG(:,ch));
                        end
                    end
                    
                    %epoch re-scaling option
                    if(preferences.normalize_epochs == 1)
                        for ch = 1:preferences.ensemble_len
                            filt_EEG(:,ch) = (filt_EEG(:,ch) - min(filt_EEG(:,ch)))/(max(filt_EEG(:,ch)) - min(filt_EEG(:,ch)));
                        end
                    end                   
                    
                    X_trial = [X_trial, filt_EEG];

                    bd = bd + preferences.bandsize;

                end
            end
            
            % for heuristic-based methods (with or without CSP) use the tailored filters
            if preferences.do_heuristic == 1
                
               for ens = 1:preferences.ensemble_len
                
                    bd = 1;

                    while(bd <= size(preferences.final_filters,1))

                        if(isnan(preferences.final_filters(bd,1,ens)) == 0)
                            [myButter.b, myButter.a] = butter(4, [preferences.final_filters(bd,1,ens) preferences.final_filters(bd,2,ens)]/Nyq,'bandpass');

                            [myButter.d, myButter.c] = iirnotch(Wo,BW);

                            filt_EEG = filtfilt(myButter.b, myButter.a, EEG(:,ens));
                            filt_EEG = filter(myButter.d, myButter.c, filt_EEG);
                        
                            if preferences.center_epochs == 1
                                filt_EEG = filt_EEG - repmat(mean(filt_EEG),preferences.epoch_size,1);
                            end

                            if preferences.detrend_epochs == 1
                                filt_EEG = detrend(filt_EEG);
                            end
                            
                            %epoch re-scaling option
                            if(preferences.normalize_epochs == 1)
                                filt_EEG  = (filt_EEG - min(filt_EEG))/(max(filt_EEG) - min(filt_EEG));
                            end

                            X_trial = [X_trial, filt_EEG];
                            
                        end

                        bd = bd + 1;

                    end
                
                end
                
            end
            
            if preferences.do_heuristic == 2
                
                for ens = 1:preferences.ensemble_len
                
                    bd_idx = 1;

                    while(isnan(preferences.final_filters(ens,bd_idx)) == 0 && bd_idx < size(preferences.final_filters,2))

                        [myButter.b, myButter.a] = butter(4, [preferences.final_filters(ens,bd_idx) preferences.final_filters(ens,bd_idx+1)]/Nyq,'bandpass');
                        [myButter.d, myButter.c] = iirnotch(Wo,BW);
                        filt_EEG = filtfilt(myButter.b, myButter.a, EEG(:,ens));
                        filt_EEG = filter(myButter.d, myButter.c, filt_EEG);

                        if preferences.center_epochs == 1
                            filt_EEG = filt_EEG - repmat(mean(filt_EEG),preferences.epoch_size,1);
                        end

                        if preferences.detrend_epochs == 1
                            filt_EEG = detrend(filt_EEG);
                        end
                        
                        %epoch re-scaling option
                        if(preferences.normalize_epochs == 1)
                            filt_EEG  = (filt_EEG - min(filt_EEG))/(max(filt_EEG) - min(filt_EEG));
                        end
                        
                        X_trial = [X_trial, filt_EEG];
                        bd_idx = bd_idx + 2;

                    end
                    
                end
                
            end
            

            if dir == 1
                X_up = [X_up;X_trial];
            else
                X_down = [X_down;X_trial];
            end

            X_trial = [];
            bd = 1;

            if sample_idx + preferences.window_step <= (TrialsLen(trial_idx))
                EEG = circshift(EEG, -preferences.window_step); % moving window
                EEG(end - preferences.window_step  + 1 : end, : ) = Trials(i - preferences.window_step + 1 : i,:); %update EEG buffer window with some new data
                sample_idx = sample_idx + preferences.window_step ;
                i = i + preferences.window_step;
                
            else break;
                
            end


        end

    end
end

Data.UP = X_up;
Data.DOWN = X_down;


end

