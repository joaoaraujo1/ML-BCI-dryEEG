function plotTrialExample(trialsUp,trialsDown,trialsUp_len,trialsDown_len,preferences)

%Filter variables
Nyq = preferences.Fs/2;
Wo = 50/Nyq;
BW = Wo/20;
[myButter.b, myButter.a] = butter(4, [1 80]/Nyq,'bandpass');
[myButter.d, myButter.c] = iirnotch(Wo,BW);
Fs = preferences.Fs;

%% Plots of activity across time for each channel
%Plot up trial (trial 14)

filt_EEG_UP = filtfilt(myButter.b, myButter.a, trialsUp(sum(trialsUp_len(1:13))+1:sum(trialsUp_len(1:13)) + trialsUp_len(14),:));
filt_EEG_UP = filter(myButter.d, myButter.c, filt_EEG_UP);

figure
suptitle('Example of a UP(red) and DOWN(blue) trial')
set(gca,'fontsize',10)

ch_n = size(filt_EEG_UP,2);
for i = 1:ch_n
    subplot(ch_n,2,2*i-1);
    plot(1/Fs:1/Fs:trialsDown_len(2)/Fs,filt_EEG_UP(:,i),'r')
    box off
    ylim([-.00005, .00005])
    xlim([0, trialsUp_len(14)/Fs]);
    set(gca,'fontsize',9)
    ylabel('V');
    if i == ch_n
        xlabel('Seconds')
    end
    title(['Channel ' num2str(i) ' - Activity over Time'])
end

%Plot down trial (trial 2)

filt_EEG_DOWN = filtfilt(myButter.b, myButter.a, trialsDown(sum(trialsDown_len(1:1))+1:sum(trialsDown_len(1:1)) + trialsDown_len(2),:));
filt_EEG_DOWN = filter(myButter.d, myButter.c, filt_EEG_DOWN);
for i = 1:ch_n
    subplot(ch_n,2,2*i);
    plot(1/Fs:1/Fs:trialsDown_len(2)/Fs,filt_EEG_DOWN(:,i),'b')
    box off
    ylim([-.00005, .00005])
    xlim([0, trialsDown_len(2)/Fs]);
    set(gca,'fontsize',9)
    ylabel('V');
    if i == ch_n
        xlabel('Seconds')
    end
    title(['Channel ' num2str(i) ' - Activity over Time'])
end

%% Spectrograms of the same trials
figure
suptitle('Spectrogram for UP(left) and DOWN(right) trials')
set(gca,'fontsize',10)
for i = 1:ch_n
    subplot(ch_n,2,2*i-1);
    spectrogram(filt_EEG_UP(:,i),hamming(125 + 62),62,[],Nyq*2,'yaxis'); 
    caxis([-130, -110]);
    title(['Channel ' num2str(i)]);
    ylim([1, 80]);
    set(gca,'fontsize',9);
end

for i = 1:ch_n
    subplot(ch_n,2,2*i);
    spectrogram(filt_EEG_DOWN(:,i),hamming(125 + 62),62,[],Nyq*2,'yaxis'); 
    caxis([-130, -110]);
    title(['Channel ' num2str(i)]);
    ylim([1, 80]);
    set(gca,'fontsize',9);
    ylabel('Hz');
end


%% Frequency-Power representations (FFT)

%  Parameters of the Welch method
nfft = length(filt_EEG_UP);% datapoints
window = nfft/5;           % 5 segments
noverlap = window/2;       % 50% overlap

figure
suptitle('FFT for UP(red) and DOWN(blue) trials')
set(gca,'fontsize',10)
%Up trial
for i = 1:ch_n
    subplot(ch_n,2,2*i-1);
    [pxx,fx] = pwelch(sqrt(filt_EEG_UP(:,i)), hamming(window),noverlap,nfft,Nyq*2);
    semilogy(fx,pxx,'r');
    title(['Channel ' num2str(i) ' - Power over Frequency']);
    set(gca,'fontsize',9);
    xlim([1,80]);
    ylim([10^-9, 10^-6]);
    xlabel('Hz');
    ylabel('V/Hz^1^/^2');
    grid on; grid minor;
end

%Down trial
for i = 1:ch_n
    subplot(ch_n,2,2*i);
    [pxx,fx] = pwelch(sqrt(filt_EEG_DOWN(:,i)), hamming(window),noverlap,nfft,Nyq*2);
    semilogy(fx,pxx,'b');
    title(['Channel ' num2str(i) ' - Power over Frequency']);    
    set(gca,'fontsize',9);
    xlim([1,80]);
    ylim([10^-9, 10^-6]);
    xlabel('Hz');
    ylabel('V/Hz^1^/^2');
    grid on; grid minor;
end

%% Channel raw data correlations
figure
corrDif = abs(corr(trialsDown)-corr(trialsUp));
subplot(3,3,2)  
heatmap(corr(trialsDown), {'C1','C2','C3','C4','C5'}, {'C1','C2','C3','C4','C5'}, '%0.2f', 'TextColor', 'w', ...
        'Colormap', 'copper');
title('Channel correlations for DOWN trials:');
subplot(3,3,5)
heatmap(corr(trialsUp), {'C1','C2','C3','C4','C5'}, {'C1','C2','C3','C4','C5'}, '%0.2f', 'TextColor', 'w', ...
        'Colormap', 'copper');
title('Channel correlations for UP trials:');
subplot(3,3,8)
heatmap(corrDif, {'C1','C2','C3','C4','C5'}, {'C1','C2','C3','C4','C5'}, '%0.2f', 'TextColor', 'w', ...
        'Colormap', 'copper');
title('Channel correlation absolute differences between UP/DOWN trials:');


%% Channel-Band correlations
X_up = [];
X_down = [];
ch = 1;
%Retrieve trial features
for dir = 1:2 %UP and DOWN trials
    
    trial_idx = 0;
    i = 0;
    X_trial = [];
    
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

            filt_EEG = filtfilt(myButter.b, myButter.a, EEG);
            filt_EEG = filter(myButter.d, myButter.c, filt_EEG);

            %for ch=1:preferences.ensemble_len
                sig = detrend(filt_EEG(:,ch));
                [psd,~] = pwelch(sig,[],[],[],preferences.Fs);
                %X_trial = [X_trial, log(mean(psd(1:4))), log(mean(psd(5:8))), log(mean(psd(15:23))), log(mean(psd(32:83)))];
                for k = 1:80/5
                   X_trial = [X_trial, log(mean(psd((k-1)*5 + 1:(k-1)*5 + 5)))];
                end

            %end

            if dir == 1
                X_up = [X_up;X_trial];
            else
                X_down = [X_down;X_trial];
            end

            X_trial = [];

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

varNames = {'C1 - bd1','C1 - bd2','C1 - bd3','C1 - bd4',...
            'C1 - bd5','C1 - bd6','C1 - bd7','C1 - bd8',...
            'C1 - bd9','C1 - bd10','C1 - bd11','C1 - bd12',...
            'C1 - bd13','C1 - bd14','C1 - bd15','C1 - bd16'};
corrDifBands = abs(corr(X_up)-corr(X_down));
figure
heatmap(corr(X_down), varNames, varNames, '%0.2f', 'TextColor', 'w', ...
        'Colormap', 'copper', 'Colorbar', true, 'TickAngle', 45,'ShowAllTicks', true);
title('Channel-Band correlations for DOWN trials:');
figure
heatmap(corr(X_up), varNames, varNames, '%0.2f', 'TextColor', 'w', ...
        'Colormap', 'copper', 'Colorbar', true, 'TickAngle', 45,'ShowAllTicks', true);
title('Channel-Band correlations for UP trials:');
figure
heatmap(corrDifBands, varNames, varNames, '%0.2f', 'TextColor', 'w', ...
        'Colormap', 'copper', 'Colorbar', true, 'TickAngle', 45,'ShowAllTicks', true);
title('Channel-Band correlation absolute differences between UP/DOWN trials:');



end