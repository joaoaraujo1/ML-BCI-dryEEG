# Decoding user intention with dry-EEG data #
In this repository, I try to create the best classifier to decode user intention from his dry-EEG data (5 channels) recorded while he performed a BCI task (controlling a ball to go UP or DOWN on a pc screen). The whole dataset consists of 5 BCI sessions of training data and a full 6th session of test data. With this code, you get an intuition on the data with visualization techniques, do some feature engineering and train a few classifiers for the best possible outcome. You can try to tweak several parameters and train several classifiers by changing the preferences in the main script header.

**Data**
- *trialsUpN* - m1*c matrix where m1 is the number of datapoints recorded on session N while the participant was trying to move the ball upwards and c is the number of channels (5)
- *trialsDownN* - m2*c matrix where m2 is the number of datapoints recorded on session N while the participant was trying to move the ball downwards and c is the number of channels (5)


## What data? Time-series plot and basic channel correlations ##
Let's plot the 5-channel data of an entire UP and DOWN trial (with a 4th order Butterworth bandpass filter of 1-80Hz and a notch filter at 50Hz to remove powerline noise), both of the same length to get an idea of the data we are working with:
![raw](https://user-images.githubusercontent.com/40466329/51772189-4b082200-20e3-11e9-827d-ba437542294a.jpg)

By eye inspection, both trials look somewhat similar and the data is quite noisy. From this sample, Channel 3 appears to show the biggest waveform differences across trial types. Keep in mind that we need to make sure that such differences are generalized for the whole training dataset before choosing features based on this information. So now, with all the data, let's check the cross-correlation (Pearson's rho) of the channel signal and see if we get any interesting differences on these correlations across UP and DOWN trials:
![corrrawheat](https://user-images.githubusercontent.com/40466329/51774336-d389c100-20e9-11e9-88eb-92fcfb696c0d.jpg)

We can observe a bunch of high cross-correlations between EEG channels which, most likely, are closer to eachother. From these correlation heatmaps we can already take some information as Channels 2 and 3 show a .08  correlation difference between UP and DOWN trials. In fact, Channel 3 seems to yield, overall, the smaller cross-correlation values and, at the same time, the biggest differences in signal cross-correlation between the two conditions.

## One step further: Power-Frequency analysis and Band-Channel correlations ##
One popular method of analysing EEG data is by using power-frequency analysis, as different signal bandwidths have been associated with different emotional/cognitive states:
<img src="http://www.yogatoeaseanxiety.com/uploads/2/6/6/9/26696125/3706085.jpg?429" alt="alt text" width="550" height="300">


It is relevant to note that beta (and mu) band is implicated in a special type of mental imagery, namely *motor* imagery, which could possibly be relevant to consider given the nature of the BCI task. Let's now plot a spectrogram and mean power spectral density per band of our example trials:
![spectrogram](https://user-images.githubusercontent.com/40466329/51777062-709d2780-20f3-11e9-98d9-87bc0462f12e.jpg)
![fft](https://user-images.githubusercontent.com/40466329/51777079-80b50700-20f3-11e9-82ba-e6f35602b399.jpg)

This time, it is a bit harder to get any intuition from the data: in some channels beta and gamma seem to slightly differ from trial types but it could simply be a coincidence. So next, I got the full dataset of every session, divided it in 3-second long epochs (with 2 seconds overlap) and calculated the difference (UP-DOWN) of the median logarithmized power spectral density across the 2 conditions for the 1-80Hz bandidth. I have done so separately for each session to have an idea how this difference would hold across different days. As the profile of each channel was very similar, I have chosen to only show one representative channel (Channel 4):
![diffc4](https://user-images.githubusercontent.com/40466329/51777842-cde6a800-20f6-11e9-99b2-51bd6df1a13b.jpg)

In the figure, the black line represents the median difference and the grey bands the standard deviation across different testing days (sessions). From this figure we can see that:
- Delta band power seems to be slightly higher in UP epochs but with quite some variation across days;
- Theta and alpha bands power is, overall, slightly higher in DOWN epochs
- Beta band power is mostly higher for UP epochs although this tendency tends to invert at around 20Hz.
- Gamma band power is consistently higher for UP trials.

So now we have an idea of what are the main band power differences across the 2 conditions. However, we still do not know if the way they co-variate changes across conditions. Therefore, the next step  would be to see if the correlations between 1)bands of the same channel and 2) bands of different channels change across conditions. Overall, delta, theta, beta and gamma bands were considered for the next analysis. As alpha can be contaminated by artifacts such as [eyes closing](https://en.wikipedia.org/wiki/Hans_Berger), it was not considered this time. So let's create a plot for band-channel cross-correlations for both conditions and then calculate their absolute difference. For the figures below, D = delta, T = theta, B = beta, G = gamma and the numbers after the letters represent the channel Ids.

![bandup](https://user-images.githubusercontent.com/40466329/51778566-531f8c00-20fa-11e9-80ca-e9c91733df5b.jpg)

![banddown](https://user-images.githubusercontent.com/40466329/51778567-561a7c80-20fa-11e9-851e-97752720d164.jpg)

![banddiff](https://user-images.githubusercontent.com/40466329/51778569-57e44000-20fa-11e9-91a0-8d6638d071ff.jpg)

Interestingly, this analysis does not show the discriminative power of Channel 3 anymore. On the other hand, Channels 2 and 5 show remarkable cross-correlation changes up to **.33**. Channel 4 also shows some decent correlation changes of up to .25 and Channel 1 has one correlation change of .30.

## Feature discriminability: Common Spatial Patterns ##
Although the above mentioned bands are historically important for studying cognitive processes, it is not guaranteed that those specific bandwidth intervals are the best choice for filtering our signal on this specific task for this specific participant. In fact, part of our beta band power (at ~20Hz) shows the inverse tendency of variation across trial types as the rest of the 12-30Hz interval. To further prove this point, let's just take Channel 1 from the ensemble, divide its signal in small bands of ~5Hz (16 bands for 1-80Hz signal) and plot the cross-correlation differences between UP and DOWN trials on its bands power. As we can see, we can get much higher cross-correlation differences in Channel 1:

![fivehzdiff](https://user-images.githubusercontent.com/40466329/51811021-f7443700-22a2-11e9-95b1-2af500070240.jpg)

However, if we stack a lot of these smaller band features, we end up with high-dimensional data (up to 400-dimensions if we use 1Hz bands from 1-80Hz across 5 channels). So it would be interesting if we could reduce the dimensionality of these data while maximizing the discriminability of the resulting features. Enter Common Spatial Patterns. CSP creates a set of spatial filters where we project our data to enhance discriminability between our two conditions by whitening and rotatating our data. Generally this is what it does:
- We start with ∑1 and ∑2 that correspond to the covariance matrices of our bandpass-filtered “up” and “down” epochs across our 5 channels.
- Simultaneously diagonalize the 2 covariance matrices (real symmetric matrices) to get a common W: 

  *WT ∑1 W = Λ1* and *WT ∑2 W = Λ2*
  
- The simultaneous decomposition ends up scaling W (the generalized eigenvectors) so that

  *Λ1  +  Λ2  = I*

- And so we verify the following:

![image](https://user-images.githubusercontent.com/40466329/51809307-7895cc80-2297-11e9-85b1-cefddae57e8c.png)

In the end, a large value of λ will correspond to a large eigenvalue for ∑1 and a small eigenvalue for ∑2. This means that a corresponding spatial filter that yields high variance for a condition will yield low variance for the other and vice-versa. Therefore, from the resulting W matrix, we take only the the first and last *n* filters (column vectors). To project the data into one of these spatial filters, we do matrix-vector multiplication between our filtered signal (dim: m X c) and the spatial filter (c X 1). In the end we end up with a new feature resulting on the weighted combination of the data of our 5 channels. For more about CSP, check this [online course](https://sccn.ucsd.edu/wiki/Introduction_To_Modern_Brain-Computer_Interface_Design) by Christian Kothe (lecture 7) and [this](https://ieeexplore.ieee.org/document/4408441) paper from Blankertz et al.

### Filter Bank CSP ###
A way to apply the CSP idea to our multi-band data is to calculate the filters for each bandpass-filtered signal separately and then choose the best K filters. This is called the Filter Bank CSP (FBCSP) and it was first introduced [here](https://ieeexplore.ieee.org/document/4634130) by Ang et al. In my code, I followed closely the paper's implementation with one exception: Instead of using Mutual Information to choose the best filters, I used Linear Discriminant Analysis as it yielded much better results for this dataset. So let's try to apply this in 3-sec epochs and we can choose, for instance, 10Hz bands. Then we plot some of the best FBCSP features that allow us to discriminate between UP and DOWN trials:
![fbcspplot](https://user-images.githubusercontent.com/40466329/51855014-1fc04580-2324-11e9-9be2-4c185c0d02dc.jpg)

Although we still have quite an overlap it is interesting that we can separate, even by inspection a considerable amount of UP and DOWN epochs only with 2 engineered features (in the case from bands 30-40Hz and 40-50Hz, both within the gamma range). This is quite remarkable, especially when we realize how similar the original distributions of the pre-FBCSP bands across all channels were for UP and DOWN trials:
![varbands](https://user-images.githubusercontent.com/40466329/51855013-1fc04580-2324-11e9-941d-6066d8e15937.jpg)

## Optimizing bandwidths and multi-band CSP ##
COMING SOON
