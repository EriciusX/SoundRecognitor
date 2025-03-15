clear; clc; close all;

%% Parameters

numTrainingFiles = 11;
numTestFiles     = 8;
trainingFiles = './GivenSpeech_Data/Training_Data/s%d.wav';  % Files
plotfile = 2;
trim_threshold = 0.01;                                      % Threshold for trimming silence
testFiles     = './GivenSpeech_Data/Test_Data/s%d.wav';

% MFCC parameters
frameLength = 512;      % Frame length in samples
numMelFilters = 20;     % Number of Mel filter banks
numMfccCoeffs = 20;     % Total number of MFCC coefficients

% VQ-LBG parameters
targetCodebookSize = 8; % The desired number of codewords in the final codebook
epsilon = 0.01;         % Splitting parameter
tol = 1e-3;             % Iteration stopping threshold

% Plotting parameters
% e.g. We will plot the 6th and 7th MFCC coefficients of speaker 2 and 10
dim1 = 6;
dim2 = 7;
speakerList = [2, 10]; % select 2 speakers

% Screen parameters 
screenSize = get(0, 'ScreenSize');
screenWidth = screenSize(3);
screenHeight = screenSize(4);

%% Test 2.1
% Load and plot speech signal in time domain
[y, Fs] = audioread(sprintf(trainingFiles, plotfile));
sound(y, Fs);

% Normalize
y = y / max(abs(y));

% Calculate time for 256 samples 
time_ms = (256 / Fs) * 1000;
fprintf('Sampling rate: %d Hz\n', Fs);
fprintf('Duration of 256 samples: %.2f milliseconds\n', time_ms);

% Plot signal in time domain before trimming
t = (0:length(y)-1) / Fs;  
fig1 = figure;
set(fig1, 'Position', [100, screenHeight-500, 600, 400]);
subplot(2,1,1);
plot(t, y);
xlim([0 max(t)]);
xlabel('Time (s)');
ylabel('Amplitude');
title('Signal before Trimming in Time Domain');

% Load the speech after auto trim silence from the beginning and end of the signal
[y_trim, Fs] = autoTrimSilence(sprintf(trainingFiles, plotfile), frameLength, trim_threshold);
sound(y_trim, Fs);

% Plot signal in time domain after trimming
t_trim = (0:length(y_trim)-1) / Fs;
subplot(2,1,2);
plot(t_trim, y_trim);
xlim([0 max(t_trim)]);
xlabel('Time (s)');
ylabel('Amplitude');
title('Signal after Trimming in Time Domain');

%% Test 2.2

% Plot spectrogram of speech signal for different frame sizes
frame_sizes = [128, 256, 512];

fig2 = figure;
set(fig2, 'Position', [700, screenHeight-500, 600, 400]);

for k = 1:length(frame_sizes)
    N = frame_sizes(k);
    M = round(N / 3); % frame increment
    
    num_frames = floor((length(y) - N) / M) + 1;
    stft_result = zeros(N, num_frames);
    window = hamming(N);

    % Compute STFT
    for i = 1:num_frames
        frame_start = (i-1)*M + 1;
        frame = y(frame_start:frame_start+N-1) .* window;
        stft_result(:,i) = abs(fft(frame)).^2;
    end

    stft_result = stft_result(1:N/2+1, :);
    stft_result_db = 10*log10(stft_result);

    % Create time and frequency vectors for plotting
    t = ((0:num_frames-1) * M) / Fs * 1000;  % Time in milliseconds
    f = (0:N/2) * Fs / N;  % Frequency in Hz

    % Plot spectrogram before trimming
    subplot(4, 1, k);
    imagesc(t, f, stft_result_db);
    axis xy;  % Put low frequencies at bottom
    colorbar;
    xlabel('Time (ms)');
    ylabel('Frequency (Hz)');
    title(sprintf('STFT Spectrogram (Frame Size = %d, Frame Increment = %d)', N, M));

    % Find region with most energy
    [max_energy_val, max_idx] = max(stft_result_db(:));
    [freq_idx, time_idx] = ind2sub(size(stft_result_db), max_idx);
    max_energy_time = t(time_idx);
    max_energy_freq = f(freq_idx);

    fprintf('Frame size N=%d: Maximum energy at %.2f ms and %.2f Hz\n', ...
            N, max_energy_time, max_energy_freq);

    % Optional: Mark the maximum energy point
    hold on;
    plot(max_energy_time, max_energy_freq, 'r.', 'MarkerSize', 20);
    hold off;
end

% Plot spectrogram of trimmed speech signal
num_frames_trim = floor((length(y_trim) - N) / M) + 1;
stft_result_trim = zeros(N, num_frames_trim);

% Compute STFT for trimmed signal
for i = 1:num_frames_trim
    frame_start = (i-1)*M + 1;
    frame = y_trim(frame_start:frame_start+N-1) .* window;
    stft_result_trim(:,i) = abs(fft(frame)).^2;
end

stft_result_trim = stft_result_trim(1:N/2+1, :);
stft_result_trim_db = 10*log10(stft_result_trim);

% Create time and frequency vectors for plotting
t_trim = ((0:num_frames_trim-1) * M) / Fs * 1000;  % Time in milliseconds
f_trim = (0:N/2) * Fs / N;  % Frequency in Hz

% Plot spectrogram of trimmed signal
subplot(4,1,4);
imagesc(t_trim, f_trim, stft_result_trim_db);
axis xy;  % Put low frequencies at bottom
colorbar;
xlabel('Time (ms)');
ylabel('Frequency (Hz)');
title(sprintf('STFT Spectrogram of Trimmed Signal (Frame Size = %d, Frame Increment = %d)', N, M));

%% Test 3.1
% Plot Mel-spaced filterbank
numFilters = 20;

fig4 = figure;
subplot(2,1,1);
set(fig4, 'Position', [100, screenHeight-1000, 600, 400]);
mel_filters = melfb(numFilters, N, Fs);
plot(linspace(0, (Fs/2), N/2+1), mel_filters);
title('Mel-spaced filterbank'), xlabel('Frequency (Hz)');
ylabel('Amplitude');

% Plot theoretical Mel filterbank response
subplot(2,1,2);
hold on;

% Convert frequency to Mel scale
f_min = 0;
f_max = Fs / 2;
mel_min = 2595 * log10(1 + f_min / 700);
mel_max = 2595 * log10(1 + f_max / 700);

% Generate center frequencies on the Mel scale
mel_points = linspace(mel_min, mel_max, numFilters + 2);
f_points = 700 * (10.^(mel_points / 2595) - 1);
bins = floor((N + 1) * f_points / Fs);

% Plot theoretical triangular filters
for i = 2:numFilters + 1
    x1 = f_points(i - 1);
    x2 = f_points(i);
    x3 = f_points(i + 1);
    y = [0, 1, 0];
    
    plot([x1, x2, x3], y);
end
title('Theoretical Mel-spaced filter bank (Triangle Shape)');
xlabel('Frequency (Hz)');
ylabel('Amplitude');
hold off;

%% Test 3.2
% Plot spectrogram before wrapping
figure;
subplot(2,1,1);
imagesc(t, f, stft_result_db);
axis xy;  % Put low frequencies at bottom
colorbar;
xlabel('Time (ms)');
ylabel('Frequency (Hz)');
title('Spectrogram before the mel frequency wrapping');

mel_wrap = zeros(numFilters, num_frames);

for i = 1:num_frames
    power_spectrum = stft_result(:, i);
    mel_wrap(:, i) = mel_filters * power_spectrum;
end

% Plot power spectrum after Mel frequency wrapping
subplot(2,1,2);
imagesc(t, f, 10*log10(mel_wrap));
axis xy;  % Put low frequencies at bottom
colorbar;
xlabel('Time (ms)');
ylabel('Frequency (Hz)');
title('Spectrogram after the mel frequency wrapping');

%% Test 4
% Compute MFCC features
mfcc_coeff = 20;

% Initialize MFCC matrix
mfcc_features = zeros(mfcc_coeff-1, num_frames);

% Compute MFCC for each frame
for i = 1:num_frames
    % Apply mel filterbank
    mel_energies = mel_wrap(:, i);
    
    % Apply DCT to get MFCC
    mfcc_frames = dct(log(mel_energies));
    
    % Keep only the specified number of coefficients
    mfcc_features(:, i) = mfcc_frames(2:mfcc_coeff);
end

% Create time vector for plotting
t = ((0:num_frames-1) * M) / Fs * 1000;  % Time in milliseconds
    
% Plot MFCC features
fig3 = figure;
set(fig3, 'Position', [1300, screenHeight-500, 600, 400]);
subplot(2,1,1);
imagesc(t, 2:mfcc_coeff, mfcc_features);
colorbar;
xlabel('Time (ms)');
ylabel('MFCC Coefficient');
title(sprintf('MFCC Features (Frame Size = %d)', N));

% Compute MFCC features for trimmed signal
mfcc_features_trim = zeros(mfcc_coeff-1, num_frames_trim);

% Compute MFCC for each frame of trimmed signal
for i = 1:num_frames_trim
    % Power spectrum
    power_spectrum = stft_result_trim(:, i);
    
    % Apply mel filterbank
    mel_energies = mel_filters * power_spectrum;
    
    % Apply DCT to get MFCC
    mfcc_frames = dct(log(mel_energies));
    
    % Keep only the specified number of coefficients
    mfcc_features_trim(:, i) = mfcc_frames(2:mfcc_coeff);
end

% Create time vector for plotting
t_trim = ((0:num_frames_trim-1) * M) / Fs * 1000;  % Time in milliseconds

% Plot MFCC features for trimmed signal
subplot(2,1,2);
imagesc(t_trim, 2:mfcc_coeff, mfcc_features_trim);
colorbar;
xlabel('Time (ms)');
ylabel('MFCC Coefficient');
title(sprintf('MFCC Features of Trimmed Signal (Frame Size = %d)', N));

%% Test 5

colors = lines(length(speakerList));

fig5 = figure;
screenSize = get(0, 'ScreenSize');
screenHeight = screenSize(4);
set(fig5, 'Position', [700, screenHeight-1000, 1200, 400]);

subplot(1,2,1);
hold on;
for i = 1:length(speakerList)
    trainingFile = sprintf(trainingFiles, speakerList(i));
    [y, Fs] = audioread(trainingFile);
    
    % Extract MFCC features
    mfcc_features = mfcc(y, Fs, frameLength, numMelFilters, numMfccCoeffs);
    mfcc_features = mfcc_features';
    
    scatter(mfcc_features(:, dim1), mfcc_features(:, dim2), 10, colors(i,:));
end
title('Raw Audio MFCC Space');
xlabel(sprintf('MFCC - %d', dim1));
ylabel(sprintf('MFCC - %d', dim2));
legend(arrayfun(@(x) sprintf('Speaker %d', x), speakerList, 'UniformOutput', false));
grid on;
hold off;

subplot(1,2,2);
hold on;
for i = 1:length(speakerList)
    trainingFile = sprintf(trainingFiles, speakerList(i));
    [y, Fs] = autoTrimSilence(trainingFile);
    
    % Extract MFCC features
    mfcc_features = mfcc(y, Fs, frameLength, numMelFilters, numMfccCoeffs);
    mfcc_features = mfcc_features';
    
    scatter(mfcc_features(:, dim1), mfcc_features(:, dim2), 10, colors(i,:));
end
title('Trimmed Audio MFCC Space');
xlabel(sprintf('MFCC - %d', dim1));
ylabel(sprintf('MFCC - %d', dim2));
legend(arrayfun(@(x) sprintf('Speaker %d', x), speakerList, 'UniformOutput', false));
grid on;
hold off;

%% Test 6

allFeatures_Raw = [];
allLabels_Raw = [];
featuresCell_Raw = cell(length(speakerList), 1);

allFeatures_Trim = [];
allLabels_Trim = [];
featuresCell_Trim = cell(length(speakerList), 1);

colors = lines(length(speakerList) + 2);

% Loop over the speakers in speakerList (e.g., [2, 10])
for i = 1:length(speakerList)
    speaker = speakerList(i);
    audioFile = sprintf(trainingFiles, speaker);
    
    % Method 1: Read raw audio using audioread
    [y_raw, Fs] = audioread(audioFile);
    % Extract MFCC features from raw audio
    mfcc_raw = mfcc(y_raw, Fs, frameLength, numMelFilters, numMfccCoeffs);
    mfcc_raw = mfcc_raw';
    allFeatures_Raw = [allFeatures_Raw; mfcc_raw];
    allLabels_Raw = [allLabels_Raw; repmat(speaker, size(mfcc_raw, 1), 1)];
    featuresCell_Raw{i} = mfcc_raw;
    
    % Method 2: Read trimmed audio using autoTrimSilence
    [y_trim, Fs_trim] = autoTrimSilence(audioFile);
    % Extract MFCC features from trimmed audio
    mfcc_trim = mfcc(y_trim, Fs_trim, frameLength, numMelFilters, numMfccCoeffs);
    mfcc_trim = mfcc_trim';
    allFeatures_Trim = [allFeatures_Trim; mfcc_trim];
    allLabels_Trim = [allLabels_Trim; repmat(speaker, size(mfcc_trim, 1), 1)];
    featuresCell_Trim{i} = mfcc_trim;
end

% Separate features for Speaker 2 and Speaker 10 for raw method
features_spk2_Raw = allFeatures_Raw(allLabels_Raw == 2, :);
features_spk10_Raw = allFeatures_Raw(allLabels_Raw == 10, :);

% Separate features for Speaker 2 and Speaker 10 for trimmed method
features_spk2_Trim = allFeatures_Trim(allLabels_Trim == 2, :);
features_spk10_Trim = allFeatures_Trim(allLabels_Trim == 10, :);

% Compute VQ codebooks for raw audio features using LBG algorithm
codeword2_Raw = vq_lbg(features_spk2_Raw, targetCodebookSize, epsilon, tol);
codeword10_Raw = vq_lbg(features_spk10_Raw, targetCodebookSize, epsilon, tol);

% Compute VQ codebooks for trimmed audio features using LBG algorithm
codeword2_Trim = vq_lbg(features_spk2_Trim, targetCodebookSize, epsilon, tol);
codeword10_Trim = vq_lbg(features_spk10_Trim, targetCodebookSize, epsilon, tol);

% Create a figure with two horizontal subplots
fig = figure;
screenSize = get(0, 'ScreenSize');
screenHeight = screenSize(4);
set(fig, 'Position', [700, screenHeight-1200, 1200, 400]);

subplot(1,2,1);
hold on;
% Plot raw MFCC features for Speaker 2 and Speaker 10
scatter(featuresCell_Raw{1}(:, dim1), featuresCell_Raw{1}(:, dim2), 10, colors(1,:));
scatter(featuresCell_Raw{2}(:, dim1), featuresCell_Raw{2}(:, dim2), 10, colors(2,:));
% Overlay VQ codebooks for raw audio
scatter(codeword2_Raw(:, dim1), codeword2_Raw(:, dim2), 25, 'r', 'filled');
scatter(codeword10_Raw(:, dim1), codeword10_Raw(:, dim2), 25, 'g', 'filled');
title('Raw Audio MFCC Space with VQ Codebook');
xlabel(sprintf('MFCC - %d', dim1));
ylabel(sprintf('MFCC - %d', dim2));
legend({'Speaker 2', 'Speaker 10', 'Speaker 2 VQ', 'Speaker 10 VQ'}, 'Location', 'best');
grid on;
hold off;

subplot(1,2,2);
hold on;
% Plot trimmed MFCC features for Speaker 2 and Speaker 10
scatter(featuresCell_Trim{1}(:, dim1), featuresCell_Trim{1}(:, dim2), 10, colors(1,:));
scatter(featuresCell_Trim{2}(:, dim1), featuresCell_Trim{2}(:, dim2), 10, colors(2,:));
% Overlay VQ codebooks for trimmed audio
scatter(codeword2_Trim(:, dim1), codeword2_Trim(:, dim2), 25, 'r', 'filled');
scatter(codeword10_Trim(:, dim1), codeword10_Trim(:, dim2), 25, 'g', 'filled');
title('Trimmed Audio MFCC Space with VQ Codebook');
xlabel(sprintf('MFCC - %d', dim1));
ylabel(sprintf('MFCC - %d', dim2));
legend({'Speaker 2', 'Speaker 10', 'Speaker 2 VQ', 'Speaker 10 VQ'}, 'Location', 'best');
grid on;
hold off;