clear; clc; close all;

%% Parameters

numTrainingFiles = 11;
numTestFiles     = 8;
trainingFiles = './GivenSpeech_Data/Training_Data/s%d.wav';  % Files
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

%% Test 2
% Load and plot speech signal in time domain
[y, Fs] = audioread(sprintf(trainingFiles, 1));
sound(y, Fs);

% Normalize
y = y / max(abs(y));

% Calculate time for 256 samples 
time_ms = (256 / Fs) * 1000;
fprintf('Sampling rate: %d Hz\n', Fs);
fprintf('Duration of 256 samples: %.2f milliseconds\n', time_ms);

% Plot signal in time domain
t = (0:length(y)-1) / Fs;  
fig1 = figure;
set(fig1, 'Position', [100, screenHeight-500, 600, 400]);
plot(t, y);
xlim([0 max(t)]);
xlabel('Time (s)');
ylabel('Amplitude');
title('Signal for s1.wav in Time Domain');

% Plot spectrogram of speech signal
N = 512;  % Frame size
M = round(N/3); % frame increment

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

% Plot spectrogram
fig2 = figure;
set(fig2, 'Position', [700, screenHeight-500, 600, 400]);
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

%% Test 3
% Plot Mel-spaced filterbank
fig4 = figure;
set(fig4, 'Position', [100, screenHeight-1000, 600, 400]);
mel_filter = melfb(20, N, Fs);
plot(linspace(0, (Fs/2), N/2+1), melfb(20, 512, 12500)');
title('Mel-spaced filterbank'), xlabel('Frequency (Hz)');

%% Test 4
% Compute MFCC features
% Parameters for melfb
num_mel_filters = 20;
mfcc_coeff = 20;

% Initialize MFCC matrix
mfcc_features = zeros(mfcc_coeff-1, num_frames);

% Get mel filterbank
mel_filters = melfb(num_mel_filters, N, Fs);

% Compute MFCC for each frame
for i = 1:num_frames
    frame_start = (i-1)*M + 1;
    frame = y(frame_start:frame_start+N-1) .* window;
    
    % Power spectrum
    power_spectrum = abs(fft(frame)).^2;
    power_spectrum = power_spectrum(1:N/2+1);
    
    % Apply mel filterbank
    mel_energies = mel_filters * power_spectrum;
    
    % Apply DCT to get MFCC
    mfcc_frames = dct(log(mel_energies(1:mfcc_coeff)));
    
    % Keep only the specified number of coefficients
    mfcc_features(:, i) = mfcc_frames(2:end);
end

% Create time vector for plotting
t = ((0:num_frames-1) * M) / Fs * 1000;  % Time in milliseconds
    
% Plot MFCC features
fig3 = figure;
set(fig3, 'Position', [1300, screenHeight-500, 600, 400]);
imagesc(t, 2:mfcc_coeff, mfcc_features);
colorbar;
xlabel('Time (ms)');
ylabel('MFCC Coefficient');
title(sprintf('MFCC Features (Frame Size = %d)', N));

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