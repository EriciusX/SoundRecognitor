clear; clc; close all;

%% Load and plot speech signal in time domain
[y, Fs] = audioread('GivenSpeech_Data\Test_Data\s1.wav');
sound(y, Fs);

% Normalize
y = y / max(abs(y));

% Calculate time for 256 samples 
time_ms = (256 / Fs) * 1000;
fprintf('Sampling rate: %d Hz\n', Fs);
fprintf('Duration of 256 samples: %.2f milliseconds\n', time_ms);

% Plot signal in time domain
t = (0:length(y)-1) / Fs;  
figure;
plot(t, y);
xlim([0 max(t)]);
xlabel('Time (s)');
ylabel('Amplitude');
title('Signal for s1.wav in Time Domain');

%% Plot spectrogram of speech signal
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
figure;
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

%% Plot Mel-spaced filterbank
figure;
mel_filter = melfb(20, N, Fs);
plot(linspace(0, (Fs/2), N/2+1), melfb(20, 512, 12500)');
title('Mel-spaced filterbank'), xlabel('Frequency (Hz)');

%% Compute MFCC features
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
    mfcc = dct(log(mel_energies(1:mfcc_coeff)));
    
    % Keep only the specified number of coefficients
    mfcc_features(:, i) = mfcc(2:end);
end

% Create time vector for plotting
t = ((0:num_frames-1) * M) / Fs * 1000;  % Time in milliseconds
    
% Plot MFCC features
figure;
imagesc(t, 2:mfcc_coeff, mfcc_features);
colorbar;
xlabel('Time (ms)');
ylabel('MFCC Coefficient');
title(sprintf('MFCC Features (Frame Size = %d)', N));

%% Select frames with highest energy
% Select frames with high energy
energy_per_frame = sum(stft_result, 1);
[sorted_energy, sorted_indices] = sort(energy_per_frame, 'descend');
num_selected_frames = ceil(num_frames*0.8);
selected_frames_indices = sorted_indices(1:num_selected_frames);

% Extract the selected frames
selected_stft_result_db = stft_result_db(:, selected_frames_indices);

% Plot the selected frames' spectrogram
figure;
imagesc(t(selected_frames_indices), f, selected_stft_result_db);
axis xy;
colorbar;
xlabel('Time (ms)');
ylabel('Frequency (Hz)');
title('STFT Spectrogram of Selected Frames with Highest Energy');

%% Compute MFCC features for selected frames
% Initialize MFCC matrix
selected_mfcc_features = zeros(mfcc_coeff-1, num_selected_frames);

% Compute MFCC for each frame
for i = 1:num_selected_frames
    frame_idx = selected_frames_indices(i);
    power_spectrum = stft_result(:, frame_idx);
    
    % Apply mel filterbank
    mel_energies = mel_filters * power_spectrum;
    
    % Apply DCT to get MFCC
    mfcc = dct(log(mel_energies(1:mfcc_coeff)));
    
    % Keep only the specified number of coefficients
    selected_mfcc_features(:, i) = mfcc(2:end);
end

% Create time vector for plotting
t = ((selected_frames_indices-1) * M) / Fs * 1000;  % Time in milliseconds

% Plot MFCC features of selected frames
figure;
imagesc(t, 2:mfcc_coeff, selected_mfcc_features);
colorbar;
xlabel('Time (ms)');
ylabel('MFCC Coefficient');
title(sprintf('MFCC Features of Selected Frames (Frame Size = %d)', N));
