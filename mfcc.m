function mfcc_features = mfcc(y, Fs, N, num_mel_filters, mfcc_coeff, select_coef)
    % Reads an audio file or a Signal and computes its Mel-Frequency Cepstral Coefficients (MFCCs)
    %
    % Inputs:
    %   y               - Signal
    %   Fs              - Sample Rate
    %   N               - Frame size (default: 512)
    %   num_mel_filters - Number of Mel filters (default: 20)
    %   mfcc_coeff      - Number of MFCC coefficients (default: 13)
    %   select_coef     - Selector for frame filtering based on power (default: 1)
    %
    % Output:
    %   mfcc_features   - Matrix of MFCC features for the selected frames

    if nargin < 6
        select_coef = 1;
    end
    if nargin < 5
        mfcc_coeff = 13;
    end
    if nargin < 4
        num_mel_filters = 20;
    end
    if nargin < 3
        N = 512;
    end

    % Normalize
    y = y / max(abs(y)); 

    M = round(N/3); % frame increment
    num_frames = floor((length(y) - N) / M) + 1;
    window = hamming(N);

    % Initialize MFCC matrix
    mfcc_features = zeros(mfcc_coeff-1, num_frames);

    % Get mel filterbank
    mel_filters = melfb(num_mel_filters, N, Fs);

    stft_result = zeros(N, num_frames);

    % Compute STFT
    for i = 1:num_frames
        frame_start = (i-1)*M + 1;
        frame = y(frame_start:frame_start+N-1) .* window;
        stft_result(:,i) = abs(fft(frame)).^2;
    end

    stft_result = stft_result(1:N/2+1, :);

    % Compute the energy of each frame
    energy_per_frame = sum(stft_result, 1);

    % Compute the threshold
    threshold = quantile(energy_per_frame, 1 - select_coef);

    % Select frame indices where energy is above or equal to the threshold 
    selected_frames_indices = find(energy_per_frame >= threshold);

    % Extract the selected frames
    selected_stft_result = stft_result(:, selected_frames_indices);

    % Compute MFCC for each frame
    for i = 1:length(selected_frames_indices)
        power_spectrum = selected_stft_result(:, i);
        
        % Apply mel filterbank
        mel_energies = mel_filters * power_spectrum;
        
        % Apply DCT to get MFCC
        mfcc = dct(log(mel_energies(1:mfcc_coeff)));
        
        % Keep only the specified number of coefficients
        mfcc_features(:, i) = mfcc(2:end);
    end
end