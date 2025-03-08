function mfcc_features = mfcc_selected(file_name, N, num_mel_filters, mfcc_coeff)

    if nargin < 4
        mfcc_coeff = 13;
    end
    if nargin < 3
        num_mel_filters = 20;
    end
    if nargin < 2
        N = 512;
    end

    selected = 0.8;
    [y, Fs] = audioread(file_name);

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
    threshold = quantile(energy_per_frame, 1 - selected);

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
        selected_mfcc_features(:, i) = mfcc(2:end);
    end

    mfcc_features = selected_mfcc_features;
end