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

    % % Compute MFCC for each frame
    % for i = 1:num_frames
    %     frame_start = (i-1)*M + 1;
    %     frame = y(frame_start:frame_start+N-1) .* window;
        
    %     % Power spectrum
    %     power_spectrum = abs(fft(frame)).^2;
    %     power_spectrum = power_spectrum(1:N/2+1);
        
    %     % Apply mel filterbank
    %     mel_energies = mel_filters * power_spectrum;
        
    %     % Apply DCT to get MFCC
    %     mfcc = dct(log(mel_energies(1:mfcc_coeff)));
        
    %     % Keep only the specified number of coefficients
    %     mfcc_features(:, i) = mfcc(2:end);
    % end

    % Select frames with high energy
    energy_per_frame = sum(stft_result, 1);
    [sorted_energy, sorted_indices] = sort(energy_per_frame, 'descend');
    num_selected_frames = ceil(num_frames*selected);
    selected_frames_indices = sorted_indices(1:num_selected_frames);

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

    mfcc_features = selected_mfcc_features;
end