clear; clc; close all;

%% 1. Parameter Setting
% Audio Files
numTrainingFiles = 11;
numTestFiles     = 8;
% trainingFiles = './EEC201AudioRecordings/Eleven Training/s%d.wav'; 
% testFiles     = './EEC201AudioRecordings/Eleven Test/s%d.wav';
% trainingFiles = './EEC201AudioRecordings/Five Training/s%d.wav'; 
% testFiles     = './EEC201AudioRecordings/Five Test/s%d.wav';
% trainingFiles = './2024StudentAudioRecording/Zero-Training/Zero_train%d.wav'; 
% testFiles     = './2024StudentAudioRecording/Zero-Testing/Zero_test%d.wav';
% trainingFiles = './2024StudentAudioRecording/Twelve-Training/Twelve_train%d.wav'; 
% testFiles     = './2024StudentAudioRecording/Twelve-Testing/Twelve_test%d.wav';
trainingFiles   = './GivenSpeech_Data/Training_Data/s%d.wav'; 
testFiles       = './GivenSpeech_Data/Test_Data/s%d.wav';

% MFCC parameters
frameLength   = 512;     % Frame length in samples
numMelFilters = 20;      % Number of Mel filter banks
numMfccCoeffs = 20;      % Total number of MFCC coefficients
select_coef   = 1;       % Selector for frame filtering based on power (default: 1).

% VQ-LBG parameters
targetCodebookSize = 16;   % The desired number of codewords in the final codebook
epsilon            = 0.01; % Splitting parameter
tol                = 1e-3; % Iteration stopping threshold

% Notch filter parameters
f0 = 1500; % center frequency
Q  = 30;   % quality factor
R  = 1;    % Pole radius

%% 2. Build VQ codebooks for each training speaker

trainCodebooks = cell(numTrainingFiles, 1);

for i = 1:numTrainingFiles
    trainingFile = sprintf(trainingFiles, i);
    if exist(trainingFile, 'file')
        % Extract MFCC features for current speaker
        [y, Fs] = autoTrimSilence(trainingFile, frameLength);
        mfcc_training = mfcc(y, Fs, frameLength, numMelFilters, numMfccCoeffs, select_coef);
        
        % Compute VQ codebook for the current speaker using the LBG algorithm
        codebook = vq_lbg(mfcc_training', targetCodebookSize, epsilon, tol);
        trainCodebooks{i} = codebook;
    end
end

%% 3. Speaker Recognition on the Test Set

correct = 0;  % Counter for correct recognition
total = 0;    % Total number of test files

for i = 1:numTestFiles
    testFile = sprintf(testFiles, i);
    if exist(testFile, 'file')
        % Extract MFCC features for the test file
        [y, Fs] = autoTrimSilence(testFile, frameLength);
        mfcc_test = mfcc(y, Fs, frameLength, numMelFilters, numMfccCoeffs, select_coef);
        mfcc_test = mfcc_test'; 
        
        % Compute average distortion for each speaker's codebook
        distortions = inf(numTrainingFiles, 1);
        for spk = 1:numTrainingFiles
            if isempty(trainCodebooks{spk})
                continue;
            end
            % Retrieve the codebook for speaker spk
            cb = trainCodebooks{spk};
    
            % Compute Euclidean distances (squared) between test vectors and codebook vectors
            
            dists = pdist2(mfcc_test, cb, 'euclidean').^2;
    
            % For each test vector, take the minimum distance to any codeword
            min_dists = min(dists, [], 2);
            
            % Average distortion for this speaker's codebook
            distortions(spk) = mean(min_dists);
        end
        
        % The predicted speaker is the one with the minimum average distortion
        [~, predicted] = min(distortions);
        fprintf('True Speaker: %d, Predicted Speaker: %d\n', i, predicted);
        
        if predicted == i
            correct = correct + 1;
        end
        total = total + 1;
    end
end

recognitionRate = correct / total;
fprintf('Overall Recognition Rate: %.2f%%\n\n', recognitionRate * 100);

%% 4. Speaker Recognition on the Test Set with Notch Filter

correct = 0;  % Counter for correct recognition
total = 0;    % Total number of test files

for i = 1:numTestFiles
    testFile = sprintf(testFiles, i);
    if exist(testFile, 'file')
        % Read the audio file
        [y, Fs] = autoTrimSilence(testFile, frameLength);
        
        % Apply the notch filter to the audio signal
        y_filtered = NotchFilter(y, Fs, f0, Q);
        
        % Extract MFCC features from the filtered signal.
        mfcc_test = mfcc(y_filtered, Fs, frameLength, numMelFilters, numMfccCoeffs, select_coef);
        mfcc_test = mfcc_test';
        
        % Compute average distortion for each speaker's codebook (same as before)
        distortions = inf(numTrainingFiles, 1);
        for spk = 1:numTrainingFiles
            if isempty(trainCodebooks{spk})
                continue;
            end
            cb = trainCodebooks{spk};
            dists = pdist2(mfcc_test, cb, 'euclidean').^2;
            min_dists = min(dists, [], 2);
            distortions(spk) = mean(min_dists);
        end
        
        % The predicted speaker is the one with the minimum average distortion
        [~, predicted] = min(distortions);
        fprintf('True Speaker: %d, Predicted Speaker (Notch Filtered): %d\n', i, predicted);
        
        if predicted == i
            correct = correct + 1;
        end
        total = total + 1;
    end
end

recognitionRate = correct / total;
fprintf('Overall Recognition Rate: %.2f%%\n', recognitionRate * 100);

%% Function of Notch filter
function y_filtered = NotchFilter(y, Fs, f0, Q, R)

if nargin < 3
    f0 = 1500;
end
if nargin < 4
    Q = 30;
end
if nargin < 5
    R = 1;
end

W0 = f0 / (Fs/2);

% Compute the normalized bandwidth using the quality factor Q.
BW = W0 / Q;

% Design the notch filter
[b, a] = iirnotch(W0, BW, R);

%figure;
%freqz(b, a, 1024, Fs);
%title(sprintf('Notch Filter Frequency Response (f0 = %d Hz, Q = %d)', f0, Q));

% Apply the notch filter to the input signal
y_filtered = filter(b, a, y);
end