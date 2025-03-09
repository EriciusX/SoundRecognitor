clear; clc; close all;

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

%% 1. Parameter Setting

% Audio Files
numTrainingFiles = 11;
numTestFiles = 8;
trainingFiles = './GivenSpeech_Data/Training_Data/s%d.wav'; 
testFiles     = './GivenSpeech_Data/Test_Data/s%d.wav';

% MFCC parameters
frameLength = 512;       % Frame length in samples
numMelFilters = 20;      % Number of Mel filter banks
numMfccCoeffs = 20;      % Total number of MFCC coefficients
select_coef   = 0.8;     % Selector for frame filtering based on power (default: 1).

% VQ-LBG parameters
targetCodebookSize = 16; % The desired number of codewords in the final codebook
epsilon = 0.01;          % Splitting parameter
tol = 1e-3;              % Iteration stopping threshold

% Notch filter parameters
f0 = 2000; % center frequency
Q = 30;    % quality factor
R = 1;     % Pole radius

%% 2. Build VQ codebooks for each training speaker

trainCodebooks = cell(numTrainingFiles, 1);

for i = 1:numTrainingFiles
    trainingFile = sprintf(trainingFiles, i);
  
    % Extract MFCC features for current speaker
    mfcc_training = mfcc(trainingFile, frameLength, numMelFilters, numMfccCoeffs);
    
    % Compute VQ codebook for the current speaker using the LBG algorithm
    codebook = vq_lbg(mfcc_training', targetCodebookSize, epsilon, tol);
    trainCodebooks{i} = codebook;
end

%% 3. Speaker Recognition on the Test Set

correct = 0;  % Counter for correct recognition
total = 0;    % Total number of test files

for i = 1:numTestFiles
    testFile = sprintf(testFiles, i);
    
    % Extract MFCC features for the test file
    mfcc_test = mfcc(testFile, frameLength, numMelFilters, numMfccCoeffs, select_coef);
    mfcc_test = mfcc_test'; 
    
    % Compute average distortion for each speaker's codebook
    distortions = zeros(numTrainingFiles, 1);
    for spk = 1:numTrainingFiles

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

recognitionRate = correct / total;
fprintf('Overall Recognition Rate: %.2f%%\n\n', recognitionRate * 100);

%% 4. Speaker Recognition on the Test Set with Notch Filter

correct = 0;  % Counter for correct recognition
total = 0;    % Total number of test files

for i = 1:numTestFiles
    testFile = sprintf(testFiles, i);
    
    % Read the audio file
    [y, Fs] = audioread(testFile);
    
    % Apply the notch filter to the audio signal
    y_filtered = NotchFilter(y, Fs, f0, Q);
    
    % Extract MFCC features from the filtered signal.
    mfcc_test = mfcc(y_filtered, frameLength, numMelFilters, numMfccCoeffs, select_coef);
    mfcc_test = mfcc_test';
    
    % Compute average distortion for each speaker's codebook (same as before)
    distortions = zeros(numTrainingFiles, 1);
    for spk = 1:numTrainingFiles
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

recognitionRate = correct / total;
fprintf('Overall Recognition Rate: %.2f%%\n', recognitionRate * 100);
