clear; clc; close all;

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

% VQ-LBG parameters
targetCodebookSize = 8;  % 目标码本大小（码元个数）
epsilon = 0.01;          % 分裂参数
tol = 1e-3;              % 迭代收敛容忍度

%% 2. Build VQ codebooks for each training speaker

trainCodebooks = cell(numTrainingFiles, 1);

for i = 1:numTrainingFiles
    trainingFile = sprintf(trainingFiles, i);
  
    % Extract MFCC features for current speaker
    mfcc_training = mfcc_selected(trainingFile, frameLength, numMelFilters, numMfccCoeffs);
    
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
    mfcc_test = mfcc_selected(testFile, frameLength, numMelFilters, numMfccCoeffs);
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
fprintf('Overall Recognition Rate: %.2f%%\n', recognitionRate * 100);
