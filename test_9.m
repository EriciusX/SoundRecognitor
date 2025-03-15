clear; clc; close all;

%% 1. Parameter Setting

% Audio Files
numTrainFiles = 11;
numTestFiles = 8;
trainFiles = './GivenSpeech_Data/Training_Data/s%d.wav'; 
testFiles     = './GivenSpeech_Data/Test_Data/s%d.wav';

% Student recordings from 2024 (10 students with "zero" speech)
numStudentFiles = 10;

% Generate random numbers for student files
availableIndices = setdiff(1:19, 5);
studentFileIndices = availableIndices(randperm(length(availableIndices), numStudentFiles));
studentTrainFiles = './2024StudentAudioRecording/Zero-Training/Zero_train%d.wav';
studentTestFiles = './2024StudentAudioRecording/Zero-Testing/Zero_test%d.wav';

num_train_all = numTrainFiles + numStudentFiles;
num_test_all = numTestFiles + numStudentFiles;
trainSpeakers = cell(num_train_all, 1);
testSpeakers = cell(numTestFiles, 1);
TrainPath = cell(num_train_all, 1);
TestPath = cell(num_train_all, 1);

% Add student training files to the map
for i = 1:numStudentFiles
    speakerID = sprintf('2024 student s%d', studentFileIndices(i));
    trainSpeakers{i} = speakerID;
    testSpeakers{i} = speakerID;
    TrainPath{i} = sprintf(studentTrainFiles, studentFileIndices(i));
    TestPath{i} = sprintf(studentTestFiles, studentFileIndices(i));
end

% Add non-student training files to the map
for i = 1:numTrainFiles
    speakerID = sprintf('non-student s%d', i);
    trainSpeakers{i+numStudentFiles} = speakerID;
    TrainPath{i+numStudentFiles} = sprintf(trainFiles, i);
end

% Add non-student test files to the map
for i = 1:numTestFiles
    testSpeakers{i+numStudentFiles} = sprintf('non-student s%d', i);
    TestPath{i+numStudentFiles} = sprintf(testFiles, i);
end

% speaker used in training
fprintf('Speaker used in training set:\n');
for i = 1:num_train_all
    fprintf('%s\n', trainSpeakers{i});
end
fprintf('\n');

% MFCC parameters
frameLength = 512;       % Frame length in samples
numMelFilters = 20;      % Number of Mel filter banks
numMfccCoeffs = 20;      % Total number of MFCC coefficients
select_coef   = 1;     % Selector for frame filtering based on power (default: 1).

% VQ-LBG parameters
targetCodebookSize = 8; % The desired number of codewords in the final codebook
epsilon = 0.01;          % Splitting parameter
tol = 1e-3;              % Iteration stopping threshold

% Notch filter parameters
f0 = 1500; % center frequency
Q = 30;    % quality factor
R = 1;     % Pole radius

%% 2. Build VQ codebooks for each training speaker

trainCodebooks = cell(num_train_all, 1);

for i = 1:num_train_all
    trainingFile = TrainPath{i};
    [y, Fs] = autoTrimSilence(trainingFile);
    % Extract MFCC features for current speaker
    mfcc_training = mfcc(y, Fs, frameLength, numMelFilters, numMfccCoeffs);
    
    % Compute VQ codebook for the current speaker using the LBG algorithm
    codebook = vq_lbg(mfcc_training', targetCodebookSize, epsilon, tol);
    trainCodebooks{i} = codebook;
end

%% 3. Speaker Recognition on the Test Set

correct = 0;  % Counter for correct recognition
total = 0;    % Total number of test files

for i = 1:num_test_all
    testFile = TestPath{i};
    [y, Fs] = autoTrimSilence(testFile);
    % Extract MFCC features for the test file
    mfcc_test = mfcc(y, Fs, frameLength, numMelFilters, numMfccCoeffs);
    mfcc_test = mfcc_test'; 
    
    % Compute average distortion for each speaker's codebook
    distortions = zeros(num_test_all, 1);
    for spk = 1:num_train_all

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
    fprintf('True Speaker: %s, Predicted Speaker: %s\n', testSpeakers{i}, trainSpeakers{predicted});
    
    if predicted == i
        correct = correct + 1;
    end
    total = total + 1;
end

recognitionRate = correct / total;
fprintf('Overall Recognition Rate: %.2f%%\n\n', recognitionRate * 100);
