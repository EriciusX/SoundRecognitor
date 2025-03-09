%% Main Script for Speaker Recognition Experiments
clear; clc; close all;

%% 1. Parameter Setting

% Audio Files for "zero"
numTrainingFiles = 23;
numTestFiles = 23;
elevenTrainingFiles = './EEC201AudioRecordings/Eleven Training/s%d.wav'; 
elevenTestFiles     = './EEC201AudioRecordings/Eleven Test/s%d.wav';

% Audio Files for "twelve"
fiveTrainingFiles = './EEC201AudioRecordings/Five Training/s%d.wav'; 
fiveTestFiles     = './EEC201AudioRecordings/Five Test/s%d.wav';

% MFCC parameters
frameLength = 512;        % Frame length in samples
numMelFilters = 20;       % Number of Mel filter banks
numMfccCoeffs = 20;       % Total number of MFCC coefficients
select_coef   = 1;        % Selector for frame filtering based on power

% VQ-LBG parameters
targetCodebookSize = 16;  % Desired number of codewords in the final codebook
epsilon = 0.01;           % Splitting parameter
tol = 1e-3;               % Iteration stopping threshold

% Notch filter parameters (for later experiments)
f0 = 1500;  % Center frequency in Hz
Q  = 30;    % Quality factor
R  = 1;     % Pole radius

%% 2. Original Experiment: Recognition using "Eleven" Utterances

% Build VQ codebooks for each training speaker (using "zero" samples)
trainCodebooks_zero = cell(numTrainingFiles, 1);
trainCodebooks_twelve = cell(numTrainingFiles, 1);

for i = 1:numTrainingFiles
    % Train codebook using "zero" samples
    zeroTrainFile = sprintf(elevenTrainingFiles, i);
    if exist(zeroTrainFile, 'file')
        mfcc_zero = mfcc(autoTrimSilence(zeroTrainFile), frameLength, numMelFilters, numMfccCoeffs);
        codebook_zero = vq_lbg(mfcc_zero', targetCodebookSize, epsilon, tol);
        trainCodebooks_zero{i} = codebook_zero;
    end
    
    % Train codebook using "twelve" samples
    twelveTrainFile = sprintf(fiveTrainingFiles, i);
    if exist(twelveTrainFile, 'file')
        mfcc_twelve = mfcc(autoTrimSilence(twelveTrainFile), frameLength, numMelFilters, numMfccCoeffs);
        codebook_twelve = vq_lbg(mfcc_twelve', targetCodebookSize, epsilon, tol);
        trainCodebooks_twelve{i} = codebook_twelve;
    end
end

combinedTrainCodebooks = cell(numTrainingFiles, 1);

for i = 1:numTrainingFiles
    % Initialize container for combined codebooks for speaker i
    combinedTrainCodebooks{i}.zero = [];
    combinedTrainCodebooks{i}.twelve = [];
    
    % Process "zero" training sample
    zeroTrainFile = sprintf(elevenTrainingFiles, i);
    if exist(zeroTrainFile, 'file')
        mfcc_zero = mfcc(autoTrimSilence(zeroTrainFile), frameLength, numMelFilters, numMfccCoeffs);
        codebook_zero = vq_lbg(mfcc_zero', targetCodebookSize, epsilon, tol);
        combinedTrainCodebooks{i}.zero = codebook_zero;
    end
    
    % Process "twelve" training sample
    twelveTrainFile = sprintf(fiveTrainingFiles, i);
    if exist(twelveTrainFile, 'file')
        mfcc_twelve = mfcc(autoTrimSilence(twelveTrainFile), frameLength, numMelFilters, numMfccCoeffs);
        codebook_twelve = vq_lbg(mfcc_twelve', targetCodebookSize, epsilon, tol);
        combinedTrainCodebooks{i}.twelve = codebook_twelve;
    end
end

%% Question 1
% Test with "Eleven" Test Samples
correct_combined_zero = 0;
total_combined_zero = 0;
for i = 1:numTestFiles
    testZeroFile = sprintf(elevenTestFiles, i);
    if exist(testZeroFile, 'file')
        mfcc_test = mfcc(autoTrimSilence(testZeroFile), frameLength, numMelFilters, numMfccCoeffs, select_coef);
        mfcc_test = mfcc_test';  % Each row is a feature vector
        distortions = inf(numTrainingFiles, 1);
        for spk = 1:numTrainingFiles
            if isempty(trainCodebooks_zero{spk})
                continue;
            end
            cb_combined = trainCodebooks_zero{spk};
            dists = pdist2(mfcc_test, cb_combined, 'euclidean').^2;
            distortions(spk) = mean(min(dists, [], 2));
        end
        [~, predicted] = min(distortions);
        fprintf('Combined Test ("Eleven") - True Speaker: %d, Predicted: %d\n', i, predicted);
        if predicted == i
            correct_combined_zero = correct_combined_zero + 1;
        end
        total_combined_zero = total_combined_zero + 1;
    end
end
accuracy_combined_zero = correct_combined_zero / total_combined_zero;
fprintf('"Eleven" Test Accuracy: %.2f%%\n\n', accuracy_combined_zero * 100);

% Test with "Five" Test Samples
correct_combined_twelve = 0;
total_combined_twelve = 0;
for i = 1:numTestFiles
    testTwelveFile = sprintf(fiveTestFiles, i);
    if exist(testTwelveFile, 'file')
        mfcc_test = mfcc(autoTrimSilence(testTwelveFile), frameLength, numMelFilters, numMfccCoeffs, select_coef);
        mfcc_test = mfcc_test';  % Each row is a feature vector
        distortions = inf(numTrainingFiles, 1);
        for spk = 1:numTrainingFiles
            if isempty(trainCodebooks_twelve{spk})
                continue;
            end
            cb_combined = trainCodebooks_twelve{spk};
            dists = pdist2(mfcc_test, cb_combined, 'euclidean').^2;
            distortions(spk) = mean(min(dists, [], 2));
        end
        [~, predicted] = min(distortions);
        fprintf('Combined Test ("Five") - True Speaker: %d, Predicted: %d\n', i, predicted);
        if predicted == i
            correct_combined_twelve = correct_combined_twelve + 1;
        end
        total_combined_twelve = total_combined_twelve + 1;
    end
end
accuracy_combined_twelve = correct_combined_twelve / total_combined_twelve;
fprintf('"Five" Test Accuracy: %.2f%%\n', accuracy_combined_twelve * 100);

%% Question 2
totalTests = 0;
correctSpeaker = 0;
correctWord = 0;
correctBoth = 0;

% Process "zero" test files
for i = 1:numTestFiles
    testFile = sprintf(elevenTestFiles, i);
    if exist(testFile, 'file')
        mfcc_test = mfcc(autoTrimSilence(testFile), frameLength, numMelFilters, numMfccCoeffs, select_coef);
        mfcc_test = mfcc_test';  % Each row is a feature vector
        
        bestDistortion = Inf;
        predictedSpeaker = NaN;
        predictedWord = '';
        
        % For each speaker, compute distortions for both "zero" and "twelve" codebooks.
        for spk = 1:numTrainingFiles
            if isempty(combinedTrainCodebooks{spk}.zero) || isempty(combinedTrainCodebooks{spk}.twelve)
                continue;
            end
            
            % Distortion for "zero" codebook
            cb_zero = combinedTrainCodebooks{spk}.zero;
            dists_zero = pdist2(mfcc_test, cb_zero, 'euclidean').^2;
            distortion_zero = mean(min(dists_zero, [], 2));
            
            % Distortion for "twelve" codebook
            cb_twelve = combinedTrainCodebooks{spk}.twelve;
            dists_twelve = pdist2(mfcc_test, cb_twelve, 'euclidean').^2;
            distortion_twelve = mean(min(dists_twelve, [], 2));
            
            % Choose the lower distortion between "zero" and "twelve" for speaker spk
            if distortion_zero < distortion_twelve
                currentDistortion = distortion_zero;
                currentWord = 'zero';
            else
                currentDistortion = distortion_twelve;
                currentWord = 'twelve';
            end
            
            % Update the best prediction if current distortion is lower
            if currentDistortion < bestDistortion
                bestDistortion = currentDistortion;
                predictedSpeaker = spk;
                predictedWord = currentWord;
            end
        end
        
        % True labels for "zero" test sample: speaker i, word "zero"
        trueSpeaker = i;
        trueWord = 'zero';
        fprintf('Combined Test (Eleven) - True: (Speaker %d, %s), Predicted: (Speaker %d, %s)\n', ...
            trueSpeaker, trueWord, predictedSpeaker, predictedWord);
        totalTests = totalTests + 1;
        if predictedSpeaker == trueSpeaker
            correctSpeaker = correctSpeaker + 1;
        end
        if strcmp(predictedWord, trueWord)
            correctWord = correctWord + 1;
        end
        if (predictedSpeaker == trueSpeaker) && strcmp(predictedWord, trueWord)
            correctBoth = correctBoth + 1;
        end
    end
end

% Process "twelve" test files
for i = 1:numTestFiles
    testFile = sprintf(fiveTestFiles, i);
    if exist(testFile, 'file')
        mfcc_test = mfcc(autoTrimSilence(testFile), frameLength, numMelFilters, numMfccCoeffs, select_coef);
        mfcc_test = mfcc_test';  % Each row is a feature vector
        
        bestDistortion = Inf;
        predictedSpeaker = NaN;
        predictedWord = '';
        
        for spk = 1:numTrainingFiles
            if isempty(combinedTrainCodebooks{spk}.zero) || isempty(combinedTrainCodebooks{spk}.twelve)
                continue;
            end
            
            cb_zero = combinedTrainCodebooks{spk}.zero;
            dists_zero = pdist2(mfcc_test, cb_zero, 'euclidean').^2;
            distortion_zero = mean(min(dists_zero, [], 2));
            
            cb_twelve = combinedTrainCodebooks{spk}.twelve;
            dists_twelve = pdist2(mfcc_test, cb_twelve, 'euclidean').^2;
            distortion_twelve = mean(min(dists_twelve, [], 2));
            
            if distortion_zero < distortion_twelve
                currentDistortion = distortion_zero;
                currentWord = 'zero';
            else
                currentDistortion = distortion_twelve;
                currentWord = 'twelve';
            end
            
            if currentDistortion < bestDistortion
                bestDistortion = currentDistortion;
                predictedSpeaker = spk;
                predictedWord = currentWord;
            end
        end
        
        % True labels for "twelve" test sample: speaker i, word "twelve"
        trueSpeaker = i;
        trueWord = 'twelve';
        fprintf('Combined Test (Five) - True: (Speaker %d, %s), Predicted: (Speaker %d, %s)\n', ...
            trueSpeaker, trueWord, predictedSpeaker, predictedWord);
        totalTests = totalTests + 1;
        if predictedSpeaker == trueSpeaker
            correctSpeaker = correctSpeaker + 1;
        end
        if strcmp(predictedWord, trueWord)
            correctWord = correctWord + 1;
        end
        if (predictedSpeaker == trueSpeaker) && strcmp(predictedWord, trueWord)
            correctBoth = correctBoth + 1;
        end
    end
end

% Calculate overall accuracies
speakerAccuracy = correctSpeaker / totalTests;
wordAccuracy = correctWord / totalTests;
jointAccuracy = correctBoth / totalTests;
fprintf('Combined System - Speaker Accuracy: %.2f%%\n', speakerAccuracy * 100);
fprintf('Combined System - Word Accuracy: %.2f%%\n', wordAccuracy * 100);
fprintf('Combined System - Joint (Speaker & Word) Accuracy: %.2f%%\n', jointAccuracy * 100);


%% Function of autoTrimSilence
function [trimmedSignal, Fs] = autoTrimSilence(audioFile, frameSize, overlapRatio, percentage)
% AUTOTRIMSILENCE Automatically trims silence at the beginning and end of an audio file.
%
% Inputs:
%   audioFile   : Path to the input audio file (string).
%   frameSize   : Number of samples in each frame (e.g., 512).
%   overlapRatio: Overlap ratio for consecutive frames (e.g., 0.66 means 66% overlap).
%                 Default is 2/3 if not specified.
%
% Output:
%   trimmedSignal: Audio signal after removing silent parts from the beginning and the end.

    if nargin < 2
        frameSize = 512;
    end
    if nargin < 3
        overlapRatio = 2/3; 
    end
    if nargin < 4
        percentage = 0.03; 
    end

    % Read the audio file
    [y, Fs] = audioread(audioFile);
    % Normalize the waveform to avoid amplitude issues
    y = y / (max(abs(y)) + eps);

    % Define frame increment based on overlap ratio
    increment = round(frameSize * (1 - overlapRatio));

    % Compute the number of frames
    numFrames = floor((length(y) - frameSize) / increment) + 1;

    % Pre-allocate array for short-time energy
    energy = zeros(numFrames, 1);

    % Calculate short-time energy for each frame
    for i = 1:numFrames
        startIndex = (i - 1) * increment + 1;
        frame = y(startIndex : startIndex + frameSize - 1);
        energy(i) = sum(frame .^ 2);
    end

    % Set a threshold, e.g., 1% of the maximum energy
    threshold = percentage * max(energy);

    % Find frames that exceed the threshold
    voicedFrames = find(energy >= threshold);

    % If no frames exceed the threshold, the signal might be entirely silent
    if isempty(voicedFrames)
        trimmedSignal = [];
        return;
    end

    % Identify the first and last frames that exceed the threshold
    firstFrame = min(voicedFrames);
    lastFrame  = max(voicedFrames);

    % Convert frame indices to sample indices
    startSample = (firstFrame - 1) * increment + 1;
    endSample   = (lastFrame - 1) * increment + frameSize;

    % Trim the signal
    trimmedSignal = y(startSample : endSample);
end