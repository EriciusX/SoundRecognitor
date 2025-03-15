%% Main Script for Speaker Recognition Experiments
clear; clc; close all;

%% 1. Parameter Setting

% Audio Files for "eleven"
numTrainingFiles = 23;
numTestFiles = 23;
elevenTrainingFiles = './EEC201AudioRecordings/Eleven Training/s%d.wav'; 
elevenTestFiles     = './EEC201AudioRecordings/Eleven Test/s%d.wav';

% Audio Files for "five"
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

%% 2. Original Experiment: Recognition using "Eleven" and "Five" Utterances

% Build VQ codebooks for each training speaker
trainCodebooks_eleven = cell(numTrainingFiles, 1);
trainCodebooks_five = cell(numTrainingFiles, 1);

for i = 1:numTrainingFiles
    % Train codebook using "eleven" samples
    elevenTrainFile = sprintf(elevenTrainingFiles, i);
    if exist(elevenTrainFile, 'file')
        [y_eleven, Fs_eleven] = autoTrimSilence(elevenTrainFile, frameLength);
        mfcc_eleven = mfcc(y_eleven, Fs_eleven, frameLength, numMelFilters, numMfccCoeffs);
        codebook_eleven = vq_lbg(mfcc_eleven', targetCodebookSize, epsilon, tol);
        trainCodebooks_eleven{i} = codebook_eleven;
    end
    
    % Train codebook using "five" samples
    fiveTrainFile = sprintf(fiveTrainingFiles, i);
    if exist(fiveTrainFile, 'file')
        [y_five, Fs_five] = autoTrimSilence(fiveTrainFile, frameLength);
        mfcc_twelve = mfcc(y_five, Fs_five, frameLength, numMelFilters, numMfccCoeffs);
        codebook_twelve = vq_lbg(mfcc_twelve', targetCodebookSize, epsilon, tol);
        trainCodebooks_five{i} = codebook_twelve;
    end
end

combinedTrainCodebooks = cell(numTrainingFiles, 1);

for i = 1:numTrainingFiles
    % Initialize container for combined codebooks for speaker i
    combinedTrainCodebooks{i}.eleven = [];
    combinedTrainCodebooks{i}.five = [];
    
    % Process "eleven" training sample
    elevenTrainFile = sprintf(elevenTrainingFiles, i);
    if exist(elevenTrainFile, 'file')
        [y_eleven, Fs_eleven] = autoTrimSilence(elevenTrainFile, frameLength);
        mfcc_eleven = mfcc(y_eleven, Fs_eleven, frameLength, numMelFilters, numMfccCoeffs);
        codebook_eleven = vq_lbg(mfcc_eleven', targetCodebookSize, epsilon, tol);
        combinedTrainCodebooks{i}.eleven = codebook_eleven;
    end
    
    % Process "five" training sample
    fiveTrainFile = sprintf(fiveTrainingFiles, i);
    if exist(fiveTrainFile, 'file')
        [y_five, Fs_five] = autoTrimSilence(fiveTrainFile, frameLength);
        mfcc_five = mfcc(y_five, Fs_five, frameLength, numMelFilters, numMfccCoeffs);
        codebook_five = vq_lbg(mfcc_five', targetCodebookSize, epsilon, tol);
        combinedTrainCodebooks{i}.five = codebook_five;
    end
end

%% Question 1
% Test with "Eleven" Test Samples
correct_combined_eleven = 0;
total_combined_eleven = 0;
for i = 1:numTestFiles
    testElevenFile = sprintf(elevenTestFiles, i);
    if exist(testElevenFile, 'file')
        [y_eleven, Fs_eleven] = autoTrimSilence(testElevenFile, frameLength, 0.03);
        mfcc_test = mfcc(y_eleven, Fs_eleven, frameLength, numMelFilters, numMfccCoeffs, select_coef);
        mfcc_test = mfcc_test';  % Each row is a feature vector
        distortions = inf(numTrainingFiles, 1);
        for spk = 1:numTrainingFiles
            if isempty(trainCodebooks_eleven{spk})
                continue;
            end
            cb_combined = trainCodebooks_eleven{spk};
            dists = pdist2(mfcc_test, cb_combined, 'euclidean').^2;
            distortions(spk) = mean(min(dists, [], 2));
        end
        [~, predicted] = min(distortions);
        fprintf('Test ("Eleven") - True Speaker: %d, Predicted: %d\n', i, predicted);
        if predicted == i
            correct_combined_eleven = correct_combined_eleven + 1;
        end
        total_combined_eleven = total_combined_eleven + 1;
    end
end
accuracy_combined_eleven = correct_combined_eleven / total_combined_eleven;
fprintf('"Eleven" Test Accuracy: %.2f%%\n\n', accuracy_combined_eleven * 100);

% Test with "Five" Test Samples
correct_combined_five = 0;
total_combined_five = 0;
for i = 1:numTestFiles
    testFiveFile = sprintf(fiveTestFiles, i);
    if exist(testFiveFile, 'file')
        [y_five, Fs_five] = autoTrimSilence(testFiveFile, frameLength, 0.03);
        mfcc_test = mfcc(y_five, Fs_five, frameLength, numMelFilters, numMfccCoeffs, select_coef);
        mfcc_test = mfcc_test';  % Each row is a feature vector
        distortions = inf(numTrainingFiles, 1);
        for spk = 1:numTrainingFiles
            if isempty(trainCodebooks_five{spk})
                continue;
            end
            cb_combined = trainCodebooks_five{spk};
            dists = pdist2(mfcc_test, cb_combined, 'euclidean').^2;
            distortions(spk) = mean(min(dists, [], 2));
        end
        [~, predicted] = min(distortions);
        fprintf('Test ("Five") - True Speaker: %d, Predicted: %d\n', i, predicted);
        if predicted == i
            correct_combined_five = correct_combined_five + 1;
        end
        total_combined_five = total_combined_five + 1;
    end
end
accuracy_combined_five = correct_combined_five / total_combined_five;
fprintf('"Five" Test Accuracy: %.2f%%\n', accuracy_combined_five * 100);

%% Question 2
totalTests = 0;
correctSpeaker = 0;
correctWord = 0;
correctBoth = 0;

% Process "eleven" test files
for i = 1:numTestFiles
    testFile = sprintf(elevenTestFiles, i);
    if exist(testFile, 'file')
        [y_eleven, Fs_Test] = autoTrimSilence(testFile, frameLength, 0.03);
        mfcc_test = mfcc(y_eleven, Fs_Test, frameLength, numMelFilters, numMfccCoeffs, select_coef);
        mfcc_test = mfcc_test';  % Each row is a feature vector
        
        bestDistortion = Inf;
        predictedSpeaker = NaN;
        predictedWord = '';
        
        % For each speaker, compute distortions for both "eleven" and "five" codebooks.
        for spk = 1:numTrainingFiles
            if isempty(combinedTrainCodebooks{spk}.eleven) || isempty(combinedTrainCodebooks{spk}.five)
                continue;
            end
            
            % Distortion for "eleven" codebook
            cb_zero = combinedTrainCodebooks{spk}.eleven;
            dists_eleven = pdist2(mfcc_test, cb_zero, 'euclidean').^2;
            distortion_eleven = mean(min(dists_eleven, [], 2));
            
            % Distortion for "five" codebook
            cb_twelve = combinedTrainCodebooks{spk}.five;
            dists_five = pdist2(mfcc_test, cb_twelve, 'euclidean').^2;
            distortion_five = mean(min(dists_five, [], 2));
            
            % Choose the lower distortion between "eleven" and "five" for speaker spk
            if distortion_eleven < distortion_five
                currentDistortion = distortion_eleven;
                currentWord = 'eleven';
            else
                currentDistortion = distortion_five;
                currentWord = 'five';
            end
            
            % Update the best prediction if current distortion is lower
            if currentDistortion < bestDistortion
                bestDistortion = currentDistortion;
                predictedSpeaker = spk;
                predictedWord = currentWord;
            end
        end
        
        % True labels for "eleven" test sample: speaker i, word "eleven"
        trueSpeaker = i;
        trueWord = 'eleven';
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

% Process "five" test files
for i = 1:numTestFiles
    testFile = sprintf(fiveTestFiles, i);
    if exist(testFile, 'file')
        [y_five, Fs_five] = autoTrimSilence(testFile, frameLength, 0.03);
        mfcc_test = mfcc(y_five, Fs_five, frameLength, numMelFilters, numMfccCoeffs, select_coef);
        mfcc_test = mfcc_test';  % Each row is a feature vector
        
        bestDistortion = Inf;
        predictedSpeaker = NaN;
        predictedWord = '';
        
        for spk = 1:numTrainingFiles
            if isempty(combinedTrainCodebooks{spk}.eleven) || isempty(combinedTrainCodebooks{spk}.five)
                continue;
            end
            
            cb_eleven = combinedTrainCodebooks{spk}.eleven;
            dists_eleven = pdist2(mfcc_test, cb_eleven, 'euclidean').^2;
            distortion_eleven = mean(min(dists_eleven, [], 2));
            
            cb_five = combinedTrainCodebooks{spk}.five;
            dists_five = pdist2(mfcc_test, cb_five, 'euclidean').^2;
            distortion_five = mean(min(dists_five, [], 2));
            
            if distortion_eleven < distortion_five
                currentDistortion = distortion_eleven;
                currentWord = 'eleven';
            else
                currentDistortion = distortion_five;
                currentWord = 'five';
            end
            
            if currentDistortion < bestDistortion
                bestDistortion = currentDistortion;
                predictedSpeaker = spk;
                predictedWord = currentWord;
            end
        end
        
        % True labels for "five" test sample: speaker i, word "five"
        trueSpeaker = i;
        trueWord = 'five';
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