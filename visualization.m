clear; clc; close all;

%% Parameters

numTrainingFiles = 11;                                       % Number of training files
trainingFiles = './GivenSpeech_Data/Training_Data/s%d.wav';  % Files

% MFCC parameters
frameLength = 512;       % Frame length in samples
numMelFilters = 20;      % Number of Mel filter banks
numMfccCoeffs = 13;      % Total number of MFCC coefficients

% VQ-LBG parameters
targetCodebookSize = 8;  % 目标码本大小（码元个数）
epsilon = 0.01;          % 分裂参数
tol = 1e-3;              % 迭代收敛容忍度

%% Test 5

dim1 = 6;
dim2 = 7;
speakerList = [2, 10];
colors = lines(length(speakerList));

figure;
hold on;
for i = 1:length(speakerList)
    trainingFile = sprintf(trainingFiles, speakerList(i));

    mfcc_features = mfcc_selected(trainingFile, frameLength, numMelFilters, numMfccCoeffs);
    mfcc_features = mfcc_features';
    scatter(mfcc_features(:, dim1), mfcc_features(:, dim2), 10, colors(i,:));
end

title('MFCC Space');
xlabel(sprintf('MFCC - %d', dim1));
ylabel(sprintf('MFCC - %d', dim2));
legend(arrayfun(@(x) sprintf('Speaker %d', x), speakerList, 'UniformOutput', false));
grid()
hold off;

%% Test 6

allFeatures = [];
allLabels = [];

featuresCell = cell(length(speakerList),1);
colors = lines(length(speakerList)+2);

for i = 1:length(speakerList)
    speaker = speakerList(i);
    audioFile = sprintf(trainingFiles, speaker);
    
    % Extract MFCC features; mfcc_features is (numMfccCoeffs-1) x num_frames
    mfcc_features = mfcc_selected(audioFile, frameLength, numMelFilters, numMfccCoeffs);
    % Transpose so that each row is one frame's feature vector
    mfcc_features = mfcc_features';
    
    % Append features to the combined matrix
    allFeatures = [allFeatures; mfcc_features];
    allLabels = [allLabels; repmat(speaker, size(mfcc_features, 1), 1)];
    featuresCell{i} = mfcc_features;
end

features_spk2 = allFeatures(allLabels == 2, :);
features_spk10 = allFeatures(allLabels == 10, :);

codeword2 = vq_lbg(features_spk2, targetCodebookSize, epsilon, tol);
codeword10 = vq_lbg(features_spk10, targetCodebookSize, epsilon, tol);

figure;
hold on;

scatter(featuresCell{1}(:, dim1), featuresCell{1}(:, dim2), 10, colors(1,:));
scatter(featuresCell{2}(:, dim1), featuresCell{2}(:, dim2), 10, colors(2,:));

% Overlay the VQ codewords with larger red filled markers
scatter(codeword2(:, dim1), codeword2(:, dim2), 25, 'r', 'filled');
scatter(codeword10(:, dim1), codeword10(:, dim2), 25, 'g', 'filled');

title('MFCC Space with VQ Codebook');
xlabel(sprintf('MFCC - %d', dim1));
ylabel(sprintf('MFCC - %d', dim2));
legend('Speaker2', 'Speaker10', 'Speaker2 VQ Codewords', 'Speaker10 VQ Codewords');
grid()
hold off;
