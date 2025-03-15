function [trimmedSignal, Fs] = autoTrimSilence(audioFile, frameSize, thresholdFactor, overlapRatio)
% AUTOTRIMSILENCE Automatically trims silence at the beginning and end of an audio file.
%
% Inputs:
%   audioFile       : Path to the input audio file (string)
%   frameSize       : Number of samples in each frame (e.g., 512)
%   overlapRatio    : Overlap ratio for consecutive frames (e.g., 0.66 means 66% overlap)
%   thresholdFactor : The fraction of the maximum energy used as a threshold 
%                     (e.g., 0.01 means 1% of max energy)
%
% Output:
%   trimmedSignal: Audio signal after removing silent parts from the beginning and the end

    if nargin < 2
        frameSize = 512;
    end
    if nargin < 3
        thresholdFactor = 0.01;  % 1% of the maximum energy
    end    
    if nargin < 4
        overlapRatio = 2/3; 
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
    threshold = thresholdFactor * max(energy);

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
