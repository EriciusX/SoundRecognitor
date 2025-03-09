function codebook = vq_lbg(mfcc, M, epsilon, tol)
% Vector Quantization codebook design based on the LBG algorithm
%
% Input:
%   mfcc    : MFCC matrix
%   M       : The desired number of codewords in the final codebook
%   epsilon : Splitting parameter (e.g., 0.01) used to perturb codewords
%   tol     : Iteration stopping threshold (e.g., 1e-3) for checking
%             the relative improvement of the average distortion
%
% Output:
%   codebook : An M x d matrix, each row is one final codeword

if nargin < 2
    M = 8;
end
if nargin < 3
    epsilon = 0.01;
end
if nargin < 4
    tol = 1e-3;
end

% Initialize with the mean of all feature vectors
codebook = mean(mfcc, 1);
currentSize = 1;

% Grow the codebook until we have M codewords
while currentSize < M
    % Split each existing codeword into two:
    codebook = [codebook * (1 + epsilon); codebook * (1 - epsilon)];
    currentSize = size(codebook, 1);

    % Refine the codebook using iterative updating
    prevDistortion = Inf;
    while true
        % Calculate squared Euclidean distances between each feature vector and each codeword
        distances = pdist2(mfcc, codebook, 'euclidean').^2;

        % For each feature vector, find the closest codeword
        [minDists, idx] = min(distances, [], 2);

        % Calculate the average distortion.
        currentDistortion = mean(minDists);
        
        %  Update codewords by computing the mean of the assigned vectors
        newCodebook = zeros(size(codebook));
        for i = 1:currentSize
            assignedVectors = mfcc(idx == i, :);
            if ~isempty(assignedVectors)
                newCodebook(i, :) = mean(assignedVectors, 1);
            else
                newCodebook(i, :) = codebook(i, :);
            end
        end
        
        % Replace the old codebook with the new one.
        codebook = newCodebook;
        
        % Check the convergence condition
        if abs(prevDistortion - currentDistortion) / currentDistortion < tol
            break;
        end

        % Update previous distortion for the next iteration.
        prevDistortion = currentDistortion;
    end
    
    % If the codebook size exceeds M, truncate it to the first M codewords.
    if currentSize > M
        codebook = codebook(1:M, :);
        currentSize = M;
    end
end

end
