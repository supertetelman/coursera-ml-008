%Based off of sample code provided by coursera Machine Learning Course
%ml-008 taught by Andrew NG of Stanford
%@author Adam Tetelman 3/12/2015

function [X_poly] = polyFeatures(X, p)
%POLYFEATURES Maps X (1D vector) into the p-th power
%   [X_poly] = POLYFEATURES(X, p) takes a data matrix X (size m x 1) and
%   maps each example into its polynomial features where
%   X_poly(i, :) = [X(i) X(i).^2 X(i).^3 ...  X(i).^p];
%

% You need to return the following variables correctly.
X_poly = zeros(numel(X), p);

for i=1:p
    X_poly(:,i) = X .^ i;
end
