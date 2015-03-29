%ml-008 Exercise7
%Based off of sample code provided by coursera Machine Learning Course
%ml-008 taught by Andrew NG of Stanford
%@author Adam Tetelman 3/28/2015
function centroids = computeCentroids(X, idx, K)
%COMPUTECENTROIDS returs the new centroids by computing the means of the 
%data points assigned to each centroid.
%   centroids = COMPUTECENTROIDS(X, idx, K) returns the new centroids by 
%   computing the means of the data points assigned to each centroid. It is
%   given a dataset X where each row is a single data point, a vector
%   idx of centroid assignments (i.e. each entry in range [1..K]) for each
%   example, and K, the number of centroids. You should return a matrix
%   centroids, where each row of centroids is the mean of the data points
%   assigned to it.
%

% Useful variables
[m n] = size(X);

% You need to return the following variables correctly.
centroids = zeros(K, n);

for i=1:K
    
    %Create a m x n matrix with 1s mapping to elements in the centroid
    tmp = ones(m,n);
    for j=1:n
        tmp(:,j) = (idx(:) == ones(m,1) .* i);
    end
    
    X_k = X .* ones(m,n) .* tmp;
    centroids(i,:) = sum(X_k) * (1/sum(tmp(:,1)));
end


end

