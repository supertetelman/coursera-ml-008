%ml-008 Exercise1
%Based off of sample code provided by coursera Machine Learning Course
%ml-008 taught by Andrew NG of Stanford
%@author Adam Tetelman 2/10/2015

%Normalize a single row given a precalculated mu/sigma
function X_norm = rowNormalize(X, mu, sigma)
[m n] = size(X);

X_norm = (X - mu)./sigma;

end
