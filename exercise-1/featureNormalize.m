%ml-008 Exercise1
%Based off of sample code provided by coursera Machine Learning Course
%ml-008 taught by Andrew NG of Stanford
%@author Adam Tetelman 2/10/2015

function [X_norm, mu, sigma] = featureNormalize(X)
[m n] = size(X);

%Calculate mu and sigma and fill a X sized matrix for easier math 
one_X = ones(1,m);
mu = (mean(X)'*one_X)';
sigma = (std(X)'*one_X)';

X_norm = (X - mu)./sigma;

%Return only 1 row
mu = mu(1,:);
sigma = sigma(1,:);

end
