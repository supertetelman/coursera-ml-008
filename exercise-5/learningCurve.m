%Based off of sample code provided by coursera Machine Learning Course
%ml-008 taught by Andrew NG of Stanford
%@author Adam Tetelman 3/12/2015

function [error_train, error_val] = ...
    learningCurve(X, y, Xval, yval, lambda)
%LEARNINGCURVE Generates the train and cross validation set errors needed 
%to plot a learning curve
%   [error_train, error_val] = ...
%       LEARNINGCURVE(X, y, Xval, yval, lambda) returns the train and
%       cross validation set errors for a learning curve. In particular, 
%       it returns two vectors of the same length - error_train and 
%       error_val. Then, error_train(i) contains the training error for
%       i examples (and similarly for error_val(i)).
%
%   In this function, you will compute the train and test errors for
%   dataset sizes from 1 up to m. In practice, when working with larger
%   datasets, you might want to do this in larger intervals.
%

% Number of training examples
m = size(X, 1);

% You need to return these values correctly
error_train = zeros(m, 1);
error_val   = zeros(m, 1);

for i=1:m
    %Use regularization when determining lambda
    theta = trainLinearReg(X(1:i,:),y(1:i,:),lambda);
    
    %For calculating the error don't use regularization and only use 
    %The ith training set
    error_train(i) = linearRegCostFunction(X(1:i,:),y(1:i,:),theta,0);
    
    %Use the whole cross validation set for each given theta, no
    %regularization
    error_val(i) = linearRegCostFunction(Xval,yval,theta,0);
end

end
