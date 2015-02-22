%ml-008 Exercise3
%Based off of sample code provided by coursera Machine Learning Course
%ml-008 taught by Andrew NG of Stanford
%@author Adam Tetelman 2/22/2015

function [J, grad] = lrCostFunction(theta, X, y, lambda)
% Initialize some useful values
m = length(y); % number of training examples
J = -1;
grad = zeros(size(theta));
H = classhyp(X,theta);
y_zeros = 1 - y;% y matrix where 0/1 are swapped

%Compute non-regularized J and gradient 
J = sum((-log(H) .* y) + (-log(1-H) .* y_zeros)); 
grad =  X' * (H - y); 

%Regularize J and grad (but no on theta(1)
J = J + sum(lambda*(theta(2:end).^2)/2);
grad(2:end) = grad(2:end) + lambda*theta(2:end);

%Divide out by number of training rows
J = J/m;
grad = grad/m;
end
