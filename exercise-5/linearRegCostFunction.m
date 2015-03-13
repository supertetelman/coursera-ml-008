%ml-008 Exercise5
%Based off of sample code provided by coursera Machine Learning Course
%ml-008 taught by Andrew NG of Stanford
%@author Adam Tetelman 3/12/2015

function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

H = X*theta;

J = (sum((H - y) .^2)  + (lambda * sum((theta(2:end,:) .^ 2)))) / ( 2*m );
grad(1) = (sum((H - y) .* X(:,1))) / m;
for i=2:length(theta)
    grad(i) = (sum((H - y) .* X(:,i)) + lambda * sum(theta(i,:)))/m ;
end


grad = grad(:);

end
