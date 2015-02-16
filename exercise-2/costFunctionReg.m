%ml-008 Exercise2
%Based off of sample code provided by coursera Machine Learning Course
%ml-008 taught by Andrew NG of Stanford
%@author Adam Tetelman 2/15/2015
function [J, grad] = costFunctionReg(theta, X, y, lambda)
% Initialize some useful values
J = 0;
grad = zeros(size(theta));
m = length(y); % number of training examples
cost = sigmoid(X*theta); %cost matrix
n = length(theta);

for i=1:m    
    J = J + (-y(i)*log(cost(i,:))) - ((1-y(i))*log(1-cost(i,:)));
    grad(:) = grad(:) + (cost(i) - y(i)) * X(i,:)';%use matrix math
end

%Regularize the cost and gradient (note we do not regularize theta(0)
J = J + sum(lambda*(theta(2:n).^2)/2);
grad(2:n) = grad(2:n) + lambda*theta(2:n);


%Finish up by dividing by number of samples
J = J/m;
grad = grad/m;

end

