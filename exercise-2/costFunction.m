%ml-008 Exercise2
%Based off of sample code provided by coursera Machine Learning Course
%ml-008 taught by Andrew NG of Stanford
%@author Adam Tetelman 2/15/2015

function [J, grad] = costFunction(theta, X, y)

% Initialize some useful values
J = 0;
grad = zeros(size(theta));
m = length(y); % number of training examples
cost = sigmoid(X*theta); %cost matrix

for i=1:m    
    %Compute most of the cost function
    J = J + (-y(i)*log(cost(i,:))) - ((1-y(i))*log(1-cost(i,:)));
    grad(:) = grad(:) + (cost(i) - y(i)) * X(i,:)';%use matrix math

%   %Solve for gradient by iterating through samples
%   for j=1:length(theta)
%       grad(j) = grad(j) + (cost(i) - y(i)) * X(i,j)';
%   %end
end

%Finish up by dividing by number of samples
J = J/m;
grad = grad/m;

end
