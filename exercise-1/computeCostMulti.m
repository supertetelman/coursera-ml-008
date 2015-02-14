%ml-008 Exercise1
%Based off of sample code provided by coursera Machine Learning Course
%ml-008 taught by Andrew NG of Stanford
%@author Adam Tetelman 2/10/2015

function J = computeCostMulti(X, y, theta)

m = length(y) ; % number of training examples
n = length(theta) ; % number of variables
J = 0; %set J to 0 initially

%Solve for J by iterating/summing through all training data
for i=1:m
    %solve using math & function substitution
    %J = J +((theta(1,1)*X(i,1)+theta(2,1)*X(i,2)-y(i))^2)/(2*m); 
    
    %Solve using matrix algebra
    J = J + ((theta'*X(i,:)'-y(i))^2)/(2*m);
end

end
