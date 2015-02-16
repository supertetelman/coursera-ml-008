%ml-008 Exercise2
%Based off of sample code provided by coursera Machine Learning Course
%ml-008 taught by Andrew NG of Stanford
%@author Adam Tetelman 2/15/2015
function p = predict(theta, X)
m = size(X, 1); % Number of training examples

% You need to return the following variables correctly
p = zeros(m, 1);

cost = sigmoid(X*theta);%Calculate cost for all values

%Set prediction to be 1/0 based on .5 threshold
for i=1:m
    predict = -1;
    if cost(i) < .5; predict = 0; end
    if cost(i) >= .5; predict = 1; end
    p(i,1) = predict;
end
