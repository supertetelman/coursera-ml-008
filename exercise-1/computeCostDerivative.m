%ml-008 Exercise1
%Based off of sample code provided by coursera Machine Learning Course
%ml-008 taught by Andrew NG of Stanford
%@author Adam Tetelman 2/10/2015

function [ value ] = computeCostDerivative(X,y,theta, i)
%Returns a computed value for the derivative of the cost function at a
%given index

m = length(X);
value = 0;

for j=1:m
    value = value + (X(j,:)*theta - y(j)) * X(j,i);
end

value = value/m;
end

