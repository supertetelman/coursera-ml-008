%ml-008 Exercise3
%Based off of sample code provided by coursera Machine Learning Course
%ml-008 taught by Andrew NG of Stanford
%@author Adam Tetelman 2/22/2015

function  H = classhyp(X,theta)
    H = sigmoid(X * theta);
end