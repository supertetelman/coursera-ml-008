%ml-008 Exercise3
%Based off of sample code provided by coursera Machine Learning Course
%ml-008 taught by Andrew NG of Stanford
%@author Adam Tetelman 2/22/2015

function [all_theta] = oneVsAll(X, y, num_labels, lambda)
% Some useful variables
[m n] = size(X);
all_theta = zeros(num_labels, n + 1);
X = [ones(m, 1) X];

%DEBUG
%sprintf('Computing all_theta for X of size %d x %d\nwith lambda %d and %d training rows\nof num_labels:%d', m, n, lambda, m, num_labels)

for c = 1:num_labels
   tmp_theta = zeros(1,n + 1)'; %tmp theta specifically for this classifier
   
   options = optimset('GradObj', 'on', 'MaxIter', 50);%options for fmincg
   
   %Compute theta for classifier c and add the results to all_theta
   %fminc will use varying values of tmp_theta against lrcost function
   % at each classifier to compute the minimum cost and correspondingg
   % theta
   all_theta(c,:) = fmincg(@(t)(lrCostFunction(t, X, (y == c), ...
        lambda)), tmp_theta, options);    
end
end
