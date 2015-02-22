%ml-008 Exercise3
%Based off of sample code provided by coursera Machine Learning Course
%ml-008 taught by Andrew NG of Stanford
%@author Adam Tetelman 2/22/2015

function p = predictOneVsAll(all_theta, X)

m = size(X, 1);
num_labels = size(all_theta, 1);

p = zeros(m, 1); %results table containing the prediction for each row
all_p = zeros(m,num_labels);%tmp table containing all probablities

% Add ones to the X data matrix
X = [ones(m, 1) X];

for c=1:num_labels
    all_p(:,c) = classhyp(X,all_theta(c,:)');
end

%The column corresponds to the class for each sample
%We want to calculate which column has the max for each row and set
%the index of that row to be our predictions
[max_prob max_index] = max(all_p');
p = max_index';
end
