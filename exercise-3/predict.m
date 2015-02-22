%ml-008 Exercise3
%Based off of sample code provided by coursera Machine Learning Course
%ml-008 taught by Andrew NG of Stanford
%@author Adam Tetelman 2/22/2015

function p = predict(Theta1, Theta2, X)
%Take a data set X, two calculated layer Thetas, and make a preciction

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);
p = zeros(size(X, 1), 1);

%We need to pad the values of  X to account for the bias param
X_pad = [ones(m,1) X ];

%Compute the hypotheses at the hidden layer
a2 = classhyp(X_pad, Theta1');

%We need to pad at this layer as well
a2_pad = [ones(m,1) a2]; 

%Compute the hypothesis at layer 3
a3 = classhyp(a2_pad, Theta2');

%determine which values had the highest probablity and return that as
%The prediction
[max_prob max_index] = max(a3');
p = max_index';


end
