%ml-008 Exercise4
%Based off of sample code provided by coursera Machine Learning Course
%ml-008 taught by Andrew NG of Stanford
%@author Adam Tetelman 3/5/2015

function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

%Calculate the Hypthosesis and hidden layer vaules
a1 = X;
a1_pad = [ones(length(a1),1) a1];
a2 = sigmoid(a1_pad*Theta1');
a2_pad = [ones(length(a2),1) a2];
H = sigmoid(a2_pad*Theta2');

%Create a matrix indicating actual results
Y = ones(length(H),num_labels);
for i=1:num_labels
    Y(:,i) = y == i;
end

%Calculate the deltas
delta_3 = H - Y;
Z = sigmoidGradient(a1_pad*Theta1');
delta_2 = delta_3*Theta2 .* [ ones(length(Z),1) Z];
Theta1_grad =    (a1_pad' * delta_2(:,2:end))';
Theta2_grad =   (a2_pad' * delta_3)';

%Regularize 
Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + lambda*Theta1(:,2:end);
Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + lambda*Theta2(:,2:end);

%Do a computation for each training row and each possible outcome (0-10)
for i=1:m
    Y = (y(i,1) == 1:num_labels);
    J = J + sum(-Y .* log(H(i,:)) - (1-Y) .* log(1-H(i,:)));
end

%Regularize cost from theta values 
%Exclude the first column bias value
J = J + (lambda*(sum(sum(Theta1(:,2:end) .^ 2)) + ...
    sum(sum(Theta2(:,2:end) .^ 2))))/2;

J=J/m;

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)] ./m;

end
