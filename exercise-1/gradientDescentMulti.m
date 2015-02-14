%ml-008 Exercise1
%Based off of sample code provided by coursera Machine Learning Course
%ml-008 taught by Andrew NG of Stanford
%@author Adam Tetelman 2/10/2015

function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)

% Initialize some useful values
m = length(y); % number of training examples
n = length(theta); %number of attributes
J_history = zeros(num_iters, 1);

%Perform gradients steps until the numer of iteratinos of a 0 cost is found
for iter = 1:num_iters
    cost = computeCost(X,y,theta);
    J_history(iter) = cost; %Save the cost at each iteration

    %Print out the cost x iter as debug
    %sprintf('DEBUG: cost=%0.2f at iter:%d', cost, iter) 
    
    %stop work if we found the optimal theta
    if cost == 0
        break
    end
  
    %Update the theta values
    tmp_theta = theta; %Don't change theta inbetween updates
    for val = 1:n
        theta(val) = tmp_theta(val) - alpha *...
            computeCostDerivative(X,y,tmp_theta,val);
    end
end

