function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

[m n ] = size(X);        
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

%TODO: Vectorize and Optimize this code
first_user = 1;%Only regulariz theta once
for i=1:num_movies
    X_grad(i,:) = X_grad(i,:) + lambda*X(i,:);
    for j=1:num_users
    if first_user == 1
        Theta_grad(j,:) = Theta_grad(j,:) + lambda*Theta(j,:);
        
    end
        if R(i,j) == 0
            continue
        end
        cost = (X(i,:)*Theta(j,:)' - Y(i,j));
        J = J + cost ^ 2 ;
        Theta_grad(j,:) = Theta_grad(j,:) + cost * X(i,:);
        X_grad(i,:) = X_grad(i,:) + cost * Theta(j,:);
    end
    first_user = 0;
end
J = J + lambda*sum(sum(X .^ 2)) + lambda*sum(sum(Theta .^2));

J = J/2;

grad = [X_grad(:); Theta_grad(:)];

end
