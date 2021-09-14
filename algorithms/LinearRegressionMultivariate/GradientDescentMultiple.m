function [theta, J_history] = GradientDescentMultiple(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs multivariable gradient descent to learn the 
% values of the regression coefficients (theta vector)

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    gradientJ = 1/(m) * (X'*X*theta - X'*y);
     theta = theta - alpha * gradientJ;

    % Save the cost J in every iteration    
    J_history(iter) = LossFunction(X, y, theta);

end

end
