function [theta, cost_history] = GradientDescentSimple(X, y, theta, learning_rate, n_iterations)

% Initializing
m = length(y); % number of training examples
cost_history = zeros(n_iterations, 1); % Vector that stores the loss function values

for iter = 1:n_iterations
    x = X(:, 2);
    hypothesis = theta(1) + theta(2)*x;
    
    t1 = theta(1) - (learning_rate/m) * sum(hypothesis - y);
    t2 = theta(2) - (learning_rate/m) * sum((hypothesis - y).*x);
    theta = [t1, t2];
    
    % Save the loss
    cost_history(iter) = LossFunction(X, y, theta);
end
    cost_history(n_iterations);
end
