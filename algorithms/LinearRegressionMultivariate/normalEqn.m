function [theta] = normalEqn(X, y)


theta = zeros(size(X, 2), 1);

% Solution for the Normal Equations
theta = inv(X'*X)*X'*y

end
