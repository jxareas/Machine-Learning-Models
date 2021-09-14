function lossFunction = LossFunction(X, y, theta)
%LOSSFUNCTION Computes the Loss Function for linear regression

% Initialize some useful values
m = length(y); % number of training observations

lossFunction = 0;

lossFunction = sum((theta(1)*X(:, 1) + theta(2)*X(:, 2)- y).^2)/(2*m);

end
