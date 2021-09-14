function J = LossFunction(X, y, theta)


% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% Initialize
m = length(y); % number of training examples

% Loss Function for Linear Regression
J = sum((X*theta - y).^2)/(2*m);

end
