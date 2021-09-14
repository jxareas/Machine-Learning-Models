function [X_norm, mu, sigma] = NormalizeFeatures(X)
% Standardizes the data so that each column vector of X ~ N(0, 1)
X_norm = X;
mu = zeros(1, size(X, 2));
sigma = zeros(1, size(X, 2));

mu = mean(X);
sigma = std(X);

X_norm = (X - mu)./sigma;

end
