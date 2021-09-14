%% Loading dataset 

set(0,'DefaultFigureWindowStyle','docked') % Dock figures

data = load('MultivariateData.csv');
X = data(:, 1:2);
y = data(:, 3);
m = length(y);

%% Normalizing the data 
[X, mu, sigma] = NormalizeFeatures(X);

% Add intercept term to the Design Matrix X
X = [ones(m, 1) X];

%% Setting & Running Gradient Descent Parameters

% Run gradient descent
% Choose some alpha value
alpha = 0.1;
num_iters = 400;

% Init Theta and Run Gradient Descent 
theta = zeros(3, 1);
[theta, ~] = GradientDescentMultiple(X, y, theta, alpha, num_iters);

% Display gradient descent's result
fprintf('Theta computed from gradient descent:\n%f\n%f\n%f',theta(1),theta(2),theta(3))

%% Running Multiple Gradient Descents for diff. Learning Rates

% Run gradient descent:
% Choose some alpha value
alpha = [0.01, 0.05, 0.20];
num_iters = 100;

% Init Theta and Run Gradient Descent 
theta = zeros(3, 1);
[~, J_history_1] = GradientDescentMultiple(X, y, theta, alpha(1), num_iters);
[~, J_history_2] = GradientDescentMultiple(X, y, theta, alpha(2), num_iters);
[~, J_history_3] = GradientDescentMultiple(X, y, theta, alpha(3), num_iters);

%% Plot the convergence graph

plot(1:num_iters, J_history_1, '-g', 'LineWidth', 2);
title("Gradient Descent Convergence")
subtitle("Fixed Learning Rate")
xlabel('Number of iterations');
ylabel('Loss Function');
hold on
plot(1:num_iters, J_history_2, '-b', 'LineWidth', 2);
hold on
plot(1:num_iters, J_history_3, '-r', 'LineWidth', 2);
xlim([0 100])
hold off