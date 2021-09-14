%% Loading the dataset
set(0,'DefaultFigureWindowStyle','docked') % Dock figures

data = load('UnivariateData.csv'); % read comma separated data
X = data(:, 1);
y = data(:, 2);

%% Scatterplot
plot(X, y, 'rx', 'MarkerSize', 10, color="blue"); % Plot the data
  ylabel('Profit ($10,000s)'); % Set the y-axis label
  xlabel('City Population (10,000s)'); % Set the x-axis label

%% Initializing parameters for Gradient Descent
m = length(X); % n of training examples
X = [ones(m,1),data(:,1)]; % Add an intercept to the Design Matrix
theta = zeros(2, 1); % initialize the parameter vector
iterations = 5000;  % n of Gradient Descent iterations
alpha = 0.01; % Learning Rate

%% Computing Gradient Descent
[theta, loss_history] = GradientDescentSimple(X, y, theta, alpha, iterations);

fprintf('Theta computed from gradient descent:\n%f,\n%f',theta(1),theta(2))

%% Plotting the Gradient Descent Line of Best Fit
% Plot the linear regression fit
hold on; % keep previous plot
    plot(X(:,2), X*theta', '-', color="red", LineWidth=1.5)
    title("Gradient Descent Line of Best Fit")
legend('Training Data', 'Regression Line')
hold off % don't overlay any other plots

% Visualizing J(theta_0, theta_1):
% Grid over which we will calculate J
theta0_vals = linspace(-10, 10, 100);
theta1_vals = linspace(-1, 4, 100);

% initialize J_vals to a matrix of 0's
J_vals = zeros(length(theta0_vals), length(theta1_vals));

% Fill out J_vals
for i = 1:length(theta0_vals)
    for j = 1:length(theta1_vals)
	  t = [theta0_vals(i); theta1_vals(j)];
	  J_vals(i,j) = LossFunction(X, y, t);
    end
end

%% Plotting the Loss Function
J_vals = J_vals';

% Surface plot
figure;
surf(theta0_vals, theta1_vals, J_vals)
xlabel('\theta_0'); ylabel('\theta_1');

%% Contour plot
figure;
% Plot J_vals as 15 contours logarithmically spaced between 0.01 and 100
contour(theta0_vals, theta1_vals, J_vals, logspace(-2, 3, 20))
xlabel('\theta_0 (Intercept)'); ylabel('\theta_1 (Slope)');
hold on;
plot(theta(1), theta(2), 'rx', 'MarkerSize', 10, 'LineWidth', 2);
title("Minimum of the Linear Regression Loss Function")
hold off;
