function plotData(x, y)


figure; 

  plot(x, y, 'rx', 'MarkerSize', 10, color="blue"); % Plot the data
  ylabel('Profit ($10,000s)'); % Set the y-axis label
  xlabel('City Population (10,000s)'); % Set the x-axis label

end
