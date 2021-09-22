import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import OLSInfluence

df = pd.read_csv("./data/Boston.csv")

#%% Plotting the data

plt.scatter(x=df.lstat, y=df.medv, c="red")
plt.title("Simple Linear Regression")
plt.xlabel("% of Lower Status Population")
plt.ylabel("Median House Value (1000$)")
plt.show()

#%% Fitting the Linear Regression

lm = smf.ols(formula="medv ~ lstat", data=df).fit()

#%% Plotting the Least Squares Line of Best Fit

plt.plot(df.lstat, df.medv, "o", c="tab:purple")
plt.plot(df.lstat, lm.fittedvalues, c="red", linewidth=3)
plt.title("Ordinary Least Squares Line of Best Fit", fontweight="bold")
plt.xlabel("% of Lower Status Population", c="dimgray", fontweight="bold")
plt.ylabel("Median House Value (1000$)", c="dimgray", fontweight="bold")
plt.show()

#%% Summary of the OLS Regression

print(lm.summary())

#%% Parameters & Hypothesis Test Statistics

print(f"Regression Coefficients: {[round(x, 3) for x in lm.params.values]}")

print(f"P-values: {lm.pvalues.values}")

print(f"F-Statistic: {round(lm.fvalue, 3)}")

print(f"F-Statistic (p-value): {lm.f_pvalue}")

#%% Goodness of Fit Metrics

print(f"R-Square: {round(lm.rsquared, 2)}")

print(f"Adjusted R-Square: {round(lm.rsquared_adj, 2)}")

print(f"Mean-Squared Error: {lm.resid.map(lambda e: e ** 2).mean().round(3)}")

print(f"Root Mean-Squared Error: {np.sqrt(lm.resid.map(lambda e: e ** 2).mean()).round(3)}")

print(f"Residual Standard Deviation: {round(np.sqrt(lm.mse_resid), 3)}") # mse_resid divides by n - k - 1 (residual df)

print(f"Akaike Information Criterion: {round(lm.aic, 2)}")

print(f"Bayesian Information Criterion: {round(lm.bic, 2)}")

#%% Confidence & Prediction Intervals

prediction = lm.get_prediction(exog={"lstat":5}, transform=True)

summary = prediction.summary_frame(alpha=0.05)

# Predicting the Mean Response
print(f"Mean: {prediction.predicted_mean.round(3)}")


# 95% Confidence Interval
print(f"95% Confidence Interval: {prediction.conf_int(obs=False, alpha=0.05).round(2)}")

#95% Prediction Interval
print(f"95% Prediction Interval: {prediction.conf_int(obs=True, alpha=0.05).round(2)}")

# Prints the fitted value & both the 95% Confidence & Prediction Interval
print(summary)

#%% Residuals vs Fitted Values

# Plot shows high evidence of non-linearity
plt.scatter(lm.predict(), lm.resid, color="darkblue")

plt.title("Residuals vs Fits", fontweight="bold")

plt.axhline(y=0, color="red", linestyle="dashed")

plt.xlabel("Fitted Values")

plt.ylabel("Residuals")

plt.show()

#%% Influence Measures
influence = OLSInfluence(lm)

# Difference in Fits Dataframe (Influence)
print(f"Dfitts: {influence.dffits}")

# Cooks'Distance List: (Influence & Leverage)
print(f"Cook's Distance: {influence.cooks_distance}")

# Hat-Values List: (Leverage)
print(f"Hat-Values: {influence.hat_matrix_diag}")