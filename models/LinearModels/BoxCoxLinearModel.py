import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats.mstats import normaltest
from scipy.stats import boxcox
from scipy.special import inv_boxcox
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (mean_squared_error, r2_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (StandardScaler, PolynomialFeatures)
from sklearn.datasets import load_boston


# Preprocessing the data

def to_2d(array):
    return array.reshape(array.shape[0], -1)


def load_boston_df(description=False):
    boston = load_boston()

    data = boston.data
    target = to_2d(boston.target)
    names = boston.feature_names

    all_data = np.concatenate([data, target], axis=1)
    all_names = np.concatenate([names, np.array(['MEDV'])], axis=0)

    if description:

        return pd.DataFrame(data=all_data, columns=all_names), boston.DESCR

    else:

        return pd.DataFrame(data=all_data, columns=all_names)


df = load_boston_df(description=False)
# %% Assessing Normality of the Response Variable (medv: median house price)

# Visualizing the Kernel Density Estimation, which shows a right-skewed distribution
sns.kdeplot(data=df.MEDV, color="red")
plt.title("Density Estimation of the Median House Price", fontweight="bold")
plt.xlabel("Median House Price")
plt.ylabel("Density")
plt.show()

# Testing Normality using D'Agostino's K-Squared Test
# The higher the p-value, the closer to the Normal Distribution
print(normaltest(df.MEDV))

# %% Transforming the response using the Box-Cox Power Transform
# & Testing for Normality
df = df.assign(boxcox_medv=boxcox(df.MEDV)[0])
lambda_parameter = boxcox(df.MEDV)[1]

# Plotting the KDE
sns.kdeplot(data=df.boxcox_medv, color="blue")
plt.title("Box Cox Transformation of Median House Price", fontweight="bold")
plt.xlabel("Box-Cox Transformed Median House Price")
plt.ylabel("Density")
plt.show()

# p-value ~ 0.10
np.round(normaltest(df.boxcox_medv), 3)

# %% Visualizing the Box-Cox Transformation

sns.kdeplot(data=df.boxcox_medv, color="red",
            fill="red", label="Box-Cox Transformed Median House Price")
sns.kdeplot(data=df.MEDV, color="green",
            fill="green", label="Median House Price")
plt.title("Kernel Density Estimation", fontweight="bold", size=15)
plt.xlabel("Median House Price ($1000s)", fontweight="bold", size=12,
           color="dimgrey")
plt.ylabel("Density", fontweight="bold", size=12,
           color="dimgrey")
plt.legend()
plt.show()

# %% Train-Test Split & Fitting the Box-Cox Linear Model

df = load_boston_df()

# Fitting a Box-Cox Linear Model
bclm = LinearRegression()

response = "MEDV"

X = df.drop(response, axis=1)
y = df[response]

pf = PolynomialFeatures(degree=2, include_bias=False)
X_pf = pf.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_pf, y, test_size=0.3, random_state=1024)

s = StandardScaler()
X_train_s = s.fit_transform(X_train)

# Instructor Solution

y_train_bc = boxcox(y_train)[0]
lambda_estimate = boxcox(y_train)[1]

bclm.fit(X_train_s, y_train_bc)
X_test_s = s.transform(X_test)
bclm_pred_boxcox = bclm.predict(X_test_s)  # Prediction in BoxCox Scale

# Inverse BoxCox Transformation
bclm_pred = inv_boxcox(bclm_pred_boxcox, lambda_estimate)  # Prediction in Traditional Scale

# Goodness of Fit Metrics:
bclm_r2score = r2_score(bclm_pred, y_test)
bclm_mse = mean_squared_error(y_true=y_test, y_pred=bclm_pred)

# %% Linear Regression Model

simplelm = LinearRegression()
simplelm.fit(X_train_s, y_train)
simplelm_pred = simplelm.predict(X_test_s)
simplelm_r2score = r2_score(simplelm_pred, y_test)
simplelm_mse = mean_squared_error(y_true=y_test, y_pred=simplelm_pred)

# %% Comparing Both Models
print(f"BoxCox Linear Model with Polynomial Features: \n"
      f" R-squared: {bclm_r2score.round(2)}, MSE: {bclm_mse.round(2)}\n")

print(f"Traditional Linear Model: \n "
      f"R-squared: {simplelm_r2score.round(2)}, MSE: {simplelm_mse.round(2)}")

# BoxCox Linear Model has a higher Coefficient of Determination
# & a lower Mean Squared Error
# In overall, the BoxCox Transformation improves the fit (in this model), we can check this
# changing the Random State
