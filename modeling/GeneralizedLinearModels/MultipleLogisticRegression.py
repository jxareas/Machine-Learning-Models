import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("./data/StockMarket.csv")

df["DirectionDummy"] = df.Direction.map(lambda x: int(x == "Up"))

# %%

corr_matrix = df.select_dtypes(np.number).corr()

mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

# %% Plotting the absolute correlations as a Histogram

sns.set_theme(style="white")
sns.set_context('notebook')
# Generate a custom diverging colormap
cmap = sns.diverging_palette(10, 230, as_cmap=True)

ax = sns.heatmap(corr_matrix, mask=mask, cmap=cmap,
                 vmax=1, vmin=-1, center=0,
                 square=False, linewidths=.5,
                 cbar_kws={"shrink": .7,
                           "label": "Pearson's r"})
ax.set_yticklabels(ax.get_yticklabels(), rotation=40, ha="right")
plt.title("Correlation Heatmap", fontweight="bold", fontsize=20)
plt.show()

# Year and Volume have the highest correlation coefficient ~ 0.53.

del cmap, ax, mask
# %% Fitting the Logistic Model

response = "DirectionDummy"
predictors = df.columns.drop(labels=["Year", "Today", "Direction", response]).tolist()

formula = response + "~" + "+".join(predictors)

logitm = smf.logit(formula=formula, data=df).fit()

print(logitm.summary())

del response, predictors, formula
#%% Analyzing the Regression Results

# High p-values, coefficients are not statistically significant at
# a significance level of 0.05
logitm.pvalues.round(decimals=2)

# Probabilities of the market going up
prediction = pd.DataFrame(logitm.predict(), columns=["PredictedDirection"])
prediction = prediction["PredictedDirection"].map(lambda x: "Up" if (x > 0.5) else "Down")


#%% Error Metrics

confusion_matrix = pd.DataFrame(logitm.pred_table(threshold = 0.5),
                                columns=["PredictedDown", "PredictedUp"],
                                index=["Down", "Up"])

print(confusion_matrix)


# This model is fitted with all the data as part of training set, hence
# the training error is:
print(100 * (1 - np.mean(prediction == df.Direction)))


