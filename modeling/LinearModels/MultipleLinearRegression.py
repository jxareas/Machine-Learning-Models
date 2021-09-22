import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.stats.anova as anova

df = pd.read_csv("./data/KentuckyDerby.csv")

# %% Creating new variables from the dataset

# Creating indicator variables fast & good
df = df.assign(fast=lambda data: data.condition.map(lambda x: 1 if x == "fast" else 0)).assign(
    good=lambda data: data.condition.map(lambda x: 1 if x == "good" else 0))

# Creating a year centered variable at 1896
df = df.assign(yearnew=lambda data: data.year.map(lambda x: x - 1896))

# Recoding the fast indicator variable
df["fast_factor"] = np.where(df.fast == 1, "fast", "not fast")

#%% Fitting a Linear Model with Interaction-Effects

lm = smf.ols(formula="speed ~ yearnew + fast + yearnew:fast", data=df).fit()

lm_summ = lm.summary()

print(lm_summ)

#%% Refitting the linear model with new predictors: good & starters

lm2 = smf.ols(formula="speed ~ yearnew + fast + yearnew:fast + good + starters", data=df).fit()

lm2_summ = lm.summary()

print(lm2_summ)

#%% Nested-Model Testing

# Low p-value: we reject the Null-Hypothesis that the first linear
# model provides a significantly better fit, and we accept the
# second model (lm2)
anova_results = anova.anova_lm(lm, lm2)
print(anova_results)