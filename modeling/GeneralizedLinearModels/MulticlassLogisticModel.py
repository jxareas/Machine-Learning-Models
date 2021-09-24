import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score

df = pd.read_csv("./data/HumanActivityRecognitionUsingSmartphones.csv")

# %% Examining some important properties of the dataset

# All columns are numeric, except for the multiclass categorical response variable
df.dtypes.value_counts()

# Obtaining the classes of the response
df.select_dtypes(np.object_).apply(np.unique)

# As said in the codebook for this dataset, the data is scaled from -1 (minimum) to 1 (maximum)
list(map(lambda x: np.round([x.max(), x.min()], 2), df.select_dtypes(np.number).values))

# Examining the balance between the classes of the response variable
pd.Series(df.Activity.value_counts(normalize=True).round(2))

# %% Defining the Response Variable and the predictors

# Encoding the Response 'Activity' as a Dummy Variable
df.Activity = LabelEncoder().fit_transform(df.Activity)

response = df[["Activity"]].columns

features = df.drop(labels="Activity", axis=1).columns

# %% Splitting the data between the Test & Training Set

stratified_split = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)

train_index, test_index = next(stratified_split.split(X=df[features], y=df[response]))

X_train = df.loc[train_index, features]
y_train = df.loc[train_index, response].values.ravel()

X_test = df.loc[test_index, features]
y_test = df.loc[test_index, response].values.ravel()

del train_index, test_index, stratified_split

#%% Fitting Several Logistic Regression Models

lr = LogisticRegression(solver='liblinear').fit(X_train, y_train)

lr_l1 = joblib.load(filename="./models/L1RegularizedLogistic.pkl")
#lr_l1 = LogisticRegressionCV(Cs=10, cv=4, penalty='l1', solver='liblinear').fit(X_train, y_train)

lr_l2 = joblib.load(filename="./models/L2RegularizedLogistic.pkl")
#lr_l2 = LogisticRegressionCV(Cs=10, cv=4, penalty='l2', solver='liblinear').fit(X_train, y_train)

#%% Comparing the Coefficients of the Fitted Models

coefficients = list()

coeff_labels = ['lr', 'l1', 'l2']
coeff_models = [lr, lr_l1, lr_l2]

for lab,mod in zip(coeff_labels, coeff_models):
    coeffs = mod.coef_
    coeff_label = pd.MultiIndex(levels=[[lab], [0,1,2,3,4,5]],
                                 codes=[[0,0,0,0,0,0], [0,1,2,3,4,5]])
    coefficients.append(pd.DataFrame(coeffs.T, columns=coeff_label))

coefficients = pd.concat(coefficients, axis=1)



#%% Creating six plots for each of the Multiclass Coefficients

sns.set_context('talk')
sns.set_style('white')

fig, axList = plt.subplots(nrows=3, ncols=2)
axList = axList.flatten()
fig.set_size_inches(10, 10)

for ax in enumerate(axList):
    loc = ax[0]
    ax = ax[1]

    data = coefficients.xs(loc, level=1, axis=1)
    data.plot(marker='o', ls='', ms=2.0, ax=ax, legend=False)

    if ax is axList[0]:
        ax.legend(loc=4)

    ax.set(title='Coefficient Set ' + str(loc))

plt.tight_layout()
plt.show()

#%% Predicting the category and their probability for each model

y_pred = list()
y_prob = list()

coeff_labels = ['lr', 'l1', 'l2']
coeff_models = [lr, lr_l1, lr_l2]

for lab, mod in zip(coeff_labels, coeff_models):
    y_pred.append(pd.Series(mod.predict(X_test), name=lab))
    y_prob.append(pd.Series(mod.predict_proba(X_test).max(axis=1), name=lab))

y_pred = pd.concat(y_pred, axis=1)
y_prob = pd.concat(y_prob, axis=1)

#%% Calculating some fundamental error metrics of each model

metrics = list()
cm = dict()

for lab in coeff_labels:
    # Preciision, recall, f-score from the multi-class support function
    precision, recall, fscore, _ = score(y_test, y_pred[lab], average='weighted')

    # The usual way to calculate accuracy
    accuracy = accuracy_score(y_test, y_pred[lab])

    # ROC-AUC scores can be calculated by binarizing the data
    auc = roc_auc_score(label_binarize(y_test, classes=[0, 1, 2, 3, 4, 5]),
                        label_binarize(y_pred[lab], classes=[0, 1, 2, 3, 4, 5]),
                        average='weighted')

    # Last, the confusion matrix
    cm[lab] = confusion_matrix(y_test, y_pred[lab])

    metrics.append(pd.Series({'precision': precision, 'recall': recall,
                              'fscore': fscore, 'accuracy': accuracy,
                              'auc': auc},
                             name=lab))

metrics = pd.concat(metrics, axis=1)

#%% Plotting the Confusion Matrix for Each Model

fig, axList = plt.subplots(nrows=2, ncols=2)
axList = axList.flatten()
cmap = sns.diverging_palette(10, 230, as_cmap=True)

fig.set_size_inches(12, 7)

axList[-1].axis('off')

for ax, lab in zip(axList[:-1], coeff_labels):
    sns.heatmap(cm[lab], ax=ax, annot=True, fmt='d', cmap=cmap)
    ax.set(title=lab)

plt.tight_layout()
plt.show()

