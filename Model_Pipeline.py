"""
Mushroom data in pipeline


"""

__date__ = "2023-06-08"
__author__ = "James Morrison"



# %% --------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder, OneHotEncoder
from sklearn.datasets import fetch_openml
from sklearn.utils import shuffle, resample
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.image as mpimg
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import FunctionTransformer, Pipeline
import time
# %% --------------------------------------------------------------------------
# Set Random State
# -----------------------------------------------------------------------------
rng = np.random.RandomState(123)

# # %% ------------------------------------------------------------------------
# Read in the file
# -----------------------------------------------------------------------------
data = pd.read_csv(r'C:\Users\JamesMorrison\Documents\banking_churn_data\banking-churn\final_engineered_data2.csv')
may_2020_data = data[data['date'] == '2020-05-01']
df = data[data['date'] != '2020-05-01']
before_2018_data = df[df['date'] < '2018-01-01']
after_2018_data = df[(df['date'] >= '2018-01-01') & (df['date'] < '2020-05-01')]

covid_data = df[(df['date'] >= '2019-12-01') & (df['date'] < '2020-05-01')]

print(len(before_2018_data))
print(len(after_2018_data))

# Separate churn 0 and churn 1 data for both parts
before_2018_data_churn0 = before_2018_data[before_2018_data['Churn'] == 0]
before_2018_data_churn1 = before_2018_data[before_2018_data['Churn'] == 1]
after_2018_data_churn0 = after_2018_data[after_2018_data['Churn'] == 0]
after_2018_data_churn1 = after_2018_data[after_2018_data['Churn'] == 1]

covid_data0 = covid_data[covid_data['Churn'] == 0]
covid_data1 = covid_data[covid_data['Churn'] == 1]

# Calculate the desired sample size for resampling
desired_sample_size_before = int(len(before_2018_data_churn0))
desired_sample_size_after = int(len(after_2018_data_churn0))

desired_sample_size_covid = int(len(covid_data0))

# Resample churn 1 data for both parts separately
before_2018_resampled_churn1 = resample(before_2018_data_churn1, n_samples=desired_sample_size_before, random_state=rng)
after_2018_resampled_churn1 = resample(after_2018_data_churn1, n_samples=desired_sample_size_after, random_state=rng)

covid_data_resampled_churn1 = resample(covid_data1, n_samples=desired_sample_size_covid, random_state=rng)

# Concatenate the resampled churn 1 data with churn 0 data for both parts
before_2018_balanced_data = pd.concat([before_2018_data_churn0, before_2018_resampled_churn1])
after_2018_balanced_data = pd.concat([after_2018_data_churn0, after_2018_resampled_churn1])

covid_balanced_data = pd.concat([covid_data0, covid_data_resampled_churn1])

# Sample the combined data to get a 10% portion
combined_sample_size_before = int(0.1 * len(before_2018_balanced_data))
combined_sample_size_after = int(0.1 * len(after_2018_balanced_data))

covid_sample_size = int(0.1 * len(covid_balanced_data))

final_before = before_2018_balanced_data.sample(n=combined_sample_size_before, random_state=rng)
final_after = after_2018_balanced_data.sample(n=combined_sample_size_after, random_state=rng)

final_covid = covid_balanced_data.sample(n=covid_sample_size, random_state=rng)

print(len(final_before))
print(len(final_after))

# %% --------------------------------------------------------------------------
# Train/test split
# -----------------------------------------------------------------------------
#X = df.drop(columns=['Churn', 'account_id', 'date', "state", "customer_age"])
#y = df['Churn']

X_train = final_covid.drop(columns=['Churn', 'account_id', 'date'])
X_test = final_before.drop(columns=['Churn', 'account_id', 'date'])
y_train = final_covid['Churn']
y_test = final_before["Churn"]

#split the data into train and test
#X_train, X_test, y_train, y_test = train_test_split(
    #X, y, test_size=0.3, stratify=y, random_state=rng
#)

# %% --------------------------------------------------------------------------
# Preprocess data
# -----------------------------------------------------------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ('log_transform', FunctionTransformer(np.log1p), ['start_balance', 'monthly_balance', "account_age", "customer_age"]),
        ('min_max_scaler', MinMaxScaler(), ['interest_rates', 'consumer_sentiment', 'unemployment_rate']),  
        ('scaler', StandardScaler(), ['amount']),
        ('onehot', OneHotEncoder(drop='first'), ['state'])  
    ],
    remainder='passthrough'  # For columns that don't need any transformation
)


# %% --------------------------------------------------------------------------
# Create the Pipelines for Decision tree and Logistic Regression
# -----------------------------------------------------------------------------
gbparams = {
    'learning_rate': [0.1, 0.05],
    'n_estimators': [200, 300],
    'max_depth': [3, 5, 7],
    'random_state': [rng]
}
treepipeline = Pipeline(
    [
        ('transformer', preprocessor),
        ('gb', GridSearchCV(GradientBoostingClassifier(), gbparams, cv=5))
    ]
)

#--------
lrparams = {
    'penalty':['l2'],#,'l2'],
    'max_iter':[200],#,250],
    'random_state':[rng]
}

rf_params = {
    'n_estimators': [150],         # Number of trees in the forest
    'criterion': ['gini'],       # Criterion used to measure the quality of a split
    'max_depth': [12],            # Maximum depth of the tree (None means no maximum depth)
    'min_samples_split': [5],        # Minimum number of samples required to split an internal node
    'min_samples_leaf': [6],          # Minimum number of samples required to be at a leaf node      # Number of features to consider when looking for the best split
    'bootstrap': [True],             # Whether to use bootstrap samples when building trees
    'random_state': [123]                   # Random state for reproducibility
}

lrpipeline = Pipeline(
    [
        ('transformer', preprocessor),
        ('log', GridSearchCV(LogisticRegression(), lrparams, cv=5))
    ]
)
#--------
dtparams = {
    'min_samples_leaf': [10, 3, 5],
    'criterion': ['gini', 'log_loss'],
    'max_depth': [10, 5, 7],
    'random_state': [rng]
}
dtpipeline = Pipeline([
        ('transformer', preprocessor),
        ('feature_selection', SelectFromModel(RandomForestClassifier(random_state=rng), threshold='mean')),
        ('log', GridSearchCV(RandomForestClassifier(), rf_params, cv=5))
    ])
# %% --------------------------------------------------------------------------
# fit the models
# -----------------------------------------------------------------------------
# Start the clock
start_time = time.time()

dtpipeline.fit(X_train, y_train)
#treepipeline.fit(X_train, y_train)

# Stop the clock
end_time = time.time()

# Calculate the elapsed time
elapsed_time = end_time - start_time

# Print the elapsed time
print(f"Elapsed time: {elapsed_time} seconds")

# %% --------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------
from sklearn.metrics import recall_score, confusion_matrix

y_pred_dt = dtpipeline.predict(data[data['date'] == '2020-04-01'].drop(columns=['Churn', 'account_id', 'date']))
#y_pred_dt = dtpipeline.predict(before_2018_data.drop(columns=['Churn', 'account_id', 'date']))
#y_pred_tree = treepipeline.predict(X_test)
#y_pred_lr = lrpipeline.predict(X_test)

# Calculate accuracy for each model
accuracy_dt = accuracy_score(data[data['date'] == '2020-04-01']['Churn'], y_pred_dt)
#accuracy_dt = accuracy_score(before_2018_data["Churn"], y_pred_dt)
#accuracy_tree = accuracy_score(y_test, y_pred_tree)
#accuracy_lr = accuracy_score(y_test, y_pred_lr)

# Print the accuracy
print(f"Decision Tree Accuracy: {accuracy_dt}")
tree_recall = recall_score(data[data['date'] == '2020-04-01']['Churn'], y_pred_dt)
#tree_recall = recall_score(before_2018_data["Churn"], y_pred_dt)
print("Gradient Boosted Tree Recall:", tree_recall)
#print("Logistic Regression Recall:", lr_recall)

# %% --------------------------------------------------------------------------
# ROCAUC
# -----------------------------------------------------------------------------
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# Get predicted probabilities for the positive class (churned) from the models
y_prob_dt = dtpipeline.predict_proba(data[data['date'] == '2020-04-01'].drop(columns=['Churn', 'account_id', 'date']))[:, 1]
#y_prob_dt = dtpipeline.predict_proba(before_2018_data.drop(columns=['Churn', 'account_id', 'date']))[:, 1]
#y_prob_tree = treepipeline.predict_proba(X_test)[:, 1]
#y_prob_lr = lrpipeline.predict_proba(X_test)[:, 1]

# Calculate the ROC curve and AUC for each model
fpr_dt, tpr_dt, thresholds_dt = roc_curve(data[data['date'] == '2020-04-01']['Churn'], y_prob_dt)
#fpr_dt, tpr_dt, thresholds_dt = roc_curve(before_2018_data["Churn"], y_prob_dt)
#fpr_tree, tpr_tree, thresholds_tree = roc_curve(y_test, y_prob_tree)
#fpr_lr, tpr_lr, thresholds_lr = roc_curve(y_test, y_prob_lr)

roc_auc_dt = roc_auc_score(data[data['date'] == '2020-04-01']['Churn'], y_prob_dt)
#roc_auc_dt = roc_auc_score(before_2018_data["Churn"], y_prob_dt)
#roc_auc_tree = roc_auc_score(y_test, y_prob_tree)
#roc_auc_lr = roc_auc_score(y_test, y_prob_lr)

# Plot the ROC curve for Decision Tree model
plt.figure(figsize=(8, 6))
plt.plot(fpr_dt, tpr_dt, label=f'Decision Tree (AUC = {roc_auc_dt:.2f})')
#plt.plot(fpr_tree, tpr_tree, label=f'Tree Pipeline (AUC = {roc_auc_tree:.2f})')
#plt.plot(fpr_lr, tpr_lr, label=f'Logistic Regression (AUC = {roc_auc_lr:.2f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.show()

# %%
predictions = dtpipeline.predict_proba(may_2020_data.drop(columns=['Churn', 'account_id', 'date']))[:, 1]
predictions_df = pd.DataFrame(predictions)
predictions_df.to_csv("predictions.csv", index = False)

# %%
filename = r"C:\Users\JamesMorrison\Documents\banking_churn_data\submission_sample.csv"
sub_df = pd.read_csv(filename)

may_2020_data.reset_index(inplace=True, drop = True)
sub_df.head()
may_prob = pd.concat([may_2020_data, predictions_df], axis = 1)
may_prob.head()

# %%

may_prob.rename(columns = {0:'predict_churn'}, inplace = True)

may_prob.head()

#%%

new_submission = pd.merge(sub_df[['account_id', 'date']], may_prob[['account_id', 'predict_churn']], on = 'account_id', how = 'left')
new_submission.to_csv("submission.csv")

# %%
