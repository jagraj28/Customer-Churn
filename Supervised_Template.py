"""
Template
"""

__date__ = "2023-06-08"
__author__ = "JingeHe"

# %% ----------------------------------------------------------------------------------------------------------
# Imports
# -------------------------------------------------------------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns

from Toolkit.Numerical_data import find_low_CV_columns, find_high_correlation_pairs, find_high_correlation_with_target

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, r2_score, classification_report, ConfusionMatrixDisplay, mean_squared_error, mean_absolute_error

from IPython.display import display

# %% ----------------------------------------------------------------------------------------------------------
# Random State
# -------------------------------------------------------------------------------------------------------------

rng = np.random.RandomState(123)

# %% ----------------------------------------------------------------------------------------------------------
# Macroeconomic data 
df = pd.read_excel(r"C:\Users\bario\OneDrive - Kubrick Group\PROJECT - Banking Churn\code\macro_data.xlsx")

df['date'] = pd.to_datetime(df['date'])

df['date'] = df['date'] + pd.offsets.MonthEnd(0)

df.to_csv("macro_data_date_adjusted.csv", index = False)

print(df)

df2 = pd.read_csv(r"C:\Users\bario\OneDrive - Kubrick Group\PROJECT - Banking Churn\banking_churn_data\Cleaned.csv")
df1 = pd.read_csv(r"C:\Users\bario\OneDrive - Kubrick Group\PROJECT - Banking Churn\code\macro_data_date_adjusted.csv")
df1['date'] = pd.to_datetime(df1['date'])
df2['date'] = pd.to_datetime(df2['date'])
print(df2)
df3 = pd.merge(df1, df2, on='date', how='left')
print(df3)
df3.to_csv("Cleaned_with_macro.csv", index = False)

# %% ----------------------------------------------------------------------------------------------------------
# Monthly aggregation
df = pd.read_csv(r"C:\Users\bario\OneDrive - Kubrick Group\PROJECT - Banking Churn\code\Cleaned_with_macro.csv")
#df.drop(columns=['deposit', 'withdrawal', "customer_id"], inplace=True)

df['date'] = pd.to_datetime(df['date'])

group_cols = ['account_id', df['date'].dt.to_period('M')]
df_grouped = df.groupby(group_cols)

agg_dict = {'amount': 'sum', 'interest_rates': 'first', 'consumer_sentiment': 'first',
            'unemployment_rate': 'first', 'dob': 'first',
            'state': 'first', 'start_balance': 'first', 'creation_date': 'first', 
            "Churn": "first"}
df_summed = df_grouped.agg(agg_dict).reset_index()

print(df_summed)

df_summed.to_csv("cleaned_aggregated.csv", index = False)
# %% ----------------------------------------------------------------------------------------------------------
# Balance Column
df = pd.read_csv(r"C:\Users\bario\OneDrive - Kubrick Group\PROJECT - Banking Churn\code\cleaned_aggregated.csv")

df['date'] = pd.to_datetime(df['date'])

df_sorted = df.sort_values(by=['account_id', 'date'])

first_row_index = df_sorted.groupby('account_id')['date'].idxmin()
df_sorted.loc[first_row_index, 'monthly_balance'] = df_sorted['start_balance'] + df_sorted['amount']

df_sorted['monthly_balance'] = df_sorted.groupby('account_id')['amount'].cumsum() + df_sorted['start_balance']

# Display the resulting dataframe
print(df_sorted)

df_sorted.to_csv("cleaned_aggregated_with_balance.csv", index = False)

# %% ----------------------------------------------------------------------------------------------------------
# difference column

df = pd.read_csv(r"C:\Users\bario\OneDrive - Kubrick Group\PROJECT - Banking Churn\code\cleaned_aggregated_with_balance.csv")

df['date'] = pd.to_datetime(df['date'])

# Sort the dataframe by 'account_id' and 'date'
df_sorted = df.sort_values(by=['account_id', 'date'])

# Take the earliest entry for each account_id and set "behaviour_change" to 0 for the first row
first_row_index = df_sorted.groupby('account_id')['date'].idxmin()
df_sorted.loc[first_row_index, 'behaviour_change'] = 0

# Calculate the difference between consecutive rows in the "amount" column within each "account_id" group
df_sorted['amount_difference'] = df_sorted.groupby('account_id')['amount'].diff()

# Set "behaviour_change" to 1 if "amount" in the current row is greater than the previous row, -1 if less, and 0 if equal
df_sorted['behaviour_change'] = df_sorted['amount_difference'].apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))

# Drop the temporary column "amount_difference"
df_sorted.drop(columns=['amount_difference'], inplace=True)

# Display the resulting dataframe
print(df_sorted)

df_sorted.to_csv("cleaned_aggregated_macro_balance_diff.csv", index = False)

# %% ----------------------------------------------------------------------------------------------------------
# consecutive withdrawals column

df = pd.read_csv(r"C:\Users\bario\OneDrive - Kubrick Group\PROJECT - Banking Churn\code\cleaned_aggregated_macro_balance_diff.csv")

df['date'] = pd.to_datetime(df['date'])

df = df.sort_values(by=['account_id', 'date'])

# Create the 'consecutive_withdrawals' column
consecutive_withdrawals = 0
consecutive_withdrawals_list = []
previous_account_id = None

for index, row in df.iterrows():
    if previous_account_id is None or row['account_id'] != previous_account_id:
        consecutive_withdrawals = 0

    if row['amount'] < 0:
        consecutive_withdrawals += 1
    elif row['amount'] == 0:
        pass
    else:
        consecutive_withdrawals = 0

    consecutive_withdrawals_list.append(consecutive_withdrawals)
    previous_account_id = row['account_id']

df['consecutive_withdrawals'] = consecutive_withdrawals_list

print(df)
#%%
df.to_csv("cleaned_aggregated_macro_balance_diff_consecutive.csv", index = False)
# %% ----------------------------------------------------------------------------------------------------------
# % integrity check

df = pd.read_csv(r"cleaned_aggregated_macro_balance_diff.csv")
unique_account_ids_count = df['account_id'].nunique()
# Display the count of unique account_ids
print("Count of unique account_ids:", unique_account_ids_count)

# %% ----------------------------------------------------------------------------------------------------------
# Data Import and Inspection (Easy)
# -------------------------------------------------------------------------------------------------------------
df = pd.read_csv(r"C:\Users\bario\OneDrive - Kubrick Group\PROJECT - Banking Churn\banking_churn_data\final_engineered_data2.csv")

column_name = "monthly_balance"

churn_1_data = df[df['Churn'] == 0]

plt.figure(figsize=(10, 6))  # Adjust the figure size if necessary

plt.hist(churn_1_data[column_name], bins=10, alpha=0.5, color='red')

plt.xlabel(f"{column_name}") 
plt.ylabel('Frequency')
plt.title('Histograms for Churn 0 and Churn 1')
plt.legend(loc='upper right')
plt.grid(True)
plt.show()


# %% ----------------------------------------------------------------------------------------------------------
# Data Import and Inspection (Hard)
# -------------------------------------------------------------------------------------------------------------

target_column = ""

# Features with high correlation (excluding target)

print(find_high_correlation_pairs(data, target_column, 0.8), "\n")

# Features with low coefficient of variation

print(find_low_CV_columns(data, target_column, 10), "\n")

# Features with high target correlation

print(find_high_correlation_with_target(data, target_column, 0.8))


# %% ----------------------------------------------------------------------------------------------------------
# Test/Train Split
# -------------------------------------------------------------------------------------------------------------

X = data.drop(columns=[target_column])
y = data[target_column]

print(X)
# Use stratify argument here if data comes from classes of differing sizes

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=rng
)

preprocessor = ColumnTransformer(
    transformers=[
        ('ohe', OneHotEncoder(drop='first'), ["sex", "region", "smoker"])  
        # (more transformers here)
    ]
)
Xt_train = preprocessor.fit_transform(X_train)
Xt_test = preprocessor.transform(X_test)

print(Xt_train.shape)
# %% ----------------------------------------------------------------------------------------------------------
# Pipeline
# -------------------------------------------------------------------------------------------------------------

# Preprocess data
preprocessor = ColumnTransformer(
    transformers=[
        ('ohe', OneHotEncoder(drop='first'), ["columns here"])  
        # (more transformers here)
    ]
)

# Hyperparameter search grid
param_grid = {
    'n_neighbors': [3, 5, 7],  # Specify the values to be searched 
    'weights': ['uniform', 'distance'],  
}

model = GridSearchCV(KNeighborsClassifier(), param_grid = param_grid, cv = 5)

pipeline = Pipeline(
    [
        ('Preprocessor', preprocessor),
        ('Model', model) # Randomised search using "RandomizedSearch" and "param_distributions =" instead
    ]
)

display(pipeline)


# %% ----------------------------------------------------------------------------------------------------------
# Fit and Predict
# -------------------------------------------------------------------------------------------------------------

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

print(accuracy_score(y_test, y_pred)) # Categorisation

print(np.sqrt(mean_squared_error(y_test, y_pred))) # Regression

# %% ----------------------------------------------------------------------------------------------------------
# Model Evaluation
# -------------------------------------------------------------------------------------------------------------

# Categorisation 
fix, ax = plt.subplots()
cm_plot = ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred, ax = ax, cmap = "OrRd", display_labels = "insert labels here", xticks_rotation = 45,
    normalise = "true"
)

print(classification_report(y_test, y_pred))

# %% ----------------------------------------------------------------------------------------------------------
# Visualisation
# -------------------------------------------------------------------------------------------------------------

# Regressor coeffiecients
feature_names = X.columns.tolist()
coefficients = model.coef_
intercept = model.intercept_

#coefficients_dict = {feature_names[i]: coefficients[i] for i in range(len(feature_names))}
coefficients_dict = {feature_names[i]: coefficients[0, i] for i in range(len(feature_names))}

print("Intercept:", intercept)
print("Coefficients:")
for feature, coefficient in coefficients_dict.items():
    print(feature, ":", coefficient)

# Scatter plot
plt.scatter(X, y, label="")
plt.xlabel("")
plt.ylabel("")

# Plot the fitted curve
plt.plot(X_test, y_pred, color='red', label='Fitted Curve')

plt.title(f'')
plt.legend()
plt.show()
