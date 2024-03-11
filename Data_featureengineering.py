"""
Enter script name

Enter short description of the script
"""

__date__ = "2023-07-17"
__author__ = "AbdullahYousaf"



# %% --------------------------------------------------------------------------
# Import pre processed dataset
# -----------------------------------------------------------------------------
import pandas as pd
df = pd.read_csv(r'C:\Users\AbdullahYousaf\MLE07\Final Project\banking-churn\final_preprocessed.csv')              

# %%
# Create Net Transaction for past 3 months feature
df['date'] = pd.to_datetime(df['date'])  # ensure the date column is in datetime format

# sort by account_id and date
df = df.sort_values(['account_id', 'date'])

# calculate the rolling sum for the past 3 months for each account_id
df['Net Transaction for past 3 months'] = df.groupby('account_id')['amount'].rolling(window=3, min_periods=1).sum().reset_index(0, drop=True)

# create the binary column
df['Net Transaction for past 3 months (Binary)'] = df['Net Transaction for past 3 months'].apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))

# for the first two months for each account_id, set the values of both columns to 0
df.loc[df.groupby('account_id').cumcount() < 2, ['Net Transaction for past 3 months', 'Net Transaction for past 3 months (Binary)']] = 0

# %%
# Create account age feature

# calculate the maximum date for each account_id
max_date = df.groupby('account_id')['date'].transform('max')

# calculate the minimum date for each account_id
min_date = df.groupby('account_id')['date'].transform('min')

# calculate the account age in months
df['account_age'] = (max_date - min_date).dt.days // 30

# %%
# Create customer age feature

# Ensure the dob column is in datetime format
df['dob'] = pd.to_datetime(df['dob'])

# Create the specific date
specific_date = pd.to_datetime('31-05-2020')

# Create 'customer_age' column which is the number of years from dob to the specific date
df['customer_age'] = (specific_date - df['dob']).dt.days // 365

df = df.drop(columns=['dob'])


# %%
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

# %%
df = df.rename(columns={'Net Transaction for past 3 months (Binary)': 'Net Deposit/Withdrawal over 3 months'})

df.drop(columns=['creation_date', 'Net Transaction for past 3 months', ], inplace=True)
# %%
churn = df['Churn']
df = df.drop(columns=['Churn'])
df['Churn'] = churn

# %%
# Save file to csv
df.to_csv('final_engineered_data2.csv', index=False)
# %%
