"""
Enter script name

Enter short description of the script
"""

__date__ = "2023-07-17"
__author__ = "James Morrison"


# %% --------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %% --------------------------------------------------------------------------
# Set Random State
# -----------------------------------------------------------------------------
rng = np.random.RandomState(123)

# %% --------------------------------------------------------------------------
# reead in transactions dataframe and define churn
# -----------------------------------------------------------------------------
df_trans = pd.read_csv(r'C:\Users\JamesMorrison\Documents\banking_churn_data\banking_churn_data\transactions_tm1_e.csv')
#df_trans.tail(10)
#df_trans.info()
# %% --------------------------------------------------------------------------
# read in customer datarame
# -----------------------------------------------------------------------------
df_cust = pd.read_csv(r'C:\Users\JamesMorrison\Documents\banking_churn_data\banking_churn_data\customers_tm1_e.csv')

# %% --------------------------------------------------------------------------
# merge the two dataframes and drop transaction date
# -----------------------------------------------------------------------------
df = df_trans.merge(df_cust)
df.drop(columns=['transaction_date'])

# %% --------------------------------------------------------------------------
# preprocess - state column
# -----------------------------------------------------------------------------
df1 = df[df['state'] != 'Australia']
df2 = df1[df1['state'] != 'UNK']
df3 = df2[df2['state'] != '-999']
state_map = {'NY' : 'New York', 'MASS' : 'Massachusetts', 'TX' : 'Texas', 'CALIFORNIA' : 'California'}
df3['state'] = df3['state'].map(state_map).fillna(df['state'])

# %% --------------------------------------------------------------------------
# preprocess - remove negative starting balances and billionaires and transactions
# worth billions
# -----------------------------------------------------------------------------
df4 = df3[abs(df3['amount']) < 1234567890] # remove anomolous transactions
df5 = df4[df4['start_balance']>= 0] # remove negative starting balances
df6 = df5[df5['start_balance'] < 12345678] # remove billionaire start balance


# %% --------------------------------------------------------------------------
# 
# -----------------------------------------------------------------------------
# Ensure the date column is in datetime format
df6['date'] = pd.to_datetime(df6['date'])
data = df6.copy()

# %% --------------------------------------------------------------------------
# correct Churn column
# -----------------------------------------------------------------------------
#data = pd.read_csv(r'C:\Users\JamesMorrison\Documents\banking_churn_data\Cleaned_aggregated_with_balance.csv')
# Ensure the date column is in datetime format
data['date'] = pd.to_datetime(data['date'])

# Find the max date (final date) for each account id
data['max_date'] = data.groupby('account_id')['date'].transform('max')
data['Churn'] = 0
# iterate through the table and check for each row if the
# date is equal to the max_date, if it is change churn to 1
for index, row in data.iterrows():
    
    if row['date'] == row['max_date']:
        data.at[index, 'Churn'] = 1
    else:
        pass

df_churn = data.drop(columns=['max_date'])

# %% --------------------------------------------------------------------------
# save the file
# -----------------------------------------------------------------------------
df_churn.to_csv('Cleaned_with_churn.csv', index=False)

# %%
