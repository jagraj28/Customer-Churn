# Customer-Churn

You work as a data scientist at a US bank. The bank offers a special savings account intended for middle income professionals. Interest is accrued in the savings account on a monthly basis and is only accrued if there is a net positive change after deposits and withdrawals in that month. A small fee is charged with customers withdraw from the account. Customers are therefore incentivized to continually deposit into the account to take advantage of the interest.

The bank uses these deposits to fund activities and investments. It is in the bank’s interest to keep the savings portfolio healthy as a larger savings portfolio gives the bank a larger liquidity buffer. It has been identified that customer churn is one of the main sources of outflow of funds from the bank.

Your manager wants to identify customers who are likely to churn and see whether they can be incentivized to keep their accounts open. Your task is to develop a model to predict the likelihood of customer churn and convince others that your model and analysis are trustworthy.

Date Range
Start date	: 2007-01-31
End date	: 2020-05-31

You want to identify customers who close their accounts in the following month.

Before fitting our model, you will need to make sure our data is fit for purpose. There may also be some data pre-processing required before the data is in a form suitable as input for a supervised learning model.
External data you can consider include macroeconomic data from the Federal Reserve. The data can be obtained here: https://fred.stlouisfed.org/. You can also consider US population distributions to see whether this lines up with the data.

The task here is to build a model to predict customer churn in the following month. Think about what features are predictive of churn and experiment to determine whether they help with your predictions.
Use your model to make predictions of whether active customers in the current month are likely to churn. Your predictions will be used to validate your model.

You are required to be able to explain and interpret your model. For model interpretation, you can consider techniques such as partial dependence plots and SHAP.
