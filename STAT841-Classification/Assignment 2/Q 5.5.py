#!/usr/bin/env python
# coding: utf-8

# In[17]:


import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
from sklearn.linear_model import LassoCV, ElasticNetCV, LinearRegression
from tabulate import tabulate

# For reproducability
np.random.seed(0) 

# Generate 200 observations of 50 independent normallly distributied random variables
d = multivariate_normal.rvs(mean=[0 for i in range(50)], size=200)

# Create dataframe containing design matrix
X = pd.DataFrame(data=d, columns=['x%d' % i for i in range(1, 51)])

# Noise term for response
eps = multivariate_normal.rvs(mean=0, cov=10, size=200)

# True response
y = X.apply(lambda x: np.dot(x[['x1','x2','x3']], [4, 3, -6]), axis=1) + eps

# Sample without replacement of the row indices of X to create our training/test split
training_indices = np.random.choice(200, size=100, replace=False)

# Split up training and test data
X_training = X.iloc[training_indices,:]
y_training = y.iloc[training_indices]

X_test = X.iloc[~training_indices,:]
y_test = y.iloc[~training_indices]

# Function to calculate MSE given observed and predicted values
def mean_squared_error(observed, predictions):
    diff = observed - predictions
    rss = np.dot(diff, diff) / len(observed)
    return(rss)

## Fitting a linear regression model on all 50 covariates, predicting on training data and calculating MSE
model_a = LinearRegression(fit_intercept=False).fit(X_training, y_training)
y_a = model_a.predict(X_test)
mse_a = mean_squared_error(y_test, y_a)

## Fitting a lasso model with 10-fold cross validation, predicting on training data and calculating MSE
model_b = LassoCV(fit_intercept=False, cv=10).fit(X_training, y_training)
y_b = model_b.predict(X_test)
mse_b = mean_squared_error(y_test, y_b)

## Fitting an elastic net model with 10-fold cross validation, predicting on training data and calculating MSE
model_c = ElasticNetCV(l1_ratio=.9, fit_intercept=False, cv=10).fit(X_training, y_training)
y_c = model_c.predict(X_test)
mse_c = mean_squared_error(y_test, y_c)

## Fitting a linear regression model on only first three covariates, predicting on training data and calculating MSE
model_d = LinearRegression(fit_intercept=False).fit(X_training[['x1','x2','x3']], y_training)
y_d = model_d.predict(X_test[['x1','x2','x3']])
mse_d = mean_squared_error(y_test, y_d)

print(tabulate({"Model": ["a) Linear Regression on all variables", "b) Lasso with 10-fold CV", "c) Elastic Net with 10-fold CV",
                         "d) Linear Regression on important variables"],
                "MSE": [mse_a, mse_b, mse_c, mse_d]}, headers="keys"))#, file=open('Q 5.5 output.txt', 'w'))


# In[ ]:




