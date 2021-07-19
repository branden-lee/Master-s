#!/usr/bin/env python
# coding: utf-8

# In[6]:


from tabulate import tabulate
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import mean_squared_error

np.random.seed(0)

X = pd.DataFrame({'x%d' % num: np.random.normal(size=1000) for num in range(1, 51)})
y = -2 + 4 * X['x1'] - 3 * X['x2'] + 2 * X['x3'] + np.random.normal(size=1000)

kf = KFold(shuffle=True, random_state=1)

# Part a
lr = GridSearchCV(LinearRegression(), param_grid={}, cv=kf, scoring='neg_mean_squared_error')
lr.fit(X, y)
lr_mspe = -lr.best_score_ # The mean cross validated score (mean_squared_error) of the best model


# In[7]:


# Part b
svr_lk = GridSearchCV(SVR(), param_grid={'kernel': ['linear']}, cv=kf, scoring='neg_mean_squared_error')
svr_lk.fit(X, y)
svr_lk_mspe = -svr_lk.best_score_ # The mean cross validated score (mean_squared_error) of the best model


# In[8]:


# Part c
rbf_untuned = GridSearchCV(SVR(), param_grid={'kernel': ['rbf']}, cv=kf, scoring='neg_mean_squared_error')
rbf_untuned.fit(X, y)
rbf_untuned_mspe = -rbf_untuned.best_score_ # The mean cross validated score (mean_squared_error) of the best model


# In[9]:


# Part d
rbf_tuned = GridSearchCV(SVR(), param_grid={'kernel': ['rbf'], 'gamma':[0.1, .5, 1, 2, 5], 'C': [0.1, .5, 1, 2, 5], 'epsilon': [0.1, .5, 1, 2, 5]}, cv=kf, scoring='neg_mean_squared_error')
rbf_tuned.fit(X, y)
rbf_tuned_mse = -rbf_tuned.best_score_ # The mean cross validated score (mean_squared_error) of the best model


# In[10]:


print(tabulate(
    {'Model': 
     ['a) Linear Regression', 'b) SVR with linear kernel', 'c) SVR with RBF kernel, untuned', 'd) SVR with RBF kernel, tuned'], 
     'MSE':
     [lr_mspe, svr_lk_mspe, rbf_untuned_mspe, rbf_tuned_mse]},
     headers="keys"), file=open('Q 12.10 output.txt', 'w'))


# In[ ]:




