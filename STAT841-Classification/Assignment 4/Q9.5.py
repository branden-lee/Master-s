#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
import statsmodels.api as sm

df = pd.DataFrame({'x': [0, 1, 2, 3], 'y': [1, 2, 2, 1]})

model = sm.OLS(endog=df['y'], exog=sm.add_constant(df['x'])).fit()
print(model.summary(), file=open('Q9.5 Output.txt', 'w'))


# In[ ]:




