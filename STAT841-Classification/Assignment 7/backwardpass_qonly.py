#!/usr/bin/env python
# coding: utf-8

# In[14]:


import numpy as np

xd = (1 + np.exp(-0.8 * 3 - 1.1 * (-2) - 0.1 * 5)) ** (-1)

xe = (1 + np.exp(-0.3*3 - 0.5 * (-2) - 0.2 * 5)) ** (-1)

yf = (1 + np.exp(-1.3 * xd - 0.4 * xe)) ** (-1)
yf
xd * yf * (1-yf)


# In[15]:


w7_new = 1.3-0.5*(-(1-yf)**2 * yf * xd)
w7_new


# In[ ]:




