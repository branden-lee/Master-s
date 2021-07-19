#!/usr/bin/env python
# coding: utf-8

# In[10]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.DataFrame({'x1': [0,3,2,5,4,3,5,6], 'x2':[4,1,3,4,4,5,6,4], 'y':['y' for n in range(4)]+['g' for n in range(4)], 'new':[0,0,0,1,0,0,0,0]})

sns.scatterplot(data=df, x='x1', y='x2', hue='y', style='new', palette={'y':'y', 'g':'g'}, legend=False)

plt.plot([0,6.5], [6.5, 0], color='grey', linestyle='dashed')
plt.plot([0,6.5], [5, 3], color='red', linestyle='dashed')
plt.savefig('Q12.2.png')


# In[ ]:




