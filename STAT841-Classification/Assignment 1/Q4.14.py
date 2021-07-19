import numpy as np
import pandas as pd
import statsmodels.api as sm

data = pd.read_csv(r"C:\Users\brand\OneDrive\Desktop\hunger_games.csv", header=0)
data.exog = sm.add_constant(data[['female', 'age', 'volunteer', 'career']])
data.endog = data['surv_day1']

### part a
m = sm.Logit(data.endog, data.exog)
m_results = m.fit()
print(m_results.summary(), file=open("part_a output.txt", 'w'))


### part b
print(data[['volunteer', 'career']], file=open("part_b output.txt", 'w'))


### part c
newdata = pd.DataFrame(np.repeat(data.values, 100, axis=0), columns=data.columns)

newdata.exog = sm.add_constant(newdata[['female', 'age', 'volunteer', 'career']]).astype(int)
newdata.endog = newdata['surv_day1'].astype(int)

m2 = sm.Logit(newdata.endog, newdata.exog)
m2_results = m2.fit()

print(m2_results.summary(), file=open("part_c output.txt", 'w'))


###part d

# Remove career variable then fit logistic regression model.
newdata.exog = sm.add_constant(newdata[['female', 'age', 'volunteer']]).astype(int)

m3 = sm.Logit(newdata.endog, newdata.exog)
m3_results = m3.fit()

print(m3_results.summary(), file=open("part_d1 output.txt", 'w'))

# Remove volunteer variable then fit logistic regression model.
newdata.exog = sm.add_constant(newdata[['female', 'age', 'career']]).astype(int)

m4 = sm.Logit(newdata.endog, newdata.exog)
m4_results = m4.fit()

print(m4_results.summary(), file=open("part_d2 output.txt", 'w'))