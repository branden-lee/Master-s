#!/usr/bin/env python
# coding: utf-8

# In[111]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.svm import SVC, SVR
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler

### Part a
# Generate a matrix of indicator n-grams from a matrix of words using an inclusion threshold of 5.
def token_matrix(data, threshold, indicator):
    new_data = pd.DataFrame(index=range(len(data)))
    for n in range(len(data)):
        for word in data[n]:
            if word not in new_data.columns:
                new_data[word] = 0
            new_data[word].iloc[n] += 1
    col = new_data.columns[(new_data.sum(axis=0) >= threshold) == True]
    if indicator:
        return (new_data[col] > 0).astype(int)
    return new_data[col]

# Create the data to fit the model on. Removes punctuation, normalizes to lower case letters and applies a stemmer.
def process_text(threshold, indicator):
    data = pd.read_csv('reviews.csv', usecols=['review__evaluation', 'review__text'], keep_default_na=False)
    data = data[data['review__evaluation'] != ''] # Removes 3 rows without any review evaluation data
    data.index = range(0, len(data.index))
    
    documents = []
    word_counts = []
    translations = dict((ord(char), " ") for char in string.punctuation)
    
    stop_words = set(stopwords.words('spanish'))
    stemmer = SnowballStemmer('spanish')
    for i in data.index: 
        clean_text = data.loc[i, 'review__text'].translate(translations).lower()   
        word_list = clean_text.split()
        word_counts.append(len(word_list))  
        words = [stemmer.stem(w) for w in word_list if w not in stop_words]      
        documents.append(words)
        
    new_data = pd.concat([data['review__evaluation'].astype(int), pd.DataFrame({'word_count': word_counts}), token_matrix(documents, threshold, indicator)], axis=1)    
    return new_data

df = process_text(5, True)
X_train, X_test, y_train, y_test = train_test_split(df.loc[:,df.columns!='review__evaluation'], df.loc[:,'review__evaluation'], test_size=.2, shuffle=True, random_state=1, stratify=df.loc[:,'review__evaluation'])


# In[112]:


## Part b)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
clf = GridSearchCV(SVC(), param_grid={'kernel':['linear'], 'C':[.1, .2, .5, 1, 2, 5], 'max_iter':[200000]}, scoring='accuracy', cv=skf.split(X_train, y_train), refit=True)              
clf.fit(X_train, y_train)
acc_scoreb = accuracy_score(y_test, clf.predict(X_test))
mseb = mean_squared_error(y_test, clf.predict(X_test))

print(acc_scoreb)
print(mseb)


# In[116]:


## Part c)
clf = GridSearchCV(SVR(), param_grid={'kernel':['linear'], 'C':[.1, .2, .5, 1, 2], 'epsilon':[.5, 1, 2, 10], 'max_iter':[200000]}, scoring='neg_mean_squared_error', cv=skf.split(X_train, y_train), refit=True)              
clf.fit(X_train, y_train)
acc_scorec = accuracy_score(y_test, np.rint(clf.predict(X_test)))
msec = mean_squared_error(y_test, clf.predict(X_test))

print(acc_scorec)
print(msec)


# In[114]:


## Part d)
clf = GridSearchCV(SVC(), param_grid={'kernel':['linear'], 'C':[.1, .2, .5, 1, 2, 5], 'max_iter':[200000]}, scoring='neg_mean_squared_error', cv=skf.split(X_train, y_train), refit=True)              
clf.fit(X_train, y_train)
acc_scored = accuracy_score(y_test, np.rint(clf.predict(X_test)))
msed = mean_squared_error(y_test, clf.predict(X_test))

print(acc_scored)
print(msed)


# In[120]:


from tabulate import tabulate
print(tabulate({"Model/Criteria": ['SVC/Accuracy', 'SVR/MSE', 'SVC/MSE'], 
     "Accuracy": [acc_scoreb, acc_scorec, acc_scored],
                'MSE': [mseb, msec, msed]},
     headers="keys"), file=open('Q12.9 output.txt', 'w'))


# In[ ]:




