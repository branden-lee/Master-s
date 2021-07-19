#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import *
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from tabulate import tabulate
import statsmodels.api as sm

np.random.seed(0)

training_indices = np.random.choice(1000, size=800, replace=False)

# Read and process text file, by removing punctuation, normalizing to lower case letters and applying a porter stemmer.
# Return a dataframe with 3 columns: one for the Document stored as a list of words, another for the word count and the last
# for the positive/negative sentiment.
def process_text(stem, stop):
    temp_data =[]
    
    translations = dict((ord(char), " ") for char in string.punctuation)
        
    f=open('amazon_cells_labelled.txt')
    for line in f:
        document = line.strip().split('\t')   
        tokens = word_tokenize(document[0].translate(translations).lower())
        word_count = len(tokens)
        if stop is True:
            stop_words = set(stopwords.words('english')) 
            if stem is True:
                stemmer = PorterStemmer()
                tokens = [stemmer.stem(w) for w in tokens if w not in stop_words]
            else:
                tokens = [w for w in tokens if w not in stop_words]
        else:
            if stem is True:
                stemmer = PorterStemmer()
                tokens = [stemmer.stem(w) for w in tokens]
            else:
                tokens = [w for w in tokens]
        temp_data.append({'Document': tokens, 'word_count': word_count, 'y': int(document[1])})
        
    return(pd.DataFrame(temp_data, columns=['Document', 'word_count', 'y']))

# Create a matrix consisting of non-negative integers from a dataframe data which is meant to be created from process_text.
# A 1 in entry in row i and column j corresponds to document i having the jth word.
def token_matrix(data):
    new_data = pd.DataFrame(index=data.index)
    for n in range(len(data.index)):
        for word in data.iloc[n]['Document']:
            if word not in new_data.columns:
                new_data[word] = 0
            new_data[word].iloc[n] += 1
    return new_data

# Generate a matrix of n_grams from a matrix created from token_matrix with specified threshold.
def generate_n_grams(token_matrix, threshold):
    col = token_matrix.columns[(token_matrix.sum(axis=0) >= threshold) == True]
    return token_matrix[col]

# Fit a random forest classifier on the data and calculate prediction accuracy on test set.
def accuracy(training, test):
    model = RandomForestClassifier().fit(X=training[training.columns.difference(['Document', 'y'])], y=training['y'])
    return accuracy_score(test['y'], model.predict(X=test[test.columns.difference(['Document', 'y'])]))
    
### part a
df = process_text(True, True)
df2 = token_matrix(df)

n_grams = generate_n_grams(df2, 5)
n_grams_ind = n_grams.ge(0.5).astype(int)
df_a = pd.concat([df, n_grams], axis=1)
df_ind = pd.concat([df, n_grams_ind], axis=1)

accuracy_count = accuracy(df_a.iloc[training_indices], df_a.iloc[~training_indices])
accuracy_ind = accuracy(df_ind.iloc[training_indices], df_ind.iloc[~training_indices])

print(tabulate(
    {"N-gram Type": ['Count', 'Indicator'], 
     "Accuracy": [accuracy_count, accuracy_ind]},
     headers="keys"), file=open('Q 6.4 a output.txt', 'w'))

### part b
n_grams_3 = generate_n_grams(df2, 3)
df_3 = pd.concat([df, n_grams_3], axis=1)

n_grams_7 = generate_n_grams(df2, 7)
df_7 = pd.concat([df, n_grams_7], axis=1)

n_grams_9 = generate_n_grams(df2, 9)
df_9 = pd.concat([df, n_grams_9], axis=1)

print(tabulate({"Threshold": [3, 5, 7, 9], 
     "Accuracy":
     [accuracy(df_3.iloc[training_indices], df_3.iloc[~training_indices]),
      accuracy_count,
      accuracy(df_7.iloc[training_indices], df_7.iloc[~training_indices]),
      accuracy(df_9.iloc[training_indices], df_9.iloc[~training_indices])
     ]},
     headers="keys"), file=open('Q 6.4 b output.txt', 'w'))

### part c
stem_no_stop = process_text(True, False)
n_grams_stem_no_stop = generate_n_grams(token_matrix(stem_no_stop), 5)
df_stem_no_stop = pd.concat([stem_no_stop, n_grams_stem_no_stop], axis=1)

no_stem_stop = process_text(False, True)
n_grams_no_stem_stop = generate_n_grams(token_matrix(no_stem_stop), 5)
df_no_stem_stop = pd.concat([no_stem_stop, n_grams_no_stem_stop], axis=1)

no_stem_no_stop = process_text(False, False)
n_grams_no_stem_no_stop = generate_n_grams(token_matrix(no_stem_no_stop), 5)
df_no_stem_no_stop = pd.concat([no_stem_no_stop, n_grams_no_stem_no_stop], axis=1)

print(tabulate(
    {"With Stemming": 
     [accuracy_count, 
      accuracy(df_stem_no_stop.iloc[training_indices], df_stem_no_stop.iloc[~training_indices])], 
     "Without Stemming":
     [accuracy(df_no_stem_stop.iloc[training_indices], df_no_stem_stop.iloc[~training_indices]), 
      accuracy(df_no_stem_no_stop.iloc[training_indices], df_no_stem_no_stop.iloc[~training_indices])]},
     headers="keys", showindex=['Remove Stopwords', 'Keep Stopwords']), file=open('Q 6.4 c1 output.txt', 'w'))

print(tabulate(
    {"With Stemming": [len(n_grams.columns), len(n_grams_stem_no_stop.columns)], 
     "Without Stemming": [len(n_grams_no_stem_stop.columns), len(n_grams_no_stem_no_stop.columns)]},
     headers="keys", showindex=['Remove Stopwords', 'Keep Stopwords']), file=open('Q 6.4 c2 output.txt', 'w'))

### part d
# Try to fit logistic regression to the data, and if it fails print output describing the type of error that occurred.
try:
    logistic_model = sm.Logit(endog=df_a['y'].iloc[training_indices], 
                      exog=df_a[df_a.columns.difference(['Document', 'y'])].iloc[training_indices])
    logistic_results = logistic_model.fit()
except Exception as e:
    print(e, file=open('Q 6.4 d output.txt', 'w'))


# In[ ]:




