# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 11:11:59 2019

@author: Shivam
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

spam = pd.read_csv('spam.csv',encoding = 'latin-1')

import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, 5572):
    review = re.sub('[^a-zA-Z]', ' ',spam['v2'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
    
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
y = spam.iloc[:,0:1].values
    
from sklearn.preprocessing import LabelEncoder
labelencoder_y = LabelEncoder()
y[:,0]=labelencoder_y.fit_transform(y[:,0])
y = y.astype(int)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

'''
#Using naive bayes algo
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)
'''
'''
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)
0.9741564967695621 accuracy
'''
'''
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)
0.9210337401292176 accuracy
'''
# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

