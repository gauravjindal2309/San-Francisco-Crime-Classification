# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 16:08:20 2017

@author: gaurav
"""

# San francisco crime problem

import csv
%matplotlib inline
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import sklearn
import statsmodels.api as sm
from matplotlib import rcParams
from sklearn import preprocessing, cross_validation, svm
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss, accuracy_score

# for loading csv file
# train = csv.reader('train.csv')
# test = csv.reader('test.csv')

# for loading csv file into dataframe
# test = pd.read_csv(r'C:\Users\gaurav\Documents\python\Analytics vidhya\San Francisco Crime Classification\test.csv')
# train = pd.read_csv(r'C:\Users\gaurav\Documents\python\Analytics vidhya\San Francisco Crime Classification\train.csv')
# train.describe()
# train.dtypes



"""
Data fields

Dates - timestamp of the crime incident
Category - category of the crime incident (only in train.csv). This is the target variable you are going to predict.
Descript - detailed description of the crime incident (only in train.csv)
DayOfWeek - the day of the week
PdDistrict - name of the Police Department District
Resolution - how the crime incident was resolved (only in train.csv)
Address - the approximate street address of the crime incident
X - Longitude
Y - Latitude
"""


#z1 = zipfile.ZipFile('train.zip')
#z2 = zipfile.ZipFile('test.zip')
train = pd.read_csv('train.csv', parse_dates = ['Dates'])
test = pd.read_csv('test.csv', parse_dates = ['Dates'])

#Convert crime category (labels) to numbers
le_crime = preprocessing.LabelEncoder()
crime = le_crime.fit_transform(train.Category)

days = pd.get_dummies(train.DayOfWeek)
district = pd.get_dummies(train.PdDistrict)
year = train.Dates.dt.year
hour = train.Dates.dt.hour
day = train.Dates.dt.day
x = train.X
y = train.Y

train_data = pd.concat([year, day, hour, days, district, x, y], axis=1)
train_data['crime'] = crime

crime_data = train_data.iloc[:,:-1]
crime_label = train_data['crime']

days = pd.get_dummies(test.DayOfWeek)
district = pd.get_dummies(test.PdDistrict)
year = test.Dates.dt.year
hour = test.Dates.dt.hour
day = test.Dates.dt.day
x = test.X
y = test.Y
test_data = pd.concat([year, day, hour, days, district, x, y], axis=1)

[crime_train_data, crime_test_data, crime_train_labels, crime_test_labels] = cross_validation.train_test_split(crime_data, crime_label, test_size=0.3)

#Logistic Regression
lgr = LogisticRegression(C=1e5)
lgr.fit(crime_train_data, crime_train_labels.values.ravel())
prediction = lgr.predict_proba(crime_test_data)
log_loss(crime_test_labels, prediction)
print lgr.score(crime_test_data, crime_test_labels)
print accuracy_score(crime_test_data, crime_test_labels)
# 2.58

# Naive Bayes
crime_nb = BernoulliNB()
crime_nb.fit(crime_train_data, crime_train_labels)
prediction = np.array(crime_nb.predict_proba(crime_test_data))
log_loss(crime_test_labels, prediction) 
print accuracy_score(crime_test_labels, prediction.argmax(axis=1))

# SVM
crime_svm = svm.SVC(kernel='linear')
crime_svm.fit(crime_train_data, crime_train_labels)
predicted = np.array(crime_svm.predict_proba(crime_test_data))
log_loss(crime_test_labels, predicted) 
print crime_svm.score(crime_test_data, crime_test_labels)

# Random Forest
crime_rf = RandomForestClassifier()
crime_rf.fit(crime_train_data, crime_train_labels)
prediction = np.array(crime_rf.predict_proba(crime_test_data))
print accuracy_score(crime_test_labels, crime_rf.predict_proba(crime_test_data))
print log_loss(crime_test_labels, prediction) 

result=pd.DataFrame(prediction, columns=le_crime.classes_)
result.to_csv('testResult.csv', index = True, index_label = 'Id' )



import pandas as pd
import time
import csv
import numpy as np
import os

from sklearn import preprocessing, cross_validation
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import log_loss
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sknn.mlp import Classifier, Layer
from sklearn.decomposition import PCA

clear = lambda: os.system('cls')
clear()

train = pd.read_csv('train.csv', parse_dates = ['Dates'])

#train = train.head(n=10)
train['AddressContainOf'] = 0


start = time.time()
lenth = len(train)
cur = 0
for index in range(len(train)):
    if(index > 5000 and index/5000.0 > cur):
        print ("pre-processing the", cur*5000, "th data")
        cur += 1
    if(train.iloc[index,6].find('/') == -1):
        train.iloc[index,9] = 1

pca = PCA(n_components=2)
coor = []
for index in range(len(train)):
    coor.append([train['X'].iloc[index], train['Y'].iloc[index]])
pca.fit(coor)
pca_transform = pca.transform(coor)
cur = 0
for index in range(len(train)):
    if(index > 5000 and index/5000.0 > cur):
        print ("pre-processing the", cur*5000, "th data")
        cur += 1
    train.iloc[index,7] = pca_transform[index][0]
    train.iloc[index,8] = pca_transform[index][1]
    
train.to_csv('newtrain.csv')


#improved 

import pandas as pd
import time
import csv
import numpy as np
import os

from sklearn import preprocessing, cross_validation
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import log_loss
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sknn.mlp import Classifier, Layer
from sklearn.decomposition import PCA

clear = lambda: os.system('cls')
clear()

train = pd.read_csv('newtrain.csv', parse_dates = ['Dates'])
train = train.head(n=100000)

print train.head()


train['StreetNo'] = train['Address'].apply(lambda x: x.split(' ', 1)[0] if x.split(' ', 1)[0].isdigit() else 0)
#train['Address'] = train['Address'].apply(lambda x: x.split(' ', 1)[1] if x.split(' ', 1)[0].isdigit() else x)
train['hour'] = train['Dates'].dt.hour
train['evening'] = train['Dates'].dt.hour.isin([18,19,20,21,22,23,0,1,2,3,4,5,6])
train['Year'] = train['Dates'].dt.year
#train = train[train['Year'].isin([2011,2012,2013,2014,2015])]
train['Month'] = train['Dates'].dt.month




start = time.time()
lenth = len(train)
cur = 0

print '  -> processing time:', time.time() - start
#print train.head()
print len(set(train['StreetNo'])), len(set(train['Address']))

le = LabelEncoder()
crime = le.fit_transform(train.Category)

hour = pd.get_dummies(train.hour)
district = pd.get_dummies(train.PdDistrict)
StreetNo = pd.get_dummies(train.StreetNo)
evening = pd.get_dummies(train.evening)
ContainOf = pd.get_dummies(train.AddressContainOf)
Year = pd.get_dummies(train.Year)
Month = pd.get_dummies(train.Month)

train_data = pd.concat([hour, district, StreetNo, evening, ContainOf, train['X'], train['Y']], axis=1)
train_data['crime'] = crime
crime_data = train_data.iloc[:,:-1]
crime_label = train_data['crime']

classifiers = [
    BernoulliNB(),
    RandomForestClassifier(max_depth=10, n_estimators=1024, n_jobs=-1),
    RandomForestClassifier(max_depth=12, n_estimators=1024, n_jobs=-1),
    RandomForestClassifier(max_depth=14, n_estimators=1024, n_jobs=-1),
    RandomForestClassifier(max_depth=16, n_estimators=1024, n_jobs=-1),
    RandomForestClassifier(max_depth=18, n_estimators=1024, n_jobs=-1),
    RandomForestClassifier(max_depth=20, n_estimators=1024, n_jobs=-1),
    RandomForestClassifier(max_depth=22, n_estimators=1024, n_jobs=-1),
    KNeighborsClassifier(n_neighbors=100, weights='distance', algorithm='ball_tree', leaf_size=100, p=10, metric='minkowski'),
    #XGBClassifier(max_depth=16,n_estimators=1024),
    GradientBoostingClassifier(n_estimators=10, learning_rate=1.0,max_depth=5, random_state=0),
    AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=8), algorithm="SAMME.R", n_estimators=128),
    ]
    
#print train.head()
    
newClassifiers = [
    BernoulliNB(),
    RandomForestClassifier(max_depth=16, n_estimators=1024, n_jobs=-1),
    GradientBoostingClassifier(max_depth=16, n_estimators=1024)
    #KNeighborsClassifier(n_neighbors=50, weights='distance', algorithm='ball_tree', leaf_size=100, p=10, metric='minkowski', n_jobs=-1),
    ]

 
#[train_d, test_d, train_labels, test_labels] = cross_validation.train_test_split(crime_data, crime_label, test_size=0.2, random_state=20160217)
skf = cross_validation.StratifiedKFold(crime_label, n_folds=2, random_state=20160217, shuffle=True)
for train_index, test_index in skf:
    train_d, test_d = crime_data.iloc[train_index,:], crime_data.iloc[test_index,:]
    train_labels, test_labels = crime_label[train_index], crime_label[test_index]
    print train_d.shape, test_d.shape
    for classifier in classifiers:
        print classifier.__class__.__name__
        start = time.time()
        classifier.fit(train_d, train_labels)
        print '  -> Training time:', time.time() - start
                        
        start = time.time()        
        #score_result = classifier.score(test_d, test_labels)
        #print '  -> caluclate score time', time.time() - start
                        
        start = time.time()
        predicted = np.array(classifier.predict_proba(test_d))
        print '  -> predict_proba time:', time.time() - start
                        
        start = time.time()
        log_result = log_loss(test_labels, predicted)
        print '  -> calculate log_loss time:', time.time() - start        
                        
        #print "score = ", score_result, "log loss = ",log_result
        print "log_loss = ", log_result