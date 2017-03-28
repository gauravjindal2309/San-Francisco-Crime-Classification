# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 16:48:51 2017

@author: gaurav
"""
%matplotlib inline
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import sklearn
import statsmodels.api as sm
from matplotlib import rcParams

index = pd.date_range('1/1/2000', periods=8)
pd.Series(np.random.randn(5), index=['a', 'b', 'c', 'd', 'e'])
df = pd.DataFrame(np.random.randn(8, 3), index=index,
   ...:                   columns=['A', 'B', 'C'])

df.columns = [x.lower() for x in df.columns]
df.head()
df[:2]
df.values

from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.linear_model import SGDRegressor
import numpy as np
from sklearn import metrics as ms

msk = np.random.rand(len(beers)) < 0.8

train = beers[msk]
test = beers[~msk]

X = train [['Price', 'Net price', 'Purchase price','Hour','Product_id','product_group2']]
y = train[['Quantity']]
y = y.as_matrix().ravel()

X_test = test [['Price', 'Net price', 'Purchase price','Hour','Product_id','product_group2']]
y_test = test[['Quantity']]
y_test = y_test.as_matrix().ravel()

clf = SGDRegressor(n_iter=2000)
clf.fit(X, y)
predictions = clf.predict(X_test)
print "Accuracy:", ms.accuracy_score(y_test,predictions)