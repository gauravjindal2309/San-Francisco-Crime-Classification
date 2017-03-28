# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 19:06:55 2017

@author: gaurav
"""

## all imports
from IPython.display import HTML
import numpy as np
import urllib2
import bs4 #this is beautiful soup
import time
import operator
import socket
import cPickle
import re # regular expressions

from pandas import Series
import pandas as pd
from pandas import Dataframe

import matplotlib
import matplotlib.pyplot as plt
%matplotlib inline

import seaborn as sns
sns.set_context("talk")
sns.set_style("white")

from secret import *

u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']

users = pd.read_csv(
    'http://files.grouplens.org/datasets/movielens/ml-100k/u.user', 
    sep='|', names=u_cols)

users.head()


r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
ratings = pd.read_csv(
    'http://files.grouplens.org/datasets/movielens/ml-100k/u.data', 
    sep='\t', names=r_cols)

ratings.head() 

m_cols = ['movie_id', 'title', 'release_date', 
            'video_release_date', 'imdb_url']

movies = pd.read_csv(
    'http://files.grouplens.org/datasets/movielens/ml-100k/u.item', 
    sep='|', names=m_cols, usecols=range(5))

movies.head()

print movies.dtypes
print movies.describe()

user.head()
users['occupation'].head()
columns_you_want = ['occupation','sex']
print users[columns_you_want].head()

users[(users.age == 40) & (users.sex == 'M')].head()
users[(users.sex == 'F') & (users.occupation == 'programmer')].describe()
grouped_data = ratings['movie_id'].groupby(ratings['user_id'])

ratings_per_user = grouped_data.count()
Avg_rating = ratings['rating'].groupby(ratings['movie_id'])
maximum_rating = average_ratings.max()
maximum_rating