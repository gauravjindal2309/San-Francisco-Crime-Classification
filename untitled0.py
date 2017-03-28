# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 01:19:31 2017

@author: gaurav
"""

from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from time import time

import pandas as pd
import scipy
import zipfile
from matplotlib.backends.backend_pdf import PdfPages

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
cats = list(set(train.Category))
mapdata = np.loadtxt(drive+"sf_map_copyright_openstreetmap_contributors.txt")

dates[]
dateAll = np.array([datetime.strptime(date, '%Y-%m-%d %H:%M:%S') for date in train.Dates])
startDate = (np.min(dateAll)).date()
endDate = (np.max(dateAll)).date()
alldates = pd.bdate_range(startDate,endDate, freq ='m')
dayDF = pd.DataFrame(np.NAN,index= alldates, columns = ['x'])
 subCats = ['KIDNAPPING','PROSTITUTION','VEHICLE THEFT','LOITERING','SUICIDE','FORGERY/COUNTERFEITING','DRUNKENNESS','DRUG/NARCOTIC','LARCENY/THEFT']
 
 
 #pdf_pages = PdfPages('crimeData.pdf')
    pLoop = 1
    for cat in cats:
        saveFile = cat+'.png'
        print(saveFile)
        #just subset for display purposes
        if cat in subCats:
            try:
                fig = plt.figure(figsize = (11.69, 8.27))
                plt.title(cat)
                
                #plot image
                ax = plt.subplot(2,2,1)
                ax.imshow(mapdata, cmap=plt.get_cmap('gray'), 
                      extent=lon_lat_box)
    
                lineNum = 0
                crime  = cat
                Xcoord = (train[train.Category==cat].X).values
                Ycoord = (train[train.Category==cat].Y).values
                dates = datesAll[np.where(train.Category==cat)]
                Z = np.ones([len(Xcoord),1])
                
   
                
                
                # next kernel
                
import pandas as pd
import zipfile
import matplotlib.pyplot as pl
%matplotlib inline

train = pd.read_csv('train.csv', parse_dates=['Dates'])

train['Year'] = train['Dates'].map(lambda x: x.year)
train['Week'] = train['Dates'].map(lambda x: x.week)
train['Hour'] = train['Dates'].map(lambda x: x.hour)

print(train.head())

train.PdDistrict.value_counts().plot(kind='bar', figsize=(8,10))
pl.savefig('district_counts.png')



train.Year.value_counts().plot(kind='bar', figsize=(8,10))
train.Week.value_counts().plot(kind='bar', figsize=(8,10))
train.Hour.value_counts().plot(kind='bar', figsize=(8,10))

train['event'] = 1
weekly_events = train[['Week','Year','event']].groupby(['Year','Week']).count().reset_index()
weekly_events_years = weekly_events.pivot(index='Week', columns='Year', values='event').fillna(method='ffill')

ax = weekly_events_years.interpolate().plot(title='number of cases every 2 weeks', figsize=(10,6))
pl.savefig('events_every_two_weeks.png')


hourly_events = train[['Hour','event']].groupby(['Hour']).count()
hourly_events_hour = hourly_events.plot(kind ='bar', figsize =(6,6))hourly_events = train[['Hour','event']].groupby(['Hour']).count().reset_index()
hourly_events.plot(kind='bar', figsize=(6, 6))
pl.savefig('hourly_events.png')


hourly_district_events = train[['PdDistrict','Hour','event']].groupby(['PdDistrict','Hour']).count().reset_index()
hourly_district_events_pivot = hourly_district_events.pivot(index='Hour', columns='PdDistrict', values='event').fillna(method='ffill')
hourly_district_events_pivot.interpolate().plot(title='number of cases hourly by district', figsize=(10,6))
pl.savefig('hourly_events_by_district.png')
















