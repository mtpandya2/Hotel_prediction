# -*- coding: utf-8 -*-
"""
Created on Tue Jul 27 17:00:27 2021

@author: Nirzari Pandya
"""

import pandas as pd

df=pd.read_csv('c:/Users/Nirzari Pandya/Desktop/Nizu iphone/hotel_bookings.csv')

print(df.head())

df.shape

df1=df.isna().sum()
(df1)

def clean(df):
    df.fillna(0,inplace=True)

clean(df)


df1=df.isna().sum()
(df1)

df.columns

list=[ 'adults', 'children', 'babies']

for i in list:
    print('{} has unique value as {} '.format(i,df[i].unique()))  
    
filter=(df['adults']==0) & (df['children']==0) &(df['babies']==0)

df2=(df[filter])

pd.set_option('display.max_columns',32)
filter=(df['adults']==0) & (df['children']==0) &(df['babies']==0)

df2=(df[filter])

print(df2)

df3=df[~filter]
country_wie_data=df[df['is_canceled']==0]['country'].value_counts().reset_index()
country_wie_data
country_wie_data.columns=['country','No of guest']


#!pip install folium  

import folium

from folium.plugins import HeatMap

basemap=folium.map

#!pip install plotly

import plotly.express as px

map_guest=px.choropleth(country_wie_data,locations=country_wie_data['country'],
              color=country_wie_data['No of guest'],
              hover_name=country_wie_data['country'],
              title='home country of guestes'
              )
map_guest
import matplotlib.pyplot as plt

import seaborn as sns

data2=df[df['is_canceled']==0]

plt.figure(figsize=(12,8))
sns.boxplot(x='reserved_room_type',y='adr',hue='hotel',data=data2)
plt.title("price per room")
plt.xlabel("room type")

plt.ylabel('price(euro)')
plt.legend()
plt.show()


data_resrort=df[(df['hotel']=='Resort Hotel') &(df['is_canceled']==0)]

data_city=df[(df['hotel']=='City Hotel') &(df['is_canceled']==0)]

resort_hotel=data_resrort.groupby(['arrival_date_month'])['adr'].mean().reset_index()
resort_hotel

city_hotel=data_city.groupby(['arrival_date_month'])['adr'].mean().reset_index()
city_hotel
final=resort_hotel.merge(city_hotel,on='arrival_date_month')
final
final.columns=['month',"price for resort",'price for hotel']
final
#!pip install sort-dataframeby-monthorweek
#!pip install sorted-months-weekdays

import sort_dataframeby_monthorweek as sd

def sort_data(df,colname):
    return sd.Sort_Dataframeby_Month(df,colname)
final=sort_data(final,'month')
final

px.line(final,x='month',y=['price for resort','price for hotel'],
        title='room price per night over the months')

rush_resort=data_resrort['arrival_date_month'].value_counts().reset_index()
rush_resort.columns=['month','no of guestes']
rush_resort
rush_city=data_city['arrival_date_month'].value_counts().reset_index()
rush_city.columns=['month','no of guestes']
rush_city

final_rush=rush_resort.merge(rush_city,on='month')

final_rush.columns=['month','no of guestes in resort','no of guestes in city hotel']
final_rush=sort_data(final_rush,'month')

final_rush


cor=df.corr()['is_canceled']
cor

cor=cor.abs().sort_values(ascending=False)
cor

group=df.groupby('is_canceled')['reservation_status'].value_counts()
group
list_not=['days_in_waiting_list','arrival_date_year']

cols1=[]

for col in df.columns:
    if df[col].dtype!='O' and col  not in list_not:
        cols1.append(col)
(cols1)

(df.columns)

cat_not=['arrival_date_year','assigned_room_type','booking_changes','booking_changes',
         'country','days_in_waiting_list']


cols=[]

for col in df.columns:
    if df[col].dtype=='O' and col  not in cat_not:
        cols.append(col)
        
cols

data_cat=df[cols]

data_cat.head()

(data_cat.dtypes)

import warnings 
from warnings import filterwarnings
filterwarnings('ignore')



data_cat['reservation_status_date']=pd.to_datetime(data_cat['reservation_status_date'])
data_cat['reservation_status_date']
data_cat['year']=data_cat['reservation_status_date'].dt.year
data_cat['month']=data_cat['reservation_status_date'].dt.month
data_cat['day']=data_cat['reservation_status_date'].dt.day
data_cat['year']

data_cat.drop('reservation_status_date',axis=1,inplace=True)

data_cat['cancellation']=df['is_canceled'] 
(data_cat.head())


data_cat['market_segment'].unique()

cols=data_cat.columns[0:8]
cols


(data_cat.groupby(['hotel'])['cancellation'].mean())

for col in cols:
  (data_cat.groupby([col])['cancellation'].mean())
    
    
    
for col in cols:
    dict=data_cat.groupby([col])['cancellation'].mean().to_dict()
    data_cat[col]=data_cat[col].map(dict)
    
(data_cat.head())


data_frame=pd.concat([data_cat,df[cols1]],axis=1)
data_frame
data_frame.drop('cancellation',axis=1,inplace=True)
(data_frame.columns)



#outliers

sns.displot(data_frame['lead_time'])

import numpy as np
def hand_outlier(col):
    
    data_frame[col]=np.log1p(data_frame[col])

hand_outlier('lead_time')

sns.displot(data_frame['lead_time'])

sns.displot(data_frame['adr'])


hand_outlier('adr')


sns.displot(data_frame['adr']) 

print(data_frame.isna().sum())

data_frame.dropna(inplace=True)

print(data_frame.isna().sum())


#feature selection to prepare ml model

y=data_frame['is_canceled']
data_frame.head
x=data_frame.drop('is_canceled',axis=1)
x=x.drop('reservation_status',axis=1)
x=x.drop('customer_type',axis=1)




from sklearn.linear_model import Lasso

from sklearn.feature_selection import SelectFromModel

feature_sel_model=SelectFromModel(Lasso(alpha=0.005,random_state=0))
feature_sel_model.fit(x,y)
feature_sel_model.get_support()

cols2=x.columns

selectd_fea=cols2[feature_sel_model.get_support()]

print("total_fetures{}".format(x.shape[1]))

print("selected_fea{}".format(len(selectd_fea)))

selectd_fea
x=x[selectd_fea]


from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,
                                               random_state=0)

from sklearn.linear_model import LogisticRegression

log_reg=LogisticRegression()

log_reg.fit(x_train,y_train)

y_pred=log_reg.predict(x_test)
y_pred


from sklearn.metrics import confusion_matrix
z=confusion_matrix(y_pred,y_test)

from sklearn.metrics import accuracy_score
accuracy_score(y_pred,y_test)

from sklearn.model_selection import cross_val_score 

score=cross_val_score(log_reg,x,y,cv=10)   
score.mean()


from sklearn.naive_bayes import GaussianNB  

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier  
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


models=[]

models.append(('LogisticRegression',LogisticRegression()))

#models.append(('naive_bayes',GaussianNB))

models.append(('RandomForestClassifier',RandomForestClassifier))
models.append(('DecisionTreeClassifier',DecisionTreeClassifier()))
models.append(('KNeighborsClassifier  ',KNeighborsClassifier()))

for name,model in models:
    print(name)
    model.fit(x_train,y_train)
    pred=model.predict(x_test)
    from sklearn.metrics import confusion_matrix
    print(confusion_matrix(pred,y_test))
    print('\n')

    from sklearn.metrics import accuracy_score
    print(accuracy_score(pred,y_test))
    print('\n')

    
    






