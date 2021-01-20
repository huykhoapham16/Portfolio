# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 09:23:33 2021

@author: Khoa
"""

import numpy as np
import pandas as pd

df = pd.read_csv(r"C:\Users\Khoa\Desktop\New coding\Project HR Analytics Job change of Data Scientists\aug_train.csv")

#Cleaning up DataFrame values
df['relevent_experience'] = df['relevent_experience'].str.replace('Has relevent experience','Y')
df['relevent_experience'] = df['relevent_experience'].str.replace('No relevent experience','N')

df['enrolled_university'] = df['enrolled_university'].str.replace('Full time course','FT')
df['enrolled_university'] = df['enrolled_university'].str.replace('no_enrollment','N')
df['enrolled_university'] = df['enrolled_university'].str.replace('Part time course','PT')



#Imputing missing data with most frequent values for string and mean for numeric 
from sklearn.base import TransformerMixin
class DataFrameImputer(TransformerMixin):

    def __init__(self):
        """Impute missing values.

        Columns of dtype object are imputed with the most frequent value 
        in column.

        Columns of other types are imputed with mean of column.

        """
    def fit(self, X, y=None):

        self.fill = pd.Series([X[c].value_counts().index[0]
            if X[c].dtype == np.dtype('O') else X[c].mean() for c in X],
            index=X.columns)

        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill)

df = DataFrameImputer().fit_transform(df)


#Data transformation by Labelencoder
from sklearn.preprocessing import LabelEncoder 

le = LabelEncoder()
df['city'] = le.fit_transform(df['city'])
df['gender'] = le.fit_transform(df['gender'])
df['relevent_experience'] = le.fit_transform(df['relevent_experience'])
df['enrolled_university'] = le.fit_transform(df['enrolled_university'])
df['education_level'] = le.fit_transform(df['education_level'])
df['major_discipline'] = le.fit_transform(df['major_discipline'])
df['experience'] = le.fit_transform(df['experience'])
df['company_size'] = le.fit_transform(df['company_size'])
df['company_type'] = le.fit_transform(df['company_type'])
df['last_new_job'] = le.fit_transform(df['last_new_job'])

#or use OneHotEncoder although in this case, there seems to be some issues to be fixed tomorrow
from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder() 
df = np.array(columnTransformer.fit_transform(df), dtype = np.str) 



#start deploying keras model 


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

#using Keras Model
inputs = keras.Input(shape=(13,))
dense = layers.Dense(64,activation='relu')
x = dense(inputs)
x = layers.Dense(64, activation='relu')(x)
outputs = layers.Dense(10)(x)
model = keras.Model(inputs=inputs,outputs=outputs,name='hr_data_science_train_model')
model.summary()



#defining x,y values and its associated train,test values
x = df.iloc[:,0:13]
y = df.iloc[:,13:14]
x = np.matrix(x)
y = np.matrix(y)
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.4,random_state=42)


#compiling model 
model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.RMSprop(),
    metrics=["accuracy"],)

#Fitting data into model
history = model.fit(x_train,y_train,batch_size=64,epochs=2,validation_split=0.2)

#accuracy measurement 
test_scores = model.evaluate(x_test,y_test,verbose=2)















