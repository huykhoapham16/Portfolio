# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 09:23:33 2021

@author: Khoa
"""

import numpy as np
import pandas as pd


df = pd.read_csv(r"C:\Users\Khoa\Desktop\New coding\Project HR Analytics Job change of Data Scientists\aug_test.csv")

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


dft = pd.read_csv(r"C:\Users\Khoa\Desktop\New coding\Project HR Analytics Job change of Data Scientists\aug_train.csv")

















