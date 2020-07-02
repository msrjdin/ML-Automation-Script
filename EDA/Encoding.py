import copy
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


#Must take care when there is a combination of both categorical and ordinal
class Encoding:
    def __init__(self, df,colTypes,y, method):
        self.df = df.copy()
        self.y=y
        self.colTypes = copy.deepcopy(colTypes)
        # fetching column types for the columns in the current df
        self.colTypes['Categorical'] = set(df.columns).intersection(set(colTypes['Categorical']))
        # removing the target column for the list of categorical columns
        if self.y in self.colTypes['Categorical']:
            self.colTypes['Categorical'].remove(self.y)

        if method=='one-hot':
            self.one_hot_encoding(self.df)
        elif method=='label':
            self.label_encode(self.df)
        elif method=='one-hot-label':#Cannot be used ryt nw.. Have to include this functionality
            self.label_encode(self.df)
            self.one_hot_encoding(self.df)


    # encoding the categorical columns excluding the target column
    def one_hot_encoding(self, df):
        df1 = df
        df1 = pd.get_dummies(df1, drop_first=True, columns=list(self.colTypes['Categorical']))
        self.return_df = df1

    # encoding the categorical columns excluding the target column
    def label_encode(self, df):
        df1 = df
        for x in self.colTypes['Categorical']:
            df1[x] = LabelEncoder.fit_transform(df1, y=df1[x])
        df2=self.target_encode(df1)
        self.return_df = df2

    # encoding the categorical target column
    def target_encode(self, t_df):
        if self.y not in self.colTypes['Numeric']:
            t_df[self.y] = t_df[self.y].astype(str)
            t_df[self.y] = LabelEncoder.fit_transform(t_df, y=t_df[self.y])
        return t_df

    # returns the dict of dataframes updated in this class
    def return_result(self):
        return self.return_df
