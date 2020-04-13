import copy
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


class Encoding:
    def __init__(self, df,colTypes,y):
        self.y=y
        self.colTypes = copy.deepcopy(colTypes)
        # fetching column types for the columns in the current df
        self.colTypes['Categorical'] = set(df.columns).intersection(set(colTypes['Categorical']))
        # removing the target column for the list of categorical columns
        if self.y in self.colTypes['Categorical']:
            self.colTypes['Categorical'].remove(self.y)

        self.df = df.copy()
        self.all_dfs = []
        self.one_hot_encoding()
        self.label_encode()

    # encoding the categorical columns excluding the target column
    def one_hot_encoding(self):
        df1 = self.df.copy(deep=True)
        df_y = pd.DataFrame()
        df1 = pd.get_dummies(df1, drop_first=True, columns=list(self.colTypes['Categorical']))
        self.all_dfs.append(self.target_encode(df1))

    # encoding the categorical columns excluding the target column
    def label_encode(self):
        df1 = self.df.copy(deep=True)
        df_y = pd.DataFrame()
        for x in self.colTypes['Categorical']:
            df1[x] = LabelEncoder.fit_transform(df1, y=df1[x])
        self.all_dfs.append(self.target_encode(df1))

    # encoding the categorical target column
    def target_encode(self, t_df):
        if self.y not in self.colTypes['Numeric']:
            t_df[self.y] = LabelEncoder.fit_transform(t_df, y=t_df[self.y])
        return t_df

    # returns the dict of dataframes updated in this class
    def return_dfs(self):
        return self.all_dfs
