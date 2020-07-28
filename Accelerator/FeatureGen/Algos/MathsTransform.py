import pandas as pd
import numpy as np

coltype = {'PassengerId': 'Numeric', 'Survived': 'Categorical', 'Sex': 'Categorical', 'Pclass': 'Categorical',
           'Name': 'Text', 'Age': 'Numeric', 'SibSp': 'Categorical', 'Parch': 'Categorical', 'Fare': 'Numeric'}


class MathsTransform:

    def __init__(self, df, coltype):

        self.df = df
        self.coltype = coltype
        self.feature_generation()

    def feature_generation(self):

        numeric_columns = []
        for key, value in coltype.items():
            if value == 'Numeric':
                numeric_columns.append(key)
        numeric_sqrt = ['sqrt_' + s for s in numeric_columns]
        numeric_cr = ['cr_' + s for s in numeric_columns]
        numeric_log = ['log_' + s for s in numeric_columns]
        numeric_reciprocal = ['rec_' + s for s in numeric_columns]
        self.df[numeric_sqrt] = self.df[numeric_columns].apply(lambda x: np.sqrt(x))
        self.df[numeric_cr] = self.df[numeric_columns].apply(lambda x: np.cbrt(x))
        self.df[numeric_log] = self.df[numeric_columns].apply(lambda x: np.log(x))
        self.df[numeric_log].fillna(0, inplace=True)
        self.df[numeric_reciprocal] = self.df[numeric_columns].apply(lambda x: np.reciprocal(x))
        self.df[numeric_reciprocal].fillna(0, inplace=True)

    def return_result(self):
        return self.df



