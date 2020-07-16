
import copy
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


# df = pd.read_csv(r"C:\Users\SatyaSindhuMolleti\Downloads\sample_train.csv")
# df=df[:10]
# print(df)
# col_map={'label':['Sex'] ,'one-hot':['Embarked','PassengerId']}

# Must take care when there is a combination of both categorical and ordinal
class Encoding:
    def __init__(self, df,col_map):
        self.df = df.copy()
        self.col_map=col_map
        self.col_methods=[]
        self.df_final = df.copy(deep=True)
        # self.targetName=targetName
        # self.colTypes=colTypes
        #
        # for key, value in colTypes.items():
        #     if key == self.targetName:
        #        self.targetType = value

        #if condition for target


        for key,value in col_map.items():
            self.col_methods.extend(key)
            self.df_final.drop(columns=value, inplace=True, axis=1)
            if key == 'one-hot':
                print(value)
                self.df_final=pd.concat([self.df_final,self.one_hot_encoding(df[value])],axis=1)
            elif key == 'label':
                self.df_final=pd.concat([self.df_final,self.label_encode(df[value])],axis=1)


    # encoding the categorical columns excluding the target column
    def one_hot_encoding(self, df):
        df1 = df.copy()
        for col in df.columns:
            df_dummies = pd.get_dummies(df1[col], drop_first=True)
            df_dummies.rename(columns=lambda x: col+'_'+str(x), inplace=True)
        return df_dummies


    # encoding the categorical columns excluding the target column
    def label_encode(self, df):
        df1 = df.copy()
        for x in df.columns:
            df1[x] = LabelEncoder.fit_transform(df1, y=df1[x])
        return df1


    # encoding the categorical target column
    # def target_encode(self, t_df):
    #     t_df[self.target_name] = t_df[self.targetName].astype(str)
    #     t_df[self.targetName] = LabelEncoder.fit_transform(t_df, y=t_df[self.targetName])
    #     return t_df


    # returns the dict of dataframes updated in this class
    def return_result(self):
        # print(self.df_final.columns)
        return self.df_final


# oh=Encoding(df,col_map)
# oh.return_result()
