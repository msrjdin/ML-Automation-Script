import pandas as pd
import numpy as np

# coltype = {'PassengerId': 'Numeric', 'Survived': 'Categorical', 'Sex': 'Categorical', 'Pclass': 'Categorical',
#            'Name': 'Text', 'Age': 'Numeric', 'SibSp': 'Categorical', 'Parch': 'Categorical', 'Fare': 'Numeric'}
# df=pd.read_csv('C://Users//SindhuKarnati//Desktop//MLAccelarator_v2//Accelerator//dataframe.csv')
# df=df.head()
# col_map = {'log':['Fare'],'reciprocal':['Age']}

class MathsTransform:

    def __init__(self, df,col_map):

        self.df = df
        self.col_map = col_map
        self.df_final = pd.DataFrame()
        for key,value in self.col_map.items():
            if key == 'sqrt':
                self.sqrt_(self.df[value])
            if key == 'cbrt':
                self.cbrt_(self.df[value])
            if key == 'log':
                self.log_(self.df[value])
            if key == 'reciprocal':
                self.reciprocal_(self.df[value])


    def sqrt_(self,df):
        print('hi sqrt')
        df_ = df.copy(deep=True)
        for col in df.columns:
            self.df_final['sqrt_'+col] = np.sqrt(df[col])

        return self.df_final


    def log_(self,df):
        df1 = df.copy(deep=True)
        for col in df1.columns:
            df1['log_'+col] = np.log(df1[col])
            df1.fillna(0, inplace=True)
            self.df_final['log_'+col]=df1['log_'+col].copy()

        return self.df_final


    def cbrt_(self,df):
        df_ = df.copy(deep=True)
        for col in df.columns:
            self.df_final['cbrt_'+col] = np.cbrt(df[col])

        return self.df_final


    def reciprocal_(self,df):
        df1= df.copy(deep=True)
        for col in df1.columns:
            df1['reciprocal_'+col] = np.reciprocal(df1[col])
            df1.fillna(0, inplace=True)
            self.df_final['reciprocal_'+col]=df1['reciprocal_'+col].copy()

        return self.df_final


#dictionary of list:
#log_: [col1,col2,col3], sqrt: [col2,col3]

    # def return_result(self):
    #     print(self.df_final)
    #     return self.df_final


#
# ct=MathsTransform(df,col_map)
# ct.return_result()