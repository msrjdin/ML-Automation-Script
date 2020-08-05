import pandas as pd
import json
# from .Algos.CombinationalTransform import CombinationalTransform
# from .Algos.MathsTransform import MathsTransform
# from .Accelerator.FeatureRed.Algos.Correlations import Correlations
# import os, shutil, sys 
# #sys.path.append('..')
# from Accelerator.GlobalParameters import *

from .Algos.CombinationalTransform import CombinationalTransform
from .Algos.MathsTransform import MathsTransform
import os, shutil, sys 
# sys.path.append('..')
from Accelerator.FeatureRed.Algos.Correlations import Correlations
from Accelerator.GlobalParameters import *

class FeatureGenFlow:
    def __init__(self, appName):
        self.appName = appName
        self.df = pd.read_csv(RunFilePath1 + '/' + self.appName + '/' + inputFileName)
        file_path = RunFilePath1 + '/' +self.appName + '/' + 'config.json'
        with open(file_path) as json_file:
            json_data = json.load(json_file)
        self.target = json_data['targetCol']
        self.target_values = self.df.loc[:,self.target]
        col_map = {}
        numeric_dict = {}

        math_oper = ['sqrt','cbrt','log','reciprocal']
        for key,value in json_data['colTypes'].items():
            if value == 'Numeric':
                numeric_dict[key] = value
                for oper in math_oper:
                    if oper in col_map:
                        col_map[oper].append(key)
                    else:
                        col_map[oper] = [key]

        self.df = self.mathstransform(self.df,col_map)
        self.df = self.correlations(self.df,numeric_dict)

        #self.combtransform(self.df)

    def correlations(self, df, numeric_dict):
        sel_col = []
        for col in numeric_dict:
            print(col)
            print(self.target)
            df_red = df.filter(regex=col)
            df_red[self.target] = self.target_values 
            print(df_red.head())
            corr_obj = Correlations(df_red,numeric_dict, self.target,True)
            sel_col.append(corr_obj.col)

        df = df[sel_col]
        return df

    def mathstransform(self, df, col_map):
        math_obj = MathsTransform(df, col_map)
        df = math_obj.return_result()
        return df
    
    def combtransform(self,df):
        self.comb_obj = CombinationalTransform(df)
        self.df = self.comb_obj.return_value()

# if __name__ =="__main__":
#     featureflow = FeatureGenFlow('App2')
