import pandas as pd
import json
# from .Algos.CombinationalTransform import CombinationalTransform
# from .Algos.MathsTransform import MathsTransform
# from .Accelerator.FeatureRed.Algos.Correlations import Correlations
# import os, shutil, sys 
# #sys.path.append('..')
# from Accelerator.GlobalParameters import *

from Algos.CombinationalTransform import CombinationalTransform
from Algos.MathsTransform import MathsTransform
import os, shutil, sys 
sys.path.append('..')
from Accelerator.FeatureRed.Algos.Correlations import Correlations
from Accelerator.GlobalParameters import *

class FeatureGenFlow:
    def __init__(self,appName):
        self.appName = appName
        self.df = pd.read_csv(RunFilePath1 + '/' + self.appName + '/' + inputFileName)
        print(self.df.head())
        file_path = RunFilePath1 + '/' +self.appName + '/' + 'config.json'
        with open(file_path) as json_file:
            json_data = json.load(json_file)
        self.target = json_data['targetCol']
        col_map = {}
        numeric_list = []
        math_oper = ['sqrt','cbrt','log','reciprocal']
        for key,value in json_data['colTypes'].items():
            if value == 'Numeric':
                numeric_list.append(key)
                for oper in math_oper:
                    if oper in col_map:
                        col_map[oper].append(key)
                    else:
                        col_map[oper] = [key]
        print(numeric_list)
        self.df = self.mathstransform(self.df,col_map)
        print(self.target)
        self.df = self.correlations(self.df,numeric_list)
        print(self.df.head())
        #self.combtransform(self.df)

    def correlations(self, df, numeric_list):
        sel_col = []
        numeric_list.append(self.target)
        for col in numeric_list:
            df_red = df.filter(regex=col)
            #df_red[self.target] = df[self.target]
            print(df_red.head())
            sel_col.append(Correlations(df_red,numeric_list, self.target,True))
        self.df = df[sel_col]
        return self.df


    def mathstransform(self, df, col_map):
        self.math_obj = MathsTransform(df, col_map)
        self.df = self.math_obj.return_value()
        return self.df
    
    def combtransform(self,df):
        self.comb_obj = CombinationalTransform(self.df)
        self.df = self.comb_obj.return_value()

if __name__ =="__main__":
    featureflow = FeatureGenFlow('App2')
