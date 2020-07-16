import pandas as pd
from Algos.NullHandling import NullHandling
from Algos.Outliers import Outliers
import os, shutil, sys 
sys.path.append('..')
from Accelerator.GlobalParameters import *
import json


class EDAFlow:
    def __init__(self,appName,nullcolTypes,outcolTypes):
        self.appName = appName
        self.nullcolTypes = nullcolTypes
        self.outcolTypes = outcolTypes
        self.df = pd.read_csv(RunFilePath1 + '/' + self.appName + '/' + inputFileName)
        print(self.df.head())
        self.nullhandlings(df,)

    def nullhandlings(self,df,nullcolTypes):
        null_obj = NullHandling(self.df,self.nullcolTypes)
        self.null_obj.to_csv(RunFilePath1 + '/' + self.appName + '/' + inputFileName, index=False)
    
    def outlierhandling(self,df,outcolTypes):
        outlier_obj = OutlierHandling(self.df,self.outcolTypes)
        self.outlier_obj.to_csv(RunFilePath1 + '/' + self.appName + '/' + inputFileName, index=False)

    
if __name__ =="__main__":
    eda = EDAFlow('App2')