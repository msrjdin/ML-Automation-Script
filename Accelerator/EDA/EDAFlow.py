import pandas as pd
from Algos.NullHandling import NullHandling
from Algos.Outliers import Outliers
from Algos.Encoding import Encoding 
from Algos.TextProcessing import TextProcessing
import os, shutil, sys 
sys.path.append('..')
from Accelerator.GlobalParameters import *
import json


class EDAFlow:
    def __init__(self,appName,nullcolTypes,outcolTypes,enccolTypes,textcolTypes):
        self.appName = appName
        self.nullcolTypes = nullcolTypes
        self.outcolTypes = outcolTypes
        self.enccolTypes = enccolTypes
        self.textcolTypes = textcolTypes
        self.df = pd.read_csv(RunFilePath1 + '/' + self.appName + '/' + inputFileName)
        print(self.df.head())
        self.nullhandlings(self.df,self.nullcolTypes)
        self.outlierhandling(self.df,self.outcolTypes)
        self.encoding(self.df,self.enccolTypes)
        self.textprocessing(self.df,self.textcolTypes)

    def nullhandlings(self,df,nullcolTypes):
        null_obj = NullHandling(self.df,self.nullcolTypes)
        self.null_obj.to_csv(RunFilePath1 + '/' + self.appName + '/' + inputFileName, index=False)
        self.df = null_obj
    
    def outlierhandling(self,df,outcolTypes):
        outlier_obj = OutlierHandling(self.df,self.outcolTypes)
        self.outlier_obj.to_csv(RunFilePath1 + '/' + self.appName + '/' + inputFileName, index=False)
        self.df = outlier_obj
    
    def encoding(self,df,enccolTypes):
        encoding_obj = Encoding(self.df,self.enccolTypes)
        self.encoding_obj.to_csv(RunFilePath1 + '/' + self.appName + '/' + inputFileName, index=False)
        self.df = encoding_obj 

    def textprocessing(self,df,textcolTypes):
        text_obj = TextProcessing(self.df,self.textcolTypes)
        self.text_obj.to_csv(RunFilePath1 + '/' + self.appName + '/' + inputFileName, index=False)
        self.df = text_obj


if __name__ =="__main__":
    eda = EDAFlow('App2')
    