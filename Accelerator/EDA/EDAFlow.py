import pandas as pd
from Algos.NullHandling import NullHandling
from Algos.OutlierHandling import OutlierHandling
from Algos.Encoding import Encoding 
from Algos.TextProcessing import TextProcessing
import os, shutil, sys
# sys.path.append('..')
# from Accelerator.GlobalParameters import *
import json
from Accelerator.GlobalParameters import *


class EDAFlow:
    def __init__(self,appName):
        self.appName = appName
        self.df = pd.read_csv(RunFilePath1 + '/' + self.appName + '/' + inputFileName)
        self.suggestions()
    
    def suggestions(self):
        self.nullcolTypes = {}
        self.outcolTypes = {}
        self.enccolTypes = {}
        self.textcolTypes = {}

    def trasformations(self, nullcolTypes, outcolTypes, enccolTypes, textcolTypes):
        self.nullhandlings(nullcolTypes)
        self.outlierhandling(outcolTypes)
        self.encoding(enccolTypes)
        self.textprocessing(textcolTypes)
        self.save_results()

    def nullhandlings(self,nullcolTypes):
        if nullcolTypes is not None:
            null_obj = NullHandling(self.df,nullcolTypes)
            self.df = null_obj.return_result()
    
    def outlierhandling(self,outcolTypes):
        outlier_obj = OutlierHandling(self.df, outcolTypes)
        self.df = outlier_obj.return_result()
    
    def encoding(self, enccolTypes):
        encoding_obj = Encoding(self.df, enccolTypes)
        self.df = encoding_obj.return_result()
    
    def textprocessing(self, textcolTypes):
        if textcolTypes:
            text_obj = TextProcessing(self.df,textcolTypes)
            self.df = text_obj.return_result()

    def save_results(self):
        self.df.to_csv(RunFilePath1 + '/' + self.appName + '/' + EDAFolder + '/' + inputFileName, index=False)

if __name__ =="__main__":
    #'remove columns':[]
    nullcolTypes = {'impute mean':['Fare'],'impute knn':['PassengerId'],'categorical impute':['Sex']}
    outcolTypes = {'capping':['Fare','Age'],'zscore':['PassengerId']}
    enccolTypes = {'label':['Sex'] ,'one-hot':['Embarked']}
    #textcolTypes = {'ingredients': ['stemming', 'lemmetize'],'ingredient': ['stemming']} 
    textcolTypes = { }
    eda = EDAFlow('App2')
    eda.trasformations(nullcolTypes,outcolTypes,enccolTypes,textcolTypes)