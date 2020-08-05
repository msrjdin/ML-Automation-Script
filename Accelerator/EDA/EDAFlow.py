
import pandas as pd
from .Algos.NullHandling import NullHandling
from .Algos.OutlierHandling import OutlierHandling
from .Algos.Encoding import Encoding
from .Algos.TextProcessing import TextProcessing
import os, shutil, sys
import json
from Accelerator.GlobalParameters import *
import json
#from Accelerator.GlobalParameters import *
#sys.path.append('..')
from .Algos.AutoSuggest import AutoSuggest

class EDAFlow:
    def __init__(self, appName):
        self.appName = appName
        self.df = pd.read_csv(RunFilePath1 + '/' + self.appName + '/' + inputFileName)

    def suggestions(self):
        sugg = AutoSuggest(self.df, self.appName)
        sugg = {'nullcolTypes': sugg.nullcolTypes,
                'outcolTypes':sugg.outcolTypes,
                'enccolTypes':sugg.enccolTypes,
                'textcolTypes':sugg.textcolTypes}
        return sugg

    def transformations(self, suggestions):
        self.df = self.nullhandlings(suggestions['nullcolTypes'])
        self.outlierhandling(suggestions['outcolTypes'])
        self.encoding(suggestions['enccolTypes'])
        self.textprocessing(suggestions['textcolTypes'])
        self.save_results()

    def nullhandlings(self, nullcolTypes):
        if nullcolTypes:
            null_obj = NullHandling(self.df, nullcolTypes)
            return null_obj.return_result()
    
    def outlierhandling(self,outcolTypes):
        if outcolTypes is not None:
            outlier_obj = OutlierHandling(self.df, outcolTypes)
            self.df = outlier_obj.return_result()
    
    def encoding(self, enccolTypes):
        if enccolTypes is not None:
            encoding_obj = Encoding(self.df, enccolTypes)
            self.df = encoding_obj.return_result()
    
    def textprocessing(self, textcolTypes):
        if textcolTypes:
            text_obj = TextProcessing(self.df, textcolTypes)
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
