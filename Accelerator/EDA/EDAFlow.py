import pandas as pd
from Algos.NullHandling import NullHandling
from Algos.OutlierHandling import OutlierHandling
from Algos.Encoding import Encoding
from Algos.TextProcessing import TextProcessing
import os, shutil, sys

# sys.path.append('..')
from Accelerator.GlobalParameters import *
import json


class EDAFlow:
    def __init__(self, appName, nullcolTypes, outcolTypes, enccolTypes, textcolTypes={}):
        self.appName = appName
        self.nullcolTypes = nullcolTypes
        self.outcolTypes = outcolTypes
        self.enccolTypes = enccolTypes
        self.textcolTypes = textcolTypes
        self.df = pd.read_csv(RunFilePath1 + '/' + self.appName + '/' + inputFileName)
        print(self.df.head())
        self.nullhandlings(self.df, self.nullcolTypes)
        self.outlierhandling(self.df, self.outcolTypes)
        self.encoding(self.df, self.enccolTypes)
        self.textprocessing(self.df, self.textcolTypes)

    def nullhandlings(self, df, nullcolTypes):
        self.null_obj = NullHandling(self.df, self.nullcolTypes)
        self.df = self.null_obj.return_result()
        # null_obj.to_csv(RunFilePath1 + '/' + self.appName + '/' + inputFileName, index=False)
        print(self.df)

    def outlierhandling(self, df, outcolTypes):
        self.outlier_obj = OutlierHandling(self.df, self.outcolTypes)
        # outlier_obj.to_csv(RunFilePath1 + '/' + self.appName + '/' + inputFileName, index=False)
        self.df = self.outlier_obj.return_result()

    def encoding(self, df, enccolTypes):
        self.encoding_obj = Encoding(self.df, self.enccolTypes)
        # encoding_obj.to_csv(RunFilePath1 + '/' + self.appName + '/' + inputFileName, index=False)
        self.df = self.encoding_obj.return_result()
        self.df.to_csv(RunFilePath1 + '/' + self.appName + '/' + inputFileName, index=False)

    def textprocessing(self, df, textcolTypes):
        if self.textcolTypes:
            self.text_obj = TextProcessing(self.df, self.textcolTypes)
            # text_obj.to_csv(RunFilePath1 + '/' + self.appName + '/' + inputFileName, index=False)
            self.df = self.text_obj.return_result()
            self.df.to_csv(RunFilePath1 + '/' + self.appName + '/' + inputFileName, index=False)


if __name__ == "__main__":
    # 'remove columns':[]
    nullcolTypes = {'impute mean': ['Fare'], 'impute knn': ['PassengerId'], 'categorical impute': ['Sex']}
    outcolTypes = {'capping': ['Fare', 'Age'], 'zscore': ['PassengerId']}
    enccolTypes = {'label': ['Sex'], 'one-hot': ['Embarked']}
    # textcolTypes = {'ingredients': ['stemming', 'lemmetize'],'ingredient': ['stemming']}
    textcolTypes = {}
    eda = EDAFlow('App2', nullcolTypes, outcolTypes, enccolTypes, textcolTypes)