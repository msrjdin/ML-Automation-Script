import pandas as pd 
import numpy as np
import os, shutil, sys
#
# sys.path.append('..')
from Accelerator.GlobalParameters import *
import json
#from Accelerator.Initialisation.Algos.DetectingColTypes import DetectingColTypes

#input format of dictionary:key- column name and value col type
#ouput format of dictionary: key- handling method: value-list of columns for null, outlier, encoding and text process dictionary object types.
class AutoSuggest:
    def __init__(self, df, appName):
        self.nullcolTypes = {}
        self.outcolTypes = {}
        self.enccolTypes = {}
        self.textcolTypes = {}
        self.df = df
        self.appName = appName
        file_path = RunFilePath1 + '/' +self.appName + '/' + 'config.json'
        with open(file_path) as json_file:
            json_data = json.load(json_file)
        print(json_data['colTypes'])
        self.default_suggest(json_data['colTypes'])
        self.return_values()

    def default_suggest(self, colTypes):
        for key,value in colTypes.items():
            print(key,value)
            if value == 'Categorical':
                self.encoding_suggest(key)
            elif value == 'Numeric':
                self.outlier_suggest(key)
                self.null_suggest(key)
            elif value == 'Text':
                self.textproc_suggest(key)
    
    def encoding_suggest(self, column):
        #label
        if 'one-hot' in self.enccolTypes:
            self.enccolTypes['one-hot'].append(column)
        else:
            self.enccolTypes['one-hot'] = [column]

    def outlier_suggest(self, column):
        #zscore removing
        if 'capping' in self.outcolTypes:
            self.outcolTypes['capping'].append(column)
        else:
            self.outcolTypes['capping'] = [column]

    def null_suggest(self, column):
        if 'impute mean' in self.nullcolTypes:
            self.nullcolTypes['impute mean'].append(column)
        else:
            self.nullcolTypes['impute mean'] =[column]

    def textproc_suggest(self, column):
        if 'lemmetize' in self.textcolTypes:
            self.textcolTypes[column].append('lemmetize')
        else:
            self.textcolTypes[column] = ['lemmetize']

    def return_values(self):
        print(self.nullcolTypes)
        print(self.outcolTypes)
        print(self.enccolTypes)
        print(self.textcolTypes)

if __name__ =="__main__":
    df = pd.read_csv('../Accelerator/dataframe.csv')
    obj = AutoSuggest(df,'App2')