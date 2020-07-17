import pandas as pd
import os, shutil, sys 
sys.path.append('..')
from Accelerator.GlobalParameters import *
import json

class FeatureRedFlow:
    def __init__(self,appName):
        self.appName = appName
        self.df = pd.read_csv(RunFilePath1 + '/' + self.appName + '/' + inputFileName)
        print(self.df.head())
    

if __name__ =="__main__":
    featureflow = FeatureGenFlow('App2')