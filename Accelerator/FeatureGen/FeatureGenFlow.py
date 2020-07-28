import pandas as pd
from Algos.CombinationalTransform import CombinationalTransform
from Algos.MathsTransform import MathsTransform
import os, shutil, sys 
sys.path.append('..')
from Accelerator.GlobalParameters import *
import json

class FeatureGenFlow:
    def __init__(self,appName):
        self.appName = appName
        self.df = pd.read_csv(RunFilePath1 + '/' + self.appName + '/' + inputFileName)
        print(self.df.head())
        self.mathstransform(self.df)
        self.combtransform(self.df)
    
    def mathstransform(self,df):
        self.math_obj = MathsTransform(self.df)
        self.df = self.math_obj.return_value()
    
    def combtransform(self,df):
        self.comb_obj = CombinationalTransform(self.df)
        self.df = self.comb_obj.return_value()

if __name__ =="__main__":
    featureflow = FeatureGenFlow('App2')
