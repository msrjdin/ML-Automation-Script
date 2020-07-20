import pandas as pd
# noinspection PyUnresolvedReferences
from Algos.DetectingColTypes import DetectingColTypes
# noinspection PyUnresolvedReferences
from Algos.Insights import Insights
import os, shutil, sys
#sys.path.append('..')
from Accelerator.GlobalParameters import *
import json

class InitFlow:
    def __init__(self, appName, inputFile, separator=','):
        self.appName = appName
        self.df = pd.read_csv(inputFile, sep=separator)
        self.projectStructure()

    def projectStructure(self):
        shutil.rmtree(RunFilePath1 + self.appName, ignore_errors=True)
        os.makedirs(RunFilePath1 + self.appName)
        os.mkdir(RunFilePath1 + self.appName + '/' + InitialisationFolder)
        os.mkdir(RunFilePath1 + self.appName + '/' + EDAFolder)
        os.mkdir(RunFilePath1 + self.appName + '/' + FRFolder)
        os.mkdir(RunFilePath1 + self.appName + '/' + FGFolder)
        os.mkdir(RunFilePath1 + self.appName + '/' + ModellingFolder)
        os.mkdir(RunFilePath1 + self.appName + '/' + ProcessedFolder)
        os.mkdir(RunFilePath1 + self.appName + '/' + PredictionsFolder)


    #Detection of ColTypes from the Algorithm
    def detectingColType(self):
        obj = DetectingColTypes(self.df)
        self.colTypes = obj.returnValues()
        print(self.colTypes)

    # Input from the UI
    def targetCol(self, targetName):
        self.targetName = targetName

    def insights(self):
        obj = Insights(self.df, self.colTypes, self.targetName)
        self.ins = obj.returnValues()
        print(self.ins)


    #Confirmation of the COlTypes taken from the UI via API
    def confirmingColTypes(self, confirmedColTypes):
        self.colTypes = confirmedColTypes
        
    #Saving Config File
    def save_results(self):
        config = {'appName': self.appName,
                'colTypes': self.colTypes,
                'targetCol': self.targetName}
        json.dump(config, open(RunFilePath1 +'/'+self.appName+'/'+configFile, 'w'))
        self.df.to_csv(RunFilePath1 + '/' + self.appName + '/' + inputFileName, index=False)


if __name__ =="__main__":
    print(sys.path)
    miniFlow = InitFlow('App2', "dataframe.csv")
    miniFlow.detectingColType()
    miniFlow.targetCol('Survived')
    miniFlow.insights()
    miniFlow.save()