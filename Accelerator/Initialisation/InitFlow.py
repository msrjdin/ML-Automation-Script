import pandas as pd
# noinspection PyUnresolvedReferences
from Algos.DetectingColTypes import DetectingColTypes
# noinspection PyUnresolvedReferences
from Algos.Insights import Insights
import os, shutil
from Accelerator.GlobalParameters import *
import json


class InitFlow:
    def __init__(self, appName, inputFile):
        print('Initialisation Class')
        self.appName = appName
        self.df = pd.read_csv(inputFile, sep='|')

        self.projectStructure()

    def projectStructure(self):
        shutil.rmtree(RunFilePath1 + self.appName, ignore_errors=True)
        os.mkdir(RunFilePath1 + self.appName)
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

    #Confirmation of the COlTypes taken from the UI via API
    def confirmingColTypes(self, confirmedColTypes):
        self.colTypes = confirmedColTypes

    def insights(self):
        obj = Insights(self.df)
        self.ins = obj.returnValues()

    #Input from the UI
    def targetCol(self, targetName):
        self.targetName = targetName

    #Saving Config File
    def save(self):
        config = {'appName': self.appName,
                'colTypes': self.colTypes,
                'targetCol': self.targetName}
        json.dump(config, open(RunFilePath1 +'/'+self.appName+'/'+configFile, 'w'))
        self.df.to_csv(RunFilePath1 + '/' + self.appName + '/' + inputFileName, index=False)


miniFlow = InitFlow('App1', '../../../Account_Master.txt')
miniFlow.detectingColType()
miniFlow.insights()
miniFlow.targetCol('Target')
miniFlow.save()
