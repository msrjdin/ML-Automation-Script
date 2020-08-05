import pickle
import requests

RunFilePath = "../ApplicationRuns/" #Used to save objects in ML_API
RunFilePath1 = "../ApplicationRuns/" #Used in miniFlows

inputFileName = 'input.csv'

configFile = 'config.json'

InitialisationFolder = 'Initialisation'
EDAFolder = 'EDA'
FRFolder = 'FR'
FGFolder = 'FG'
ModellingFolder = 'Modelling'
ProcessedFolder = 'Processed'
PredictionsFolder = 'Predictions'

# InitialisationFilePath1 = "../../ApplicationRuns/"

def saveObj(obj, filepath, fileName):
    pickle.dump(obj, open(filepath+fileName+'.sav', 'wb'))

def readObj(filepath, fileName):
    obj = pickle.load(open(filepath+fileName+'.sav', 'rb'))
    return obj


url = "http://127.0.0.1:5000/"


def APICall(restAPI, route, headers=[], payload={}, files=[]):
    response = requests.request(restAPI, url + route, headers=headers, data=payload, files=files)
    return response
