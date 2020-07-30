import pickle

RunFilePath = "../ApplicationRuns/" #Used to save objects in ML_API
RunFilePath1 = "../../ApplicationRuns/" #Used in miniFlows

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
    obj = pickle.load(open(filepath+fileName+'.sav', 'wb'))
    return obj
