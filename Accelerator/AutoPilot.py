import requests
from .GlobalParameters import *

class AutoPilot:
    def __init__(self, appName):
        self.payload = {'appName': appName}

    def inputs(self, filePath, sep):
        payload = self.payload.copy()
        payload['filePath'] = filePath
        payload['sep'] = sep

        response = APICall(restAPI="POST", route='', payload=payload)
        print(response.text.encode('utf8'))

    def targetSet(self, targetCol):
        payload = self.payload.copy()
        payload['targetColumn'] = targetCol

        response = APICall(restAPI="POST", route='settingTarget', payload=payload)
        print(response.text.encode('utf8'))

    def colTypeDetection(self):
        colTypes = APICall("GET", 'detectingColTypes', payload=self.payload)
        print(colTypes)
        return colTypes

    def confirmation(self, colTypes):
        payload = self.payload.copy()
        payload['confirmedColTypes'] = colTypes

        headers = {'Content-Type': 'application/json'}

        response = APICall(restAPI="POST", route='initSave', payload=payload, headers= headers)
        print(response.text.encode('utf8'))

    def edaSuggestions(self):
        suggestion = APICall(restAPI="GET", route='eda', payload=self.payload)
        print(suggestion.text.encode('utf8'))
        return suggestion

    def edaTransform(self, suggestionConfirmed):
        payload = self.payload.copy()
        payload['nullcolTypes'] = suggestionConfirmed['nullcolTypes']
        payload['outcolTypes'] = suggestionConfirmed['outcolTypes']
        payload['enccolTypes'] = suggestionConfirmed['enccolTypes']
        payload['textcolTypes'] = suggestionConfirmed['textcolTypes']

        response = APICall(restAPI="POST", route='edaTransformations', payload=payload)
        print(response.text.encode('utf8'))

    def featureGeneration(self):
        response = APICall(restAPI="POST", route='featureGen', payload=self.payload)
        print(response.text.encode('utf8'))





if __name__=='main':
    ap = AutoPilot('test')
    ap.inputs('../../dataframe.csv', ',')


