from flask import Flask, request
from .Initialisation.InitFlow import InitFlow
from .EDA.EDAFlow import EDAFlow
from .GlobalParameters import *

app = Flask(__name__)

@app.route('/', method = ['POST'])
def inputs():
    if request.method=='POST':
        appName = request.form.get('appName')
        filePath = request.form.get('filePath')
        sep = request.form.get('sep')

        obj = InitFlow(appName, filePath, sep)

        saveObj(obj, RunFilePath+appName+'/'+InitialisationFolder+'/', 'classObject')

        return "Accelerator Initialised"

@app.route('/settingTarget', method = ['POST'])
def settingTarget():
    if request.method=='POST':
        appName = request.form.get('appName')
        targetColName = request.form.get('targetColumn')

        obj = readObj(RunFilePath+appName+'/'+InitialisationFolder+'/', 'classObject')
        obj.targetCol(targetColName)

        saveObj(obj, RunFilePath+appName+'/'+InitialisationFolder+'/', 'classObject')

        return 'Target Column Set'

@app.route('/detectingColTypes', method= ['GET'])
def colTypesDetection():
    if request.method == 'GET':
        appName = request.form.get('appName')
        obj = readObj(RunFilePath+appName+'/'+InitialisationFolder+'/', 'classObject')
        obj.detectingColTypes()
        saveObj(obj, RunFilePath+appName+'/'+InitialisationFolder+'/', 'classObject')

        return obj.colTypes

@app.route('/initSave', method = ['POST'])
def initSave():
    if request.method=='POST':
        appName = request.form.get('appName')
        confirmedColTypes = request.form.get('confirmedColTypes')

        obj = readObj(RunFilePath+appName+'/'+InitialisationFolder+'/', 'classObject')
        obj.confirmingColTypes(confirmedColTypes)
        saveObj(obj, RunFilePath+appName+'/'+InitialisationFolder+'/', 'classObject')

        obj.save_results()
        return 'Column Types Updated'

@app.route('/eda', method=['GET'])
def edaSuggestion():
    if request.method == 'GET':
        appName = request.form.get('appName')

        obj = EDAFlow(appName)
        suggestions = obj.suggestions()

        saveObj(obj, RunFilePath+appName+'/'+EDAFolder+'/', 'classObject')

        return suggestions

@app.route('EDATransformations', method = ['POST'])
def edaTransform():
    if request.method == 'POST':
        appName = request.form.get('appName')
        nullcolTypes = request.form.get('nullcolTypes')
        outcolTypes = request.form.get('outcolTypes')
        enccolTypes = request.form.get('enccolTypes')
        textcolTypes = request.form.get('textcolTypes')

        obj = readObj(RunFilePath+appName+'/'+EDAFolder+'/', 'classObject')
        obj.trasformations(nullcolTypes, outcolTypes, enccolTypes, textcolTypes)
        saveObj(obj, RunFilePath+appName+'/'+EDAFolder+'/', 'classObject')

        return  "EDA Transformation Complete"
















