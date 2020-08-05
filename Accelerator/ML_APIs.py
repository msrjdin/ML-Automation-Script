from flask import Flask, request
from Accelerator.Initialisation.InitFlow import InitFlow
from Accelerator.EDA.EDAFlow import EDAFlow
from Accelerator.GlobalParameters import *
from Accelerator.FeatureGen.FeatureGenFlow import  FeatureGenFlow
import pandas as pd

app = Flask(__name__)

@app.route('/', methods = ['POST'])
def inputs():
    if request.method=='POST':
        appName = request.form.get('appName')
        file = request.files['file']
        sep = request.form.get('sep')

        obj = InitFlow(appName, file, sep)

        saveObj(obj, RunFilePath+appName+'/'+InitialisationFolder+'/', 'classObject')

        return "Accelerator Initialised"

@app.route('/settingTarget', methods = ['POST'])
def settingTarget():
    if request.method=='POST':
        appName = request.form.get('appName')
        targetColName = request.form.get('targetColumn')

        obj = readObj(RunFilePath+appName+'/'+InitialisationFolder+'/', 'classObject')
        obj.targetCol(targetColName)

        saveObj(obj, RunFilePath+appName+'/'+InitialisationFolder+'/', 'classObject')

        return 'Target Column Set'

@app.route('/detectingColTypes')
def colTypesDetection():
    if request.method == 'GET':
        appName = request.form.get('appName')

        obj = readObj(RunFilePath+appName+'/'+InitialisationFolder+'/', 'classObject')
        obj.detectingColTypes()
        saveObj(obj, RunFilePath+appName+'/'+InitialisationFolder+'/', 'classObject')

        return obj.colTypes

@app.route('/initSave', methods = ['POST'])
def initSave():
    if request.method=='POST':
        response = request.json
        appName = response['appName']
        confirmedColTypes = response['confirmedColTypes']

        obj = readObj(RunFilePath+appName+'/'+InitialisationFolder+'/', 'classObject')
        obj.confirmingColTypes(confirmedColTypes)
        saveObj(obj, RunFilePath+appName+'/'+InitialisationFolder+'/', 'classObject')

        obj.save_results()
        return 'Column Types Updated'

@app.route('/eda')
def edaSuggestion():
    if request.method == 'GET':
        appName = request.form.get('appName')

        obj = EDAFlow(appName)
        suggestions = obj.suggestions()

        saveObj(obj, RunFilePath+appName+'/'+EDAFolder+'/', 'classObject')

        return suggestions

@app.route('/edaTransformations', methods = ['POST'])
def edaTransform():
    if request.method == 'POST':
        response = request.json

        appName = response['appName']
        suggestions ={
            "nullcolTypes" : response['nullcolTypes'],
            "outcolTypes" : response['outcolTypes'],
            "enccolTypes" : response['enccolTypes'],
            "textcolTypes" : response['textcolTypes']
        }

        obj = readObj(RunFilePath+appName+'/'+EDAFolder+'/', 'classObject')
        obj.transformations(suggestions)
        saveObj(obj, RunFilePath+appName+'/'+EDAFolder+'/', 'classObject')

        return  "EDA Transformation Complete"


@app.route('/featureGen')
def featureGen():
    if request.method == 'POST':
        appName = request.form.get('appName')

        obj = FeatureGenFlow(appName)

        saveObj(obj, RunFilePath+appName+'/'+FRFolder+'/', 'classObject')

        return "Feature Generation Completed"



app.run(debug = True)














