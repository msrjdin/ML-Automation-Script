from flask import Flask, request
from .Initialisation.InitFlow import InitFlow
from.EDA.EDAFlow import EDAFlow

app = Flask(__name__)

@app.route('/', method = ['POST'])
def inputs():
    if request.method=='POST':
        filePath = request.form.get('filePath')
        sep = request.form.get('sep')
        appName = request.form.get('appName')
        app.name = appName

        obj = InitFlow(appName, filePath, sep)

        app.config['initObject'] = obj

        return "Accelerator Initialised"

@app.route('/settingTarget', method = ['POST'])
def settingTarget():
    if request.method=='POST':
        targetColName = request.form.get('targetColumn')

        obj = app.config['initObject']
        obj.targetCol(targetColName)

        app.config['initObject'] = obj

        return 'Target Column Set'

@app.route('/detectingColTypes', method= ['GET'])
def colTypesDetection():
    if request.method == 'GET':
        obj = app.config['initObject']
        obj.detectingColTypes()
        app.config['initObject'] = obj

        return obj.colTypes

@app.route('/initSave', method = ['POST'])
def initSave():
    if request.method=='POST':
        confirmedColTypes = request.form.get('confirmedColTypes')

        obj = app.config['initObject']
        obj.confirmingColTypes(confirmedColTypes)
        app.config['initObject'] = obj

        obj.save_results()
        return 'Column Types Updated'

@app.route('/eda', method=['GET'])
def edaSuggestion():
    if request.method == 'GET':
        obj = EDAFlow(app.name)
        suggestions = obj.suggestions()

        app.config['EDAObject'] = obj

        return suggestions

@app.route('EDATransformations', method = ['POST'])
def edaTransform():
    if request.method == 'POST':
        nullcolTypes = request.form.get('nullcolTypes')
        outcolTypes = request.form.get('outcolTypes')
        enccolTypes = request.form.get('enccolTypes')
        textcolTypes = request.form.get('textcolTypes')

        obj = app.config['EDAObject']
        obj.trasformations(nullcolTypes, outcolTypes, enccolTypes, textcolTypes)
        app.config['EDAObject'] = obj

        return  "EDA Transformation Complete"














