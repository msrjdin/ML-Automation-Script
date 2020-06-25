#!/usr/bin/env python
# coding: utf-8


import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import accuracy_score,f1_score,mean_squared_error
import seaborn as sn
from MLAccelerator import MLAccelerator

from flask import Flask, render_template, request, redirect, url_for
import pandas as pd

app = Flask(__name__)
MLAcc = MLAccelerator()

# render webpage
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST', 'GET'])
def upload():
    if request.method == 'POST':
        df = pd.read_csv(request.files.get('file'),delimiter=',')
        # df.to_csv('dataframe.csv',index= False)
        MLAcc.df = df.copy()

        corrMatrix = df.corr()
        svm=sn.heatmap(corrMatrix, annot=True, cmap='coolwarm', linewidth=2)
        figure = svm.get_figure()
        figure.savefig('static\\corr_pic1.png', dpi=400,bbox_inches='tight')
        return render_template('corr_page.html')


@app.route('/coltypes1', methods=['POST', 'GET'])
def coltypes1():
    if request.method == 'POST':
        # df = pd.read_csv('dataframe.csv')
        MLAcc.colIdentification()
        dtypes = MLAcc.dtypes
        # coltype = {'Survived':'Categorical', 'Pclass':'Categorical', 'Xyz':'Numeric','abc':'Text'}
        # types = [k for k in coltype.keys() for v in coltype[k]]
        # cols = [v for k in coltype.keys() for v in coltype[k]]
        # data = pd.DataFrame.from_dict({'types': types, 'coltypes': cols})
        return render_template('col_types.html',data = dtypes)

#col_types.html returns modified dictionary of dtypes
# coltypeconfirm function calls ml program updates the Siraj column confirmation code.
# def coltypeconfirm():
#     # ml.colIdentification(df,parameters,target_name)

@app.route('/targetgraphs', methods=['POST', 'GET'])
def targetgraphs():
    if request.method == 'POST':
        # df = pd.read_csv('dataframe.csv')

        # colTypes = {'Categorical': ['Survived','Pclass','SibSp','Parch'], 'Numeric':['Age','Fare']}
        # ml = MLAccelerator(df, 'Survived', [], [])
        targetCol = request.form['target']
        print(targetCol)
        #*******
        # dtypes = request.form.getlist('AllTheColType')
        dtypes = {'Survived': 'Categorical', 'Pclass': 'Categorical', 'Age': 'Numeric', 'SibSp': 'Numeric',
                  'Fare': 'Numeric', 'Parch': 'Categorical'}

        MLAcc.y = targetCol
        MLAcc.targetType = dtypes[targetCol]
        MLAcc.dtypes = dtypes
        # print(dir(ml))
        col = MLAcc.TargetGraphs()
        return render_template('graphs.html', data=col)

#
# @app.route('/upload1', methods=['POST', 'GET'])
# def upload1():
#     if request.method == 'POST':
#         df=pd.read_csv('dataframe.csv')
#         df.to_csv('dataframe.csv',index= False)
#         cols= list(df)
#         return render_template('Home.html',data = cols)

@app.route('/submit',methods=['POST', 'GET'])   
def submit():
    if request.method == 'POST':
        flag_list ={}
        final_list={}
        cols_list = []
        metric_dict={'accuracy_score':accuracy_score,'f1_score':f1_score,'mean_squared_error':mean_squared_error}
        flag_list['outlier']=request.form.getlist('outlier')
        flag_list['nullhandle']=request.form.getlist('nullhandle')
        flag_list['feature']=request.form.getlist('feature')
        flag_list['encoding']=request.form.getlist('encoding')
        flag_list['metric']=request.form.getlist('metric')
        flag_list['model']=request.form.getlist('model')
        flag_list['vector']=request.form.getlist('vector')
        cols_list = request.form.getlist('columns')
        for key in flag_list:
            if not flag_list[key]:
                final_list[key] = [None]
            else:
                final_list[key] = flag_list[key]
        y = request.form['target']
        # df=pd.read_csv('dataframe.csv')

        # ml=MLAccelerator(df[cols_list],y,final_list,metric_dict)
        MLAcc.df = MLAcc.df[cols_list].copy()
        MLAcc.final_list = final_list
        MLAcc.metric_dict = metric_dict

        result=MLAcc.execute()

        return render_template('Output.html',data = result)

        

app.run(debug=True)

