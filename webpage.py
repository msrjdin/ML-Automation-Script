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


@app.route('/coltypes1', methods=['POST', 'GET'])
def coltypes1():
    if request.method == 'POST':
        df = pd.read_csv(request.files.get('file'),delimiter=',')
        MLAcc.df = df.copy()
        MLAcc.colIdentification()
        dtypes = MLAcc.dtypes
        return render_template('col_types.html',data = dtypes)

@app.route('/targetgraphs', methods=['POST', 'GET'])
def targetgraphs():
    if request.method == 'POST':
        targetCol = request.form['target']
        col_names = MLAcc.dtypes.keys()
        for key in col_names:
            MLAcc.dtypes[key] = request.form[key]
        MLAcc.colConfirmation()

        MLAcc.y = targetCol
        MLAcc.targetType =  MLAcc.dtypes[targetCol]
        col = MLAcc.TargetGraphs()
        columns = MLAcc.df.columns
        dict_col = {'features':col,'columns':columns}
        return render_template('graphs.html', data=dict_col)

@app.route('/submit',methods=['POST','GET'])
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
        flag_list['textp']= request.form.getlist('textp')
        cols_list = request.form.getlist('columns')
        for key in flag_list:
            if not flag_list[key]:
                final_list[key] = [None]
            else:
                final_list[key] = flag_list[key]
        for k,v in MLAcc.colTypes.items():
            MLAcc.colTypes[k] = [x for x in v if x in cols_list]
        MLAcc.df = MLAcc.df[cols_list].copy()
        MLAcc.final_list = final_list
        MLAcc.metric_dict = metric_dict
        print(flag_list['textp'])
        if len(flag_list['textp']) > 0 or len(flag_list['vector']) > 0:
            columns = MLAcc.colTypes['Text']
        else:
            columns = []
        result = MLAcc.execute()
        results = {'result':result,'columns':columns}
    return render_template('Output.html',data = results)

app.run()

