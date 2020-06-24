#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from EDA.OutlierHandling import OutlierHandling
from EDA.Encoding import Encoding
from EDA.NullHandling import NullHandling
from EDA.ColumnTypeIdentification import ColumnTypeIdentification
from EDA.FeatureReduction import FeatureReduction
import warnings
warnings.filterwarnings("ignore")
import threading
from Modelling.Classification import Classification
from Modelling.Regression import Regression
from sklearn.metrics import accuracy_score,f1_score,mean_squared_error
from itertools import product
from pathlib import Path
import json 
import matplotlib.pyplot as plt
import seaborn as sn







# In[2]:


from MLAccelerator import MLAccelerator


# In[3]:


from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
app = Flask(__name__)


# render webpage
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST', 'GET'])
def upload():
    if request.method == 'POST':
        df = pd.read_csv(request.files.get('file'),delimiter=',')
        df.to_csv('dataframe.csv',index= False)
        corrMatrix = df.corr()
        svm=sn.heatmap(corrMatrix, annot=True, cmap='coolwarm', linewidth=2)
        figure = svm.get_figure()    
<<<<<<< HEAD
        figure.savefig('C:\\Users\\RavikanthReddyKandad\\Documents\\git_latest\\ML-Automation-Script-june\\static\\corr_pic1.png', dpi=400,bbox_inches='tight')
        return render_template('corr_page.html')

@app.route('/coltypes1', methods=['POST', 'GET'])
def coltypes1():
    if request.method == 'POST':
        df = pd.read_csv('dataframe.csv')
        ml = MLAccelerator(df,'',[],[])
        coltype = ml.colIdentification(df, 'Survived')
        print(coltype)
        coltype = {'Survived':'Categorical', 'Pclass':'Categorical', 'Xyz':'Numeric','abc':'Text'}
        # types = [k for k in coltype.keys() for v in coltype[k]]
        # cols = [v for k in coltype.keys() for v in coltype[k]]
        # #data = pd.DataFrame.from_dict({'types': types, 'coltypes': cols})
        return render_template('col_types.html',data = coltype)

#col_types.html returns modified dictionary of dtypes
# coltypeconfirm function calls ml program updates the Siraj column confirmation code.
# def coltypeconfirm():
#     # ml.colIdentification(df,parameters,target_name)
#
@app.route('/targetgraphs', methods=['POST', 'GET'])
def targetgraphs():
    if request.method == 'POST':
        df = pd.read_csv('dataframe.csv')
        #colTypes = {'Survived':'Categorical' ,'Pclass':'Categorical' ,'Age':'Numeric' ,'SibSp':'Numeric' ,'Fare':'Numeric' ,'Parch':'Categorical'}
        colTypes = {'Categorical': ['Survived','Pclass','SibSp','Parch'], 'Numeric':['Age','Fare']}
        ml = MLAccelerator(df, 'Survived', [], [])
        print(dir(ml))
        col = ml.TargetGraphs(df, colTypes, 'Survived', 'Categorical')
        return render_template('graphs.html', data=col)
    # ml.cleaning methods
     # ml.target graph
     # return target_3screen.html

# def selecting_methods():
#     # ml.function_top - column-parameters
#     # ##return 4th screen
#
#
# def model_execution():
#     # model ex 5th screen
# #    return outputpage
#
#
=======
        figure.savefig('C:\\Users\\SindhuKarnati\\Desktop\\MLAccelarator\\static\\corr_pic1.png', dpi=400,bbox_inches='tight')
        return render_template('corr_page.html')



>>>>>>> ad6f30ce68088f7298abf7350447aa4af06630df
@app.route('/upload1', methods=['POST', 'GET'])
def upload1():
    if request.method == 'POST':
        df=pd.read_csv('dataframe.csv')
        df.to_csv('dataframe.csv',index= False)
        cols= list(df)
        return render_template('Home.html',data = cols)


@app.route('/submit',methods=['POST', 'GET'])   
def submit():
    if request.method == 'POST':
        flag_list ={}
        final_list={}
        cols_list=[]
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
        df=pd.read_csv('dataframe.csv')
        ml=MLAccelerator(df[cols_list],y,final_list,metric_dict)
        result=ml.execute()
        return render_template('Output.html',data = result)
        


# In[ ]:
<<<<<<< HEAD
app.run()
=======

app.run()




>>>>>>> ad6f30ce68088f7298abf7350447aa4af06630df
