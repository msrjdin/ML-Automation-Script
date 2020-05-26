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
from sklearn.metrics import accuracy_score,f1_score
from itertools import product
from pathlib import Path
import json 


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
        cols= list(df)
        return redirect(url_for('push_data', columns=cols))


@app.route('/push_data/<columns>')
def push_data(columns):
    columns = columns.replace('[','').replace(']','').replace("'",'').split(',')
    return render_template('Home.html',data = columns)


@app.route('/submit',methods=['POST', 'GET'])   
def submit():
    if request.method == 'POST':
        flag_list ={}
        final_list={}
        cols_list=[]
        metric_dict={'accuracy_score':accuracy_score,'f1_score':f1_score}
        flag_list['outlier']=request.form.getlist('outlier')
        flag_list['nullhandle']=request.form.getlist('nullhandle')
        flag_list['feature']=request.form.getlist('feature')
        flag_list['encoding']=request.form.getlist('encoding')
        flag_list['metric']=request.form.getlist('metric')
        flag_list['model']=request.form.getlist('model')
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

app.run()




