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

from MLAccelerator import MLAccelerator

from flask import Flask, render_template, request, redirect, url_for
import pandas as pd

app = Flask(__name__)


# render webpage
@app.route('/')
def home():
    return render_template('Home.html')


@app.route('/', methods=['POST', 'GET'])
def get_data():
    if request.method == 'POST':
        flag_list ={}
        final_list={}
        metric_dict={'accuracy_score':accuracy_score,'f1_score':f1_score}
        flag_list['outlier']=request.form.getlist('outlier')
        flag_list['nullhandle']=request.form.getlist('nullhandle')
        flag_list['feature']=request.form.getlist('feature')
        flag_list['encoding']=request.form.getlist('encoding')
        flag_list['metric']=request.form.getlist('metric')
        flag_list['model']=request.form.getlist('model')
        for key in flag_list:
            if not flag_list[key]:
                final_list[key] = [None]
            else:
                final_list[key] = flag_list[key]
        df = pd.read_csv(request.files.get('file'),delimiter=',')
        y = request.form['target']
        ml=MLAccelerator(df,y,final_list,metric_dict)
        result=ml.execute()
        return result

app.run()


