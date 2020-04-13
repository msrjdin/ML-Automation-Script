from functools import partial
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from hyperopt import tpe, hp, fmin,space_eval,STATUS_OK,Trials
from sklearn.metrics import mean_squared_error
from sklearn.metrics import log_loss
from sklearn.metrics import accuracy_score,f1_score
from Modelling import *
import copy
import pandas as pd
import numpy as np

hyperparams = {}


class BestModel:
    def __init__(self,y,df_all_en,metric):
    z=0
    self.y=y.copy()
    self.df_all_en =df_all_en.copy()
	self.metric=metric.copy()
    self.model_comparison()
    
    def model_comparison():
        for df in self.df_all_en:
            z=z+1
            x_train,x_test,y_train,y_test = train_test_split(df.drop(self.y,axis=1),df[self.y],test_size=0.2)
            obj=Modelling(x_train,y_train,x_test,y_test,self.metric)
            trials=Trials()
            score=0
            temp_hyperparam=space_eval(obj.space, fmin(obj.objective_func,obj.space,trials=trials,algo=tpe.suggest,max_evals=100))
            if  -min(trials.losses())>score :
                hyperparam=temp_hyperparam
                name_=str(z)+'_'+str(metric)
                score=-min(trials.losses())
   
    
    hyperparams[name_] = hyperparam
    hyperparams[name_]['score'] = -min(trials.losses())
    return hyperparams,df_all_en[int(name_[0:1])]