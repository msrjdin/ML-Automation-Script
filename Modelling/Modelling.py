from functools import partial
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from hyperopt import tpe, hp, fmin,space_eval,STATUS_OK,Trials
from sklearn.metrics import mean_squared_error
from sklearn.metrics import log_loss
from sklearn.metrics import accuracy_score,f1_score
import copy
import pandas as pd
import numpy as np


class Modelling:

    def __init__(self,x_train,y_train,x_test,y_test,metric):
        self.x_train=x_train.copy(deep=True)
        self.y_train=y_train.copy(deep=True)
        self.x_test=x_test.copy(deep=True)
        self.y_test=y_test.copy(deep=True)
        self.metric=metric
        #self.test_score = {}
        space = hp.choice('classifier',[
        {'model': RandomForestClassifier,
        'param':{'max_depth': hp.choice('max_depth', range(1,20)),
        'max_features': hp.choice('max_features', range(1,5)),
        'n_estimators': hp.choice('n_estimators', range(1,20)),
        'criterion': hp.choice('criterion', ["gini", "entropy"])
                }
        },
        {'model': LogisticRegression,
        'param': {'penalty':hp.choice('penalty',['l2']),
        'C':hp.lognormal('C',0,1),
        'solver':hp.choice('solver',['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'])}
        }
        ])
        self.space=space
        
    def objective_func(self,args):
        if args['model']==RandomForestClassifier:
            n_estimators = args['param']['n_estimators']
            max_depth = args['param']['max_depth']
            max_features = args['param']['max_features']
            clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, max_features=max_features)
        elif args['model']==LogisticRegression:
            penalty = args['param']['penalty']
            C= args['param']['C']
            solver = args['param']['solver']
            clf = LogisticRegression(penalty=penalty,C=C,solver=solver)
        clf.fit(self.x_train,self.y_train)
        y_pred_train = clf.predict(self.x_train)
        y_pred_test = clf.predict(self.x_test)
        loss = log_loss(self.y_train,y_pred_train)
        score = self.metric(y_pred_test,self.y_test)
        print("Test Score:",self.metric(y_pred_test,self.y_test))
        print("Train Score:",self.metric(y_pred_train,self.y_train))
        print("\n===============")
        #return {'loss': score,'status': STATUS_OK,'model': clf}
        return (-score)
    
    ##def return_score(self):
        #return self.test_score 
    
    