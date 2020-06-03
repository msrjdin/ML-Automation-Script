from functools import partial
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from hyperopt import tpe, hp, fmin, space_eval, STATUS_OK, Trials
from sklearn.metrics import mean_squared_error
from sklearn.metrics import log_loss
from sklearn.metrics import accuracy_score, f1_score
import copy
import pandas as pd
import numpy as np

#Parameters
#List of dfs, targetCol and the metric under consideration

class Classification:
    def __init__(self, dfs, targetCol, colTypes, metric, test_size=0.2, kfold=10):
        self.df = dfs
        self.y = targetCol
        self.metric = metric
        self.colTypes=copy.deepcopy(colTypes)
        self.test_size=test_size
        self.kfold=kfold
        # self.test_score = {}
        self.final_results={}

        self.execute()

    #Required to be passed in fmin of hyperopt
    def objective_func(self, args):
        clf=None

        if args['model'] == RandomForestClassifier:
            n_estimators = args['param']['n_estimators']
            max_depth = args['param']['max_depth']
            max_features = args['param']['max_features']
            clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, max_features=max_features)
        elif args['model'] == LogisticRegression:
            penalty = args['param']['penalty']
            C = args['param']['C']
            solver = args['param']['solver']
            clf = LogisticRegression(penalty=penalty, C=C, solver=solver)

        clf.fit(self.x_train, self.y_train)
        y_pred_train = clf.predict(self.x_train)
        y_pred_test = clf.predict(self.x_test)
        loss = log_loss(self.y_train, y_pred_train)
        score = self.metric(y_pred_test, self.y_test)
        # print("Test Score:", self.metric(y_pred_test, self.y_test))
        # print("Train Score:", self.metric(y_pred_train, self.y_train))
        # print("\n===============")
        # return {'loss': score,'status': STATUS_OK,'model': clf}
        return (-score)

    def execute(self):
        #using Hyperopt for parameter tuning
        self.space = hp.choice('classifier', [
                                            {'model': RandomForestClassifier,
                                            'param': {'max_depth': hp.choice('max_depth', range(1, 20)),
                                            'max_features': hp.choice('max_features', range(1, 5)),
                                            'n_estimators': hp.choice('n_estimators', range(1, 20)),
                                             'criterion': hp.choice('criterion', ["gini", "entropy"])
                                                    }
                                            },
                                            {'model': LogisticRegression,
                                             'param': {'penalty': hp.choice('penalty', ['l2']),
                                                       'C': hp.lognormal('C', 0, 1),
                                                       'solver': hp.choice('solver', ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'])}
                                             }
                                        ])

        score_key = 'score'

<<<<<<< HEAD

        for i, df in enumerate(self.dfs):
            df.drop(self.colTypes['Identity'], axis=1, inplace=True) #Dropping Identity cols
            self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(df.drop(self.y, axis=1), df[self.y], test_size=self.test_size)
            trials = Trials()
=======
        df=self.df.copy()
>>>>>>> 7db86ccf9079027446285fcbf67bdc0a735f8a71

        df.drop(self.colTypes['Identity'], axis=1, inplace=True) #Dropping Identity cols

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(df.drop(self.y, axis=1), df[self.y], test_size=self.test_size)
        trials = Trials()

        hyperparam = space_eval(self.space,
                                     fmin(self.objective_func, self.space, trials=trials, algo=tpe.suggest, max_evals=100))
        score = -min(trials.losses())

        self.final_results['Hyperparameter']=hyperparam
        self.final_results[score_key]=score
        self.final_results['Data']=df

    def return_results(self):
        return self.final_results
