
from functools import partial
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from hyperopt import tpe, hp, fmin, space_eval, STATUS_OK, Trials
from sklearn.metrics import mean_squared_error
import sklearn.metrics as metrics
from sklearn.metrics import accuracy_score, f1_score
import copy
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score,mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.metrics import (confusion_matrix, precision_recall_curve, auc,
                             roc_curve, recall_score, classification_report, f1_score,
                             precision_recall_fscore_support)

# Parameters
# List of dfs, targetCol and the metric under consideration

class Regression:
    def __init__(self, dfs, targetCol, colTypes, metric, test_size=0.2):
        self.dfs = [dfs]
        self.y = targetCol
        self.metric = metric
        self.colTypes = copy.deepcopy(colTypes)
        self.test_size = test_size
        # self.test_score = {}
        self.final_results = {}

        self.execute()

    # Required to be passed in fmin of hyperopt
    def objective_func(self, args):
        clf = None

        if args['model'] == RandomForestRegressor:
            n_estimators = args['param']['n_estimators']
            max_depth = args['param']['max_depth']
            max_features = args['param']['max_features']
            clf = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, max_features=max_features)
        elif args['model'] == LinearRegression:
            normalize = args['param']['normalize']
            clf = LinearRegression(normalize=normalize)

        clf.fit(self.x_train, self.y_train)
        y_pred_train = clf.predict(self.x_train)
        y_pred_test = clf.predict(self.x_test)
        mae = metrics.mean_absolute_error(self.y_train, y_pred_train)
        mse = metrics.mean_squared_error(self.y_train, y_pred_train)
        rmse = np.sqrt(mse)
        score = metrics.mean_squared_error(y_pred_test, self.y_test)
        return {'loss': -score,'status': STATUS_OK,'other_stuff':y_pred_test}

    def execute(self):
        # using Hyperopt for parameter tuning
        self.space = hp.choice('regression', [
            {'model': RandomForestRegressor,
             'param': {'max_depth': hp.choice('max_depth', range(1, 20)),
                       'max_features': hp.choice('max_features', range(1, 2)),
                       'n_estimators': hp.choice('n_estimators', range(1, 20)),
                       'criterion': hp.choice('criterion', ["mse", "mae"])
                       }
             },
            {'model': LinearRegression,
             'param': {'normalize': hp.choice('normalize', ['True', 'False']),
                       }
             }
        ])

        score_key = 'score'

        for i, df in enumerate(self.dfs):
            df.drop(self.colTypes['Identity'], axis=1, inplace=True)  # Dropping Identity cols
            self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(df.drop(self.y, axis=1), df[self.y],
                                                                                    test_size=self.test_size)
            trials = Trials()

            hyperparam = space_eval(self.space,
                                    fmin(self.objective_func, self.space, trials=trials, algo=tpe.suggest,
                                         max_evals=100))
            score = min(trials.losses())

            for d in trials.results:
                if d['loss'] == score:
                    final_d = d

            y_pred_test = final_d['other_stuff']
            sm=sn.residplot(y_pred_test,self.y_test)
            fig = sm.get_figure()
            #fig.savefig("C:\\Users\\SindhuKarnati\\Desktop\\MLAccelarator\\static\\out.png")


            if i == 0:
                self.final_results['Hyperparameter'] = hyperparam
                self.final_results[score_key] = score
                self.final_results['Data'] = df

            if self.final_results[score_key] > score:
                self.final_results['Hyperparameter'] = hyperparam
                self.final_results[score_key] = score
                self.final_results['Data'] = df




    def return_results(self):
        return self.final_results

