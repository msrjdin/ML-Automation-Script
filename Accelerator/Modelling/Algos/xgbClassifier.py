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
from sklearn.metrics import confusion_matrix
from xgboost import XGBClassifier


# df1 = pd.read_csv(r"C:\Users\SatyaSindhuMolleti\Desktop\ML-Automation-Script-master\ML-Automation-Script\Accelerator\dataframe.csv")
# df=df1[['Survived','Pclass','Fare']]



class xgbClassifier:
    def __init__(self, dfs, targetCol, metric, test_size=0.2):
        self.dfs = dfs
        self.y = targetCol
        self.metric = metric
        self.test_size = test_size
        # self.test_score = {}
        self.final_results = {}
        self.max_feat = len(self.dfs.columns)

        self.execute()

    # Required to be passed in fmin of hyperopt
    def objective_func(self, args):
        clf = None

        if args['model'] == XGBClassifier:
            n_estimators = args['param']['n_estimators']
            max_depth = args['param']['max_depth']
            criterion = args['param']['criterion']
            min_child_weight = args['param']['min_child_weight']
            subsample = args['param']['subsample']
            gamma = args['param']['gamma']
            colsample_bytree = args['param']['colsample_bytree']
            clf = XGBClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                   min_child_weight=min_child_weight,
                                subsample=subsample,gamma=gamma,colsample_bytree=colsample_bytree)

        clf.fit(self.x_train, self.y_train)
        y_pred_train = clf.predict(self.x_train)
        y_pred_test = clf.predict(self.x_test)
        # loss = log_loss(self.y_train, y_pred_train)
        score = self.metric(y_pred_test, self.y_test)
        # print("Test Score:", self.metric(y_pred_test, self.y_test))
        # print("Train Score:", self.metric(y_pred_train, self.y_train))
        # print("\n===============")
        return {'loss': -score, 'status': STATUS_OK, 'other_stuff': {'y_pred_test': y_pred_test, 'clf': clf}}


    def execute(self):
        # using Hyperopt for parameter tuning
        self.space = hp.choice('classifier', [
            {'model': XGBClassifier,
             'param': {'max_depth': hp.choice('max_depth', range(3, 10)),
                       'n_estimators': hp.choice('n_estimators', range(100, 200)),
                       'min_child_weight': hp.quniform('min_child_weight', 1, 6, 1),
                       'subsample': hp.quniform('subsample', 0.5, 1, 0.05),
                       'gamma': hp.quniform('gamma', 0.5, 1, 0.05),
                       'colsample_bytree': hp.quniform('colsample_bytree', 0.5, 1, 0.05),
                       'criterion': hp.choice('criterion', ["gini", "entropy"]),

                       }
             }
        ])

        score_key = 'score'

        # self.dfs.drop(self.colTypes['Identity'], axis=1, inplace=True)  # Dropping Identity cols
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.dfs.drop(self.y, axis=1),
                                                                                self.dfs[self.y],
                                                                                test_size=self.test_size)
        print('hi split')
        trials = Trials()

        hyperparam = space_eval(self.space,
                                fmin(self.objective_func, self.space, trials=trials, algo=tpe.suggest, max_evals=100))

        print('hi hyper')
        score = -min(trials.losses())
        for d in trials.results:
            if d['loss'] == -score:
                final_d = d

        dict_ = final_d['other_stuff']
        y_pred_test = dict_['y_pred_test']

        clf = dict_['clf']

        conf_matrix = confusion_matrix(y_pred_test, self.y_test)

        merge_df = self.x_test.copy()
        merge_df['y_test'] = list(self.y_test.copy())

        self.final_results['Hyperparameter'] = hyperparam
        self.final_results[score_key] = score
        self.final_results['Data'] = self.dfs
        self.final_results['conf_matrix'] = conf_matrix
        self.final_results['pickle_file'] = clf
        self.final_results['y_pred'] = y_pred_test
        self.final_results['metric'] = self.metric


    def return_results(self):
        # print(self.final_results)
        return self.final_results


# cl=xgbClassifier(df,"Survived",accuracy_score)
# cl.return_results()