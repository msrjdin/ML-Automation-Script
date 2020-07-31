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
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')
from sklearn.metrics import (confusion_matrix, precision_recall_curve, auc,
                             roc_curve, recall_score, classification_report, f1_score,
                             precision_recall_fscore_support)
import seaborn as sns


# df1 = pd.read_csv(r"C:\Users\SatyaSindhuMolleti\Desktop\ML-Automation-Script-master\ML-Automation-Script\Accelerator\dataframe.csv")
# df=df1[['Survived','Pclass','Fare']]

class RandomForestsRegressor:
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

        if args['model'] == RandomForestRegressor:
            n_estimators = args['param']['n_estimators']
            max_depth = args['param']['max_depth']
            max_features = args['param']['max_features']
            min_sample_split = args['param']['min_sample_split']
            criterion = args['param']['criterion']
            clf = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, max_features=max_features,
                                        criterion=criterion, min_sample_split=min_sample_split
                                        )

        clf.fit(self.x_train, self.y_train)
        y_pred_train = clf.predict(self.x_train)
        y_pred_test = clf.predict(self.x_test)
        mae = metrics.mean_absolute_error(self.y_train, y_pred_train)
        mse = metrics.mean_squared_error(self.y_train, y_pred_train)
        rmse = np.sqrt(mse)
        score = metrics.mean_squared_error(y_pred_test, self.y_test)
        return {'loss': score, 'status': STATUS_OK, 'other_stuff': {'y_pred_test': y_pred_test, 'clf': clf}}



    def execute(self):
        # using Hyperopt for parameter tuning
        self.space = hp.choice('regression', [
            {'model': RandomForestRegressor,
             'param': {'max_depth': hp.choice('max_depth', range(1, 30)),
                       'min_sample_split': hp.choice('min_sample_split', range(1, 40)),
                       'max_features': hp.choice('max_features', range(1, self.max_feat)),
                       'n_estimators': hp.choice('n_estimators', range(50, 400)),
                       'criterion': hp.choice('criterion', ["gini", "entropy"])
                       }
             }
        ])


        score_key = 'score'

        # self.dfs.drop(self.colTypes['Identity'], axis=1, inplace=True)  # Dropping Identity cols
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.dfs.drop(self.y, axis=1),
                                                                                self.dfs[self.y],
                                                                                test_size=self.test_size)

        trials = Trials()

        hyperparam = space_eval(self.space,
                                fmin(self.objective_func, self.space, trials=trials, algo=tpe.suggest, max_evals=100))
        score = min(trials.losses())
        for d in trials.results:
            if d['loss'] == score:
                final_d = d

        dict_ = final_d['other_stuff']
        y_pred_test = dict_['y_pred_test']

        clf = dict_['clf']

        merge_df = self.x_test.copy()
        merge_df['y_test'] = list(self.y_test.copy())

        plt.clf()
        sns.residplot(y_pred_test, self.y_test)
        # fig = sm.get_figure()
        plt.show()
        fig = plt
        # fig=plt.savefig('C:/Users/SindhuKarnati/Desktop/MLAccelarator/residual_file/out3.png')

        self.final_results['Hyperparameter'] = hyperparam
        self.final_results[score_key] = score
        self.final_results['Data'] = self.dfs
        self.final_results['residual'] = fig
        self.final_results['pickle_file'] = clf
        self.final_results['y_pred'] = y_pred_test


    def return_results(self):
        return self.final_results


# cl=RandomForestsRegressor(df,"Survived",accuracy_score)
# cl.return_results()