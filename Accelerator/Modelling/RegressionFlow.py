import pandas as pd
import os, shutil, sys

sys.path.append('..')
from Accelerator.GlobalParameters import *
from Accelerator.Modelling.Algos.LinearRegressor import LinearRegressor
from Accelerator.Modelling.Algos.RandomForestsRegressor import RandomForestsRegressor
from Accelerator.Modelling.Algos.xgbRegressor import xgbRegressor
from sklearn.metrics import accuracy_score,f1_score,mean_squared_error
import json




class RegressionFlow:
    def __init__(self,df,targetName,metric_lst):
        # self.df = pd.read_csv(RunFilePath1 + '/' + self.appName + '/' + inputFileName)
        self.df=df
        self.targetName=targetName
        self.results = {}
        self.metric_lst=metric_lst

        self.metric_dict = {'accuracy_score': accuracy_score, 'f1_score': f1_score, 'mean_squared_error': mean_squared_error}
        self.execute()


    def execute(self):

        metric_list1 = []
        for i in self.metric_lst:
            metric_list1.append(self.metric_dict[i])



        for metric_ in metric_list1:
            print(metric_)

            obj=RandomForestsRegressor(self.df,self.targetName,metric_)
            self.results['randomforest'+ str(metric_)] = obj.return_results()
            obj1=LinearRegressor(self.df,self.targetName,metric_)
            self.results['linear' + str(metric_)] = obj1.return_results()
            obj2=xgbRegressor(self.df,self.targetName,metric_)
            self.results['xgbclassifier' + 'accuracy_score'] = obj2.return_results()



        l = []
        for dict_ in self.results:
            sample_= self.results[dict_]
            dict_1 = sample_['Hyperparameter']
            d1={}
            d1['model'] = dict_1['model'].__name__
            d1['score'] = sample_['score']
            d1['metric'] = sample_['metric'].__name__
            d1['conf_matrix'] = sample_['conf_matrix']
            d1['pickle_file'] = sample_['pickle_file']
            d1['y_pred_test'] = sample_['y_pred']
            l.append(d1)



        metric_list_temp = []
        for d in l:
            metric_list_temp.append(d['metric'])
        metric_list_1 = list(set(metric_list_temp))



        l_final = []
        for metric in metric_list_1:
            temp = 1000000
            l_metric = []
            for d in l:
                if d['metric'] == metric:
                    l_metric.append(d)
            for d1 in l_metric:
                if d1['score'] < temp:
                    temp = d1['score']
            l_temp = []
            for d2 in l_metric:
                if d2['score'] == temp:
                    d2['best_flag'] = 'Yes'
                else:
                    d2['best_flag'] = 'No'
                l_temp.append(d2)
            l_final.append(l_temp)
        l_final = [j for i in l_final for j in i]


        # print(l_final)
        return l_final





# if __name__ == "__main__":
#     classflow = ClassificationFlow('App2')


