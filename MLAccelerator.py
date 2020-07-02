#python flow.py "data.csv" "target_col"
from time import ctime
import sys
import pandas as pd
import numpy as np
from EDA.OutlierHandling import OutlierHandling
from EDA.Encoding import Encoding
from EDA.NullHandling import NullHandling
from EDA.ColumnTypeIdentification import ColumnTypeIdentification
from EDA.ColumnTypeIdentification import ColumnTypeConfirmation
from EDA.FeatureReduction import FeatureReduction
from EDA.TargetGraphs import TargetGraphs
import warnings
warnings.filterwarnings("ignore")
import threading
from Modelling.Classification import Classification
from Modelling.Regression import Regression
from sklearn.metrics import accuracy_score,f1_score,mean_squared_error
from itertools import product
from pathlib import Path
from TextProcessing.TextProcessing import TextProcessing
import pickle
import matplotlib
matplotlib.use('Agg')



class flowThread(threading.Thread):
    def __init__(self, threadId, func, params, endFlag=True):
        threading.Thread.__init__(self)
        self.threadId=threadId
        self.func=func
        self.params=params

    def run(self):
        print('Running Thread: {}'.format(self.threadId))
        self.func(**self.params)

class MLAccelerator:
    def __init__(self, df=None, y=None, final_list=None, metric_dict=None):
        self.df=df
        self.y=y
        self.final_list = final_list
        self.metric_dict = metric_dict
        self.targetType=None
        self.results1 = {}
        self.results2 = {}
        self.results = {}
        self.colTypes = None 

    def execute(self):
        """Usage:
        [nullHandlingFlag, featureReductionFlag, outlierHandlingFlag, encodingFlag, modellingClass,
        nullHandlingMethod, featureReductionMethod, outlierHandlingMethod, encodingMethod, modellingMetric]"""
        metric_list = []
        for i in self.final_list['metric']:
            metric_list.append(self.metric_dict[i])
        allParams={#'nullHandlingFlag'       :[self.flag_list['nullhandle']],
                   #'featureReductionFlag'   :[self.flag_list['feature']],
                   #'outlierHandlingFlag'    :[self.flag_list['outlier']],
                   #'encodingFlag'           :[self.flag_list['encoding']],
                   'modellingClass'         :self.final_list['model'],
                   'nullHandlingMethod'     :self.final_list['nullhandle'],
                   'featureReductionMethod' :self.final_list['feature'],
                   'outlierHandlingMethod'  :self.final_list['outlier'],
                   'encodingMethod'         :self.final_list['encoding'],
                   'textProcessing'         :self.final_list['vector'],
                   'modellingMetric'        :metric_list}

        keys = allParams.keys()
        values = (allParams[key] for key in keys)
        #print('keys:{}'.format(keys))
        #print('values:{}'.format(values))
        possibilities = [dict(zip(keys, combination)) for combination in product(*values)]
        threads=[]

        for combNo, combParam in enumerate(possibilities):
            combParam['threadId'] = combNo
            th=flowThread(combNo, self.acceleratorExecution, combParam)
            th.start()
            threads.append(th)
            # self.results.append(self.acceleratorExecution(**combParam))

        for th in threads:
            th.join()

        #print(self.results)
        #bestResult= self.bestModel(self.results)
        #final_log = bestResult.pop('log')+str(bestResult)
        #self.logData(None, 'Result', 'Result', final_log, path='Output', flush=True)
        #return final_log

        j = len(self.results2)
        for i in range(len(self.results2)):
            self.results2[i + j] = self.results2.pop(i)

        self.results1.update(self.results2)

        self.results = self.results1

        result_final = []
        for key in self.results:
            result_final.append(self.results[key])

        if self.final_list['model']==['classification']:
            l = []
            for dict_ in result_final:
                sample_ = dict_
                d1 = {}
                dict_1 = sample_['Hyperparameter']
                d1['model'] = dict_1['model'].__name__
                d1['score'] = sample_['score']
                d1['log'] = sample_['log']
                d1['metric'] = sample_['metric']
                d1['conf_matrix'] = sample_['conf_matrix']
                d1['pickle_file'] = sample_['pickle_file']
                d1['y_pred_test'] = sample_['y_pred']
                l.append(d1)


            l_modified = []
            for d in l:
                conf_matrix = d['conf_matrix']
                headers = self.df[self.y].tolist()
                print(type(headers))
                cm_df = pd.DataFrame(conf_matrix, columns=set(headers))
                cm_df['index'] = set(headers)
                cm_df.set_index('index', inplace=True)
                #del cm_df.index.name
                d['conf_matrix'] = cm_df
                l_modified.append(d)

            metric_list_temp = []
            for d in l_modified:
                metric_list_temp.append(d['metric'])
            metric_list_1 = list(set(metric_list_temp))

            l_final = []
            for metric in metric_list_1:
                temp = 0
                l_metric = []
                for d in l:
                    if d['metric'] == metric:
                        l_metric.append(d)
                for d1 in l_metric:
                    if d1['score'] > temp:
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

            for i in range(len(l_final)):
                dict_ = l_final[i]
                pkl = dict_['pickle_file']
                y_pred_test = dict_['y_pred_test']
                pkl_filename = 'static/' + str(i) + '.pkl'
                with open(pkl_filename, 'wb') as file:
                    pickle.dump(pkl, file)
                y_pred_test = pd.DataFrame(y_pred_test)
                file_name = 'static/' + str(i) + '.csv'
                y_pred_test.to_csv(file_name)

            l_final_1 = []
            for i in range(len(l_final)):
                dict_ = l_final[i]
                dict_.pop('pickle_file')
                dict_.pop('y_pred_test')
                l_final_1.append(dict_)
            return l_final_1

        elif self.final_list['model'] == ['regression']:
            print("Hi regression")

            l = []
            for dict_ in result_final:
                sample_ = dict_
                d1 = {}
                dict_1 = sample_['Hyperparameter']
                d1['model'] = dict_1['model'].__name__
                d1['score'] = sample_['score']
                d1['log'] = sample_['log']
                d1['metric'] = sample_['metric']
                d1['pickle_file'] = sample_['pickle_file']
                d1['y_pred_test'] = sample_['y_pred']
                d1['residual'] = sample_['residual']

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

            for i in range(len(l_final)):
                dict_ = l_final[i]
                pkl = dict_['pickle_file']
                y_pred_test = dict_['y_pred_test']
                residual=dict_['residual']
                pkl_filename = 'static/' + str(i) + '.pkl'
                with open(pkl_filename, 'wb') as file:
                    pickle.dump(pkl, file)
                y_pred_test = pd.DataFrame(y_pred_test)
                file_name = 'static/' + str(i) + '.csv'
                y_pred_test.to_csv(file_name)
                residual.savefig('static/'+str(i)+'residual.png')

            l_final_1 = []
            for i in range(len(l_final)):
                dict_ = l_final[i]
                dict_.pop('pickle_file')
                dict_.pop('y_pred_test')
                dict_.pop('residual')
                l_final_1.append(dict_)
            return l_final_1

    def acceleratorExecution(self, **kwargs):
                             # nullHandlingFlag, featureReductionFlag, outlierHandlingFlag, encodingFlag, modellingClass,
                             # nullHandlingMethod, featureReductionMethod, outlierHandlingMethod, encodingMethod, modellingMetric):

        loggingSteps=''
        df=self.df.copy()
        # self.colIdentification(df, self.y)
        # loggingSteps = loggingSteps
        # self.logData(df, 'colIdentification', kwargs['threadId'], loggingSteps, flush=True)

        if kwargs['nullHandlingMethod']:
            df=self.nullHandlingStep(df, self.y, kwargs['nullHandlingMethod'])
            loggingSteps = loggingSteps + 'Null Handling with {},'.format(str(kwargs['nullHandlingMethod']))
            self.logData(df, 'nullHandling', kwargs['threadId'], loggingSteps)

        if kwargs['featureReductionMethod']:
            df=self.featureReductionStep(df, self.colTypes, self.y, self.targetType, kwargs['featureReductionMethod'])
            loggingSteps = loggingSteps+ '  Feature Reduction with {},'.format(str(kwargs['featureReductionMethod']))
            self.logData(df, 'Feature Reduction', kwargs['threadId'], loggingSteps)

        if kwargs['outlierHandlingMethod']:
            df=self.outlierHandlingStep(df, kwargs['outlierHandlingMethod'])
            loggingSteps = loggingSteps+ '  Outlier Handling with {},'.format(str(kwargs['outlierHandlingMethod']))
            self.logData(df, 'Outlier Handling', kwargs['threadId'], loggingSteps)

        if kwargs['encodingMethod']:
            df=self.encodingColumnsStep(df, kwargs['encodingMethod'])
            loggingSteps = loggingSteps+ '  Encoding with {},'.format(str(kwargs['encodingMethod']))
            self.logData(df, 'Encoding', kwargs['threadId'], loggingSteps)

        if kwargs['textProcessing']:
            df = self.textProcessingStep(df, kwargs['textProcessing'], self.final_list['textp'])
            loggingSteps = loggingSteps + '  Text Processing with {},'.format(str(kwargs['textProcessing']))
            self.logData(df, 'textProcessing', kwargs['threadId'], loggingSteps)

        if kwargs['modellingClass']=='classification':
            loggingSteps = loggingSteps+ 'Building Classification Model\n'
            result1,result2= self.classificationStep(df, self.y, self.colTypes, kwargs['modellingMetric'])
            #print(result1)
            data=result1.pop('Data')
            self.logData(data, 'Modelling {}'.format(kwargs['modellingClass']), kwargs['threadId'], loggingSteps+'\n'+str(result1))
            result1['log']=loggingSteps
            result1['metric'] = kwargs['modellingMetric'].__name__
            self.results1[kwargs['threadId']]=result1
            data=result2.pop('Data')
            self.logData(data, 'Modelling {}'.format(kwargs['modellingClass']), kwargs['threadId'], loggingSteps+'\n'+str(result2))
            result2['log']=loggingSteps
            result2['metric'] = kwargs['modellingMetric'].__name__
            self.results2[kwargs['threadId']]=result2



        if kwargs['modellingClass'] == 'regression':
            loggingSteps = loggingSteps + 'Building Regression Model\n'
            result1,result2= self.regressionStep(df, self.y, self.colTypes, kwargs['modellingMetric'])
            #print(result1)
            data=result1.pop('Data')
            self.logData(data, 'Modelling {}'.format(kwargs['modellingClass']), kwargs['threadId'], loggingSteps+'\n'+str(result1))
            result1['log']=loggingSteps
            result1['metric'] = kwargs['modellingMetric'].__name__
            self.results1[kwargs['threadId']]=result1
            data=result2.pop('Data')
            self.logData(data, 'Modelling {}'.format(kwargs['modellingClass']), kwargs['threadId'], loggingSteps+'\n'+str(result2))
            result2['log']=loggingSteps
            result2['metric'] = kwargs['modellingMetric'].__name__
            self.results2[kwargs['threadId']]=result2


    def bestModel(self, results):
        best_model = 0
        for key in results.keys():
            if results[best_model]['score'] < results[key]['score']:
                best_model = key
        return results[best_model]

    def logData(self, df, stepName, threadId, logSteps, path='intermediateFiles', flush= False):
        output_dir = Path(path+'/thread_{}'.format(threadId))
        logSteps="At Step: {} \nSequence executed:\n{}\n\n".format(stepName, logSteps)
        # if flush:
        #     rmtree(path)
        output_dir.mkdir(parents=True, exist_ok=True)
        if flush:
            log_file = open('{}/log.txt'.format(output_dir), "w+")
        else:
            log_file = open('{}/log.txt'.format(output_dir), "a")
        if df is not None:
            df.to_csv('{}/{}.csv'.format(output_dir, stepName), index=False)
        log_file.write(logSteps)
        log_file.close()


    def colIdentification(self):
        colIdentObj = ColumnTypeIdentification(self.df)
        self.colTypes = colIdentObj.colTypes
        # self.targetType = None
        self.dtypes = colIdentObj.dtypes

    def colConfirmation(self):
        colconfirm = ColumnTypeConfirmation(self.df,self.dtypes)
        #self.colTypes = colIdentObj.colTypes
        # self.targetType = None
        self.colTypes = colconfirm.colTypes
        #self.dtypes = colIdentObj.dtypes

    def TargetGraphs(self):
        obj_target = TargetGraphs(self.df, self.colTypes, self.y, self.targetType)
        col = obj_target.top_features
        return col

    def nullHandlingStep(self, df, y, strategy):
        nullHndlngObj = NullHandling(df, self.colTypes, y)
        return nullHndlngObj.impute(strategy)

    def featureReductionStep(self, df, colTypes, y, targetType, method):
        fRdctionObj = FeatureReduction(df, colTypes, y, targetType, method)
        return fRdctionObj.return_result()

    def outlierHandlingStep(self, df, method):
        OH = OutlierHandling(df, self.colTypes, self.y, self.targetType, method)
        return OH.return_result()

    def encodingColumnsStep(self, df, method):
        en = Encoding(df, self.colTypes, self.y, method)
        return en.return_result()

    def textProcessingStep(self, df, method, processing_steps):
        print("text processing begins")
        print(ctime())
        txtHandling=TextProcessing(df, self.colTypes, method, processing_steps)
        print(ctime())
        return txtHandling.return_result()

    def classificationStep(self, df, y, colTypes, metric):
        classification=Classification(df, y, colTypes, metric)
        return classification.return_results()

    def regressionStep(self, df, y, colTypes, metric):
        regression=Regression(df, y, colTypes, metric)
        return regression.return_results()

    def mean_absolute_percentage_error(y_true, y_pred):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

