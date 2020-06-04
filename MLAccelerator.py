#python flow.py "data.csv" "target_col"

import sys
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
from Modelling.Regression import Regression
from sklearn.metrics import accuracy_score,f1_score
from itertools import product
from pathlib import Path
from shutil import rmtree
from TextProcessing import TextProcessing

#Reading command line arguments into data and target
# if __name__ == "__main__":
#     if len(sys.argv) >= 3 :
#         data = sys.argv[1]
#         y = sys.argv[2]
#     else :
#         print("input the file name and target column")
#         exit()
#data='train.csv'
#y='Survived'

#df = pd.read_csv(data)






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
    def __init__(self, df, y,final_list,metric_dict):
        self.df=df
        self.y=y
        self.final_list = final_list
        self.metric_dict = metric_dict
        self.results = {}


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
                   'modellingMetric'        :metric_list}

        keys = allParams.keys()
        values = (allParams[key] for key in keys)
        #print('keys:{}'.format(keys))
        #print('values:{}'.format(values))
        possibilities = [dict(zip(keys, combination)) for combination in product(*values)]

        allParams={'modellingClass'         :['classification'],
                   'nullHandlingMethod'     :['knn', None],
                   'featureReductionMethod' :[None],
                   'outlierHandlingMethod'  :['capping'],
                   'encodingMethod'         :['one-hot'],
                   'textProcessing'         :['BOW', 'Tfidf'],
                   'modellingMetric'        :[accuracy_score]}

        keys = allParams.keys()
        values = (allParams[key] for key in keys)
        print('Keys: {}'.format(keys))
        print('Values: {}'.format(values))
        possibilities = [dict(zip(keys, combination)) for combination in product(*values)]
        print(possibilities)

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
        result_final = []
        for key in self.results:
            result_final.append(self.results[key])

        l = []
        for dict_ in result_final:
            sample_ = dict_
            d1 = {}
            dict_1 = sample_['Hyperparameter']
            d1['model'] = dict_1['model'].__name__
            d1['score'] = sample_['score']
            d1['log'] = sample_['log']
            d1['metric'] = sample_['metric']
            l.append(d1)
        print(l)


        metric_list_temp = []
        for d in l:
            metric_list_temp.append(d['metric'])
        metric_list_1 = list(set(metric_list_temp))
        print(metric_list_1)


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


        return l_final

    def acceleratorExecution(self, **kwargs):
                             # nullHandlingFlag, featureReductionFlag, outlierHandlingFlag, encodingFlag, modellingClass,
                             # nullHandlingMethod, featureReductionMethod, outlierHandlingMethod, encodingMethod, modellingMetric):
        # print(kwargs)
        loggingSteps=''
        df=self.df.copy()
        self.colIdentification(df, self.y)
        loggingSteps = loggingSteps
        self.logData(df, 'colIdentification', kwargs['threadId'], loggingSteps, flush=True)

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

        if len(self.colTypes['Text'])>0:
            df=self.textProcessingStep(df, kwargs['textProcessing'])
            loggingSteps = loggingSteps + '  txtProcessing with {},'.format(str(kwargs['textProcessing']))
            self.logData(df, 'textProcessing', kwargs['threadId'], loggingSteps)


        if kwargs['modellingClass']=='classification':
            loggingSteps = loggingSteps+ '  Building Classification Model'
            result= self.classificationStep(df, self.y, self.colTypes, kwargs['modellingMetric'])
            data=result.pop('Data')
            self.logData(data, 'Modelling {}'.format(kwargs['modellingClass']), kwargs['threadId'], loggingSteps+','+str(result))
            result['metric'] = kwargs['modellingMetric'].__name__
            result['log']=loggingSteps
            self.results[kwargs['threadId']]=result

        elif kwargs['modellingClass'] == 'regression':
            loggingSteps = loggingSteps + 'Building Regression Model\n'
            result = self.regressionStep(df, self.y, self.colTypes, kwargs['modellingMetric'])
            data = result.pop('Data')
            self.logData(data, 'Modelling {}'.format(kwargs['modellingClass']), kwargs['threadId'],
                            loggingSteps + '\n' + str(result))
            result['log'] = loggingSteps
            self.results[kwargs['threadId']] = result


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


    def colIdentification(self, df, y):
        colIdentObj = ColumnTypeIdentification(df, y)
        self.colTypes = colIdentObj.colTypes
        self.targetType = colIdentObj.target_type


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

    def textProcessingStep(self, df, method):
        txtHandling-TextProcessing(df, self.colTypes, method)
        return txtHandling.return_result()

    def classificationStep(self, df, y, colTypes, metric):
        classification=Classification(df, y, colTypes, metric)
        return classification.return_results()

    def regressionStep(self, df, y, colTypes, metric):
        regression=Regression(df, y, colTypes, metric)
        return regression.return_results()



#ml=MLAccelerator(df,y)
#a=ml.execute()

# print(a)
