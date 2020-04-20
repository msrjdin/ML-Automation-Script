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
from sklearn.metrics import accuracy_score,f1_score
from itertools import product
from pathlib import Path

#Reading command line arguments into data and target
# if __name__ == "__main__":
#     if len(sys.argv) >= 3 :
#         data = sys.argv[1]
#         y = sys.argv[2]
#     else :
#         print("input the file name and target column")
#         exit()
data='train.csv'
y='Survived'

df = pd.read_csv(data)


class flowThread(threading.Thread):
    def __init__(self, threadName, nextStep, *args, endFlag=True):
        threading.Thread.__init__(self)
        self.threadName=threadName
        self.nextStep=nextStep
        self.args=args
        self.endFlag=endFlag

    def run(self):
        print('Running: {}'.format(self.threadName))
        if self.endFlag:
            self.nextStep(self.args)


class MLAccelerator:
    def __init__(self, df, y):
        self.df=df
        self.y=y

        self.execute()

    def execute(self):
        results=[]

        #Usage
        # nullHandlingFlag, featureReductionFlag, outlierHandlingFlag, encodingFlag, modellingClass,
        # nullHandlingMethod, featureReductionMethod, outlierHandlingMethod, encodingMethod, modellingMetric
        allParams={'nullHandlingFlag':      [True],
                   'featureReductionFlag':  [True],
                   'outlierHandlingFlag':   [True],
                   'encodingFlag':          [True],
                   'modellingClass':        ['classification'],
                   'nullHandlingMethod'   : ['knn', None],
                   'featureReductionMethod':['pearson'],
                   'outlierHandlingMethod': ['capping'],
                   'encodingMethod':        ['one-hot'],
                   'modellingMetric':       [accuracy_score]}

        keys = allParams.keys()
        values = (allParams[key] for key in keys)
        possibilities= [dict(zip(keys, combination)) for combination in product(*values)]

        for combNo, combParam in enumerate(possibilities):
            combParam['threadId']=combNo
            results.append(self.acceleratorExecution(**combParam))

        return self.bestModel(results)


    def acceleratorExecution(self, **kwargs):
                             # nullHandlingFlag, featureReductionFlag, outlierHandlingFlag, encodingFlag, modellingClass,
                             # nullHandlingMethod, featureReductionMethod, outlierHandlingMethod, encodingMethod, modellingMetric):
        # print(kwargs)
        loggingSteps=''
        df=self.df.copy()
        self.colIdentification(df, self.y)
        loggingSteps+='colIdentification\n'
        self.logData(df, 'colIdentification', kwargs['threadId'], loggingSteps)

        if kwargs['nullHandlingFlag']:
            df=self.nullHandlingStep(df, self.y, kwargs['nullHandlingMethod'])
            loggingSteps += 'nullHandling with method {}\n'.format(str(kwargs['nullHandlingMethod']))
            self.logData(df, 'nullHandling', kwargs['threadId'], loggingSteps)

        if kwargs['featureReductionFlag']:
            df=self.featureReductionStep(df, self.colTypes, self.y, self.targetType, kwargs['featureReductionMethod'])
            loggingSteps += 'featureReduction with method {}\n'.format(str(kwargs['featureReductionMethod']))
            self.logData(df, 'Feature Reduction', kwargs['threadId'], loggingSteps)

        if kwargs['outlierHandlingFlag']:
            df=self.outlierHandlingStep(df, kwargs['outlierHandlingMethod'])
            loggingSteps += 'outlierHandling with method {}\n'.format(str(kwargs['outlierHandlingMethod']))
            self.logData(df, 'Outlier Handling', kwargs['threadId'], loggingSteps)

        if kwargs['encodingFlag']:
            df=self.encodingColumnsStep(df, kwargs['encodingMethod'])
            loggingSteps += 'encoding with method {}\n'.format(str(kwargs['encodingMethod']))
            self.logData(df, 'Encoding', kwargs['threadId'], loggingSteps)

        if kwargs['modellingClass']=='classification':
            loggingSteps += 'Building Classification Model'
            # result=pd.DataFrame(self.classificationStep(df, self.y, self.colTypes, kwargs['modellingMetric']))
            # self.logData(result, 'Modelling_{}'.format(kwargs['modellingClass']), kwargs['threadId'], loggingSteps)
            return [self.classificationStep(df, self.y, self.colTypes, kwargs['modellingMetric']), loggingSteps]

    def bestModel(self, results):
        best_model=results[0]
        for i in results[1:]:
            if best_model[0]['score']<i[0]['score']:
                best_model=i

        return best_model

    def logData(self, df, stepName, threadId, logSteps):
        output_dir = Path('intermediateFiles/thread_{}'.format(threadId))
        output_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv('intermediateFiles/thread_{}/{}.csv'.format(threadId, stepName), index=False)


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

    def classificationStep(self, df, y, colTypes, metric):
        classification=Classification(df, y, colTypes, metric)
        return classification.return_results()



ml=MLAccelerator(df,y)
a=ml.execute()

print(a)
