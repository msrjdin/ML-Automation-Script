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
        self.pathFollowed={}

        self.final_dfs=[]

        edaThread=flowThread("EDA Thread", self.colIdentification, self.df, self.y)
        edaThread.start()

        edaThread.join()
        self.modellingStep()

    def colIdentification(self, args):
        colIdentObj = ColumnTypeIdentification(args[0], args[1])
        self.colTypes = colIdentObj.colTypes
        self.targetType = colIdentObj.target_type

        self.nullhandlingStep()

    def nullhandlingStep(self):
        nullHndlngObj = NullHandling(self.df, self.colTypes, self.y)

        nullHandlingThreads={}
        for strategy in [None, 'mean', 'knn']:
            dfStrategy=nullHndlngObj.impute(strategy)
            name="nullHandlingThread{}".format(str(strategy))
            nullHandlingThreads[name]=flowThread(name, self.featureReductionStep, dfStrategy)
            nullHandlingThreads[name].start()

        for nme in nullHandlingThreads.keys():
            nullHandlingThreads[nme].join()

    def featureReductionStep(self, args):
        fRdctionObj = FeatureReduction(args[0], self.colTypes, self.y, self.targetType)

        featureReductionThreads={}
        all_dfs=fRdctionObj.return_dfs()
        if len(all_dfs)!=0:
            for i in enumerate(all_dfs):
                name="featureReduction_{}".format(i[0])
                featureReductionThreads[name]=flowThread(name, self.outlierHandlingStep, i[1])
                featureReductionThreads[name].start()

        for nme in featureReductionThreads.keys():
            featureReductionThreads[nme].join()


    def outlierHandlingStep(self, args):
        OH = OutlierHandling(args[0], self.colTypes, self.y, self.targetType)
        all_dfs=OH.return_dfs()
        outlierHandlingThreads={}
        if len(all_dfs)!=0:
            for i in enumerate(all_dfs):
                name="outlierHandling_{}".format(i[0])
                outlierHandlingThreads[name]=flowThread(name, self.encodingColumnsStep, i[1])
                outlierHandlingThreads[name].start()

        for nme in outlierHandlingThreads.keys():
            outlierHandlingThreads[nme].join()


    def encodingColumnsStep(self, args):
        en = Encoding(args[0], self.colTypes, self.y)
        all_dfs=en.return_dfs()

        self.final_dfs.extend(all_dfs)

    def modellingStep(self):

        classification=Classification(self.final_dfs, self.y, accuracy_score, self.colTypes)
        self.results=classification.return_results()

    def return_results(self):
        return(self.results)


ml=MLAccelerator(df,y)
a=ml.return_results()

# print(ml.colTypes)
# print(len(a))
print(a)
