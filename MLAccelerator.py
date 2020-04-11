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

        self.final_dfs=[]

        mainThread=flowThread("Main Thread", self.colIdentification, self.df, self.y)
        mainThread.start()
        mainThread.join()

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
        encodingColumnsThreads={}
        # if len(all_dfs)!=0:
        #     for i in enumerate(all_dfs):
        #         name="encodingColumns_{}".format(i[0])
        #         encodingColumnsThreads[name]=flowThread(name, "noStep",endFlag=False)
        #         encodingColumnsThreads[name].start()

    def return_all(self):
        return(self.final_dfs)


ml=MLAccelerator(df,y)
a=ml.return_all()

print(len(a))

"""

# Column Indentification
colIdentObj = ColumnTypeIdentification(df,y)
colTypes = colIdentObj.colTypes
target_type = colIdentObj.target_type

# Null handling
nullHndlngObj = NullHandling(df, colTypes, y)
df_dict = {}
for strategy in [None, 'mean', 'knn']:
    nullHndlngObj.impute(strategy)
    # df_dict['null_strategy_' + str(strategy)] = nullHndlngObj.impute(strategy)

# Feature Reduction
df_all = []
for i in df_dict.keys():
    fRdctionObj = FeatureReduction(df_dict[i],colTypes,y,target_type)
    df_all.extend(fRdctionObj.return_dfs())

# Outlier handling
df_all_oh = []
for i in df_all:
    OH = OutlierHandling(i,colTypes,y,target_type)
    df_all_oh.extend(OH.return_dfs())

# Encoding columns
df_all_en = []
for i in df_all_oh:
    en = Encoding(i,colTypes,y)
    df_all_en.extend(en.return_dfs())

# output of EDA : df_all_en
for i in df_all_en:
	print(i.head(2))"""