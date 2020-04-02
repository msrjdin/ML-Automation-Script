#python flow.py "data.csv" "target_col"

import sys
import pandas as pd
from EDA.OutlierHandling import *
from EDA.NullHandling import *
from EDA.ColumnTypeIdentification import *
from EDA.Encoding import *
from EDA.FeatureReduction import *


#Reading command line arguments into data and target   
if __name__ == "__main__":
    if len(sys.argv) >= 3 :
        data = sys.argv[1]
        y = sys.argv[2]
    else :
        print("input the file name and target column")
        exit();

df = pd.read_csv(data)

# Column Indentification
colIdentObj = ColumnTypeIdentification(df,y)
colTypes = colIdentObj.colTypes
target_type = colIdentObj.target_type

# Null handling
nullHndlngObj = NullHandling(df,colTypes,y)
df_dict = {}
for strategy in [None, 'mean', 'knn']:
    df_dict['null_strategy_' + str(strategy)] = nullHndlngObj.impute(strategy)

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
	print(i.head(2))