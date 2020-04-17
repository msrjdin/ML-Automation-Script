import copy
import numpy as np


class FeatureReduction:

    def __init__(self,df,colTypes,y,target_type):
        self.df=df
        self.colTypes=copy.deepcopy(colTypes)
        self.y=y
        self.target_type=target_type
        self.colTypes[self.target_type].remove(self.y)
        self.all_dfs=[]
        self.pearson_corr()

    #removing the continuous column with a correlation of above 0.8
    def pearson_corr(self):
        self.colTypes['Numeric']=set(self.colTypes['Numeric']).intersection(set(self.df.columns))
        corr=self.df[list(self.colTypes['Numeric'])].corr(method="pearson").abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(np.bool))
        to_drop = [column for column in upper.columns if any(upper[column] > 0.80)]
        self.df.drop(to_drop, axis=1,inplace=True)
        self.colTypes['Numeric'] = [x for x in self.colTypes['Numeric'] if x not in to_drop]
        self.all_dfs.append(self.df)

    #returns the dict of dataframes updated in this class
    def return_dfs(self):
        return self.all_dfs
