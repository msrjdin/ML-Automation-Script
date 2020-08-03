
import copy
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from scipy.stats import chi2_contingency



# df = pd.read_csv(r"C:\Users\SatyaSindhuMolleti\Downloads\sample_train.csv")
# df=df[['PassengerId','Survived']]
# colTypes={'PassengerId': 'Numeric', 'Survived': 'Categorical', 'Pclass': 'Categorical', 'Name': 'Text', 'Sex': 'Categorical', 'Age': 'Numeric', 'SibSp': 'Categorical', 'Parch': 'Categorical', 'Ticket': 'Categorical', 'Fare': 'Numeric', 'Cabin': 'Categorical', 'Embarked': 'Categorical'}
# df pid,pi_srt,target : true
# entire df :false


class Correlations:
    def __init__(self, df,colTypes,targetCol,with_target):
        self.df = df.copy()
        self.colTypes=colTypes
        self.targetCol=targetCol
        numeric_cols=[]
        cat_cols=[]
        for col,type in self.colTypes.items():
            if type=='Numeric':
                numeric_cols.append(col)
            elif type=='Categorical':
                cat_cols.append(col)

        if with_target==True:
            self.col=self.pearson_correlation_with_target()
        else:
            if len(numeric_cols)>0 :
                self.df=self.pearson_correlation_with_cols(numeric_cols)
            if len(cat_cols) > 0:
                self.df=self.cramersv_correlation(cat_cols)

        # self.return_result()

    def pearson_correlation_with_target(self):
        corr=self.df.drop(self.targetCol, axis=1).apply(lambda x: x.corr(self.df[self.targetCol]))
        return corr.idxmax(abs(corr))


    def pearson_correlation_with_cols(self,numeric_cols):
        corr = self.df[numeric_cols].corr(method="pearson").abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(np.bool))
        to_drop = [column for column in upper.columns if any(upper[column] > 0.80)]
        self.df.drop(to_drop, axis=1, inplace=True)
        return self.df

    def cramersv_correlation(self,cat_cols):
        rows = []
        for var1 in cat_cols:
            col = []
            for var2 in cat_cols:
                cramers = self.cramers_V(self.df[var1], self.df[var2])  # Cramer's V test
                col.append(round(cramers, 2))
                # if round(cramers, 2) > 0.2 and var1 != var2 and len(col) <= len(df.columns):
                #     print(var1 + var2)
            rows.append(col)

        cramers_results = np.array(rows)
        corr = pd.DataFrame(cramers_results, columns=cat_cols, index=cat_cols)
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(np.bool))
        to_drop = [column for column in upper.columns if any(upper[column] > 0.80)]
        self.df.drop(to_drop, axis=1, inplace=True)
        return self.df

    def cramers_V(self,var1, var2):
        crosstab = np.array(pd.crosstab(var1, var2, rownames=None, colnames=None))  # Cross table building
        stat = chi2_contingency(crosstab)[0]  # Keeping of the test statistic of the Chi2 test
        obs = np.sum(crosstab)  # Number of observations
        mini = min(crosstab.shape) - 1  # Take the minimum value between the columns and the rows of the cross table
        return (stat / (obs * mini))

    #
    def return_result(self):
        # print(self.df,self.col)
        return (self.df,self.col)



# oh=Correlations(df,colTypes)
# oh.return_result()
