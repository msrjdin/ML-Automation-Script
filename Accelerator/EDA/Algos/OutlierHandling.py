
import copy
import pandas as pd
import numpy as np


df = pd.read_csv(r"C:\Users\SindhuKarnati\Desktop\MLAccelarator\train.csv")
# df=df.head()
col_map={'capping':['Fare','Age'],'removing':['PassengerId']}

class OutlierHandling:
    def __init__(self,df,col_map):
        self.df = df.copy()
        self.col_map=col_map
        self.df_final=pd.DataFrame(index=self.df.index)
        self.col_methods=[]
        for key,value in col_map.items():
            self.col_methods.extend(key)
            if key == 'capping':
                pd.concat([self.df_final,self.capping_outlier(df[value])],axis=1)
            elif key == 'removing':
                pd.concat([self.df_final,self.remove_outlier(df[value])],axis=1)
            elif key == 'zscore':
                pd.concat([self.df_final,self.zscore_outlier(df[value])],axis=1)


    # capping the values to the predefined lower and upper percentiles
    def capping_outlier(self,df,lowerperc=0.01, higherperc=0.99):
        df_out = df.copy()
        for col in df.columns:
            percentiles = df[col].quantile([lowerperc, higherperc]).values
            df[col][df[col] <= percentiles[0]] = percentiles[0]
            df[col][df[col] >= percentiles[1]] = percentiles[1]
            df_out = df
        self.return_df_capping = df_out


    # removing the values which are not within iqr range
    def remove_outlier(self, df):
        df_out = df.copy()
        for col_name in df_out.columns:
            q1 = df_out[col_name].quantile(0.25)
            q3 = df_out[col_name].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            df_out = df_out.loc[(df_out[col_name] > lower) & (df_out[col_name] < upper)]
        self.return_df_removing = df_out

    # removing values which are less than predefined zscore value
    def zscore_outlier(self, df, threshold=3):
        l = []
        for i in df.columns:
            temp = []
            mean_1 = np.mean(df[i])
            std_1 = np.std(df[i])
            for y in df[i]:
                z_score = (y - mean_1) / std_1
                if np.abs(z_score) < threshold:
                    temp.append(y)
            df_temp = pd.DataFrame(temp)
            l.append(df_temp)
        df_out = pd.concat(l, axis=1, join='inner')
        df_out.columns = df.columns
        self.return_df_zscore = df_out



    def return_result(self):
        print(self.df_final)
        return self.df_final


oh=OutlierHandling(df,col_map)