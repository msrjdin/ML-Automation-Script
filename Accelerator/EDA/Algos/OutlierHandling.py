
import copy
import pandas as pd
import numpy as np


# df = pd.read_csv(r"C:\Users\SatyaSindhuMolleti\Downloads\sample_train.csv")
# col_map={'capping':['Fare','Age'],'zscore':['PassengerId']}


class OutlierHandling:
    def __init__(self, df, col_map):
        self.df = df.copy()
        self.col_map=col_map
        self.df_final=df.copy(deep=True)
        self.col_methods=[]
        for key,value in col_map.items():
            self.col_methods.extend(key)
            self.df_final.drop(columns=value, inplace=True, axis=1)
            if key == 'capping':
                self.df_final=pd.concat([self.df_final,self.capping_outlier(self.df[value])],axis=1)
            elif key == 'removing':
                self.df_final = pd.concat([self.df_final, self.remove_outlier(self.df[value])], axis=1)
            elif key == 'zscore':
                self.df_final = pd.concat([self.df_final, self.zscore_outlier(self.df[value])], axis=1)


    # capping the values to the predefined lower and upper percentiles
    def capping_outlier(self,df,lowerperc=0.01, higherperc=0.99):
        df_out = df.copy(deep=True)
        for col in df_out.columns:
            percentiles = df_out[col].quantile([lowerperc, higherperc]).values
            df_out[col][df_out[col] <= percentiles[0]] = percentiles[0]
            df_out[col][df_out[col] >= percentiles[1]] = percentiles[1]
            # print(df_out[col])
        return df_out


    # removing the values which are not within iqr range
    def remove_outlier(self, df):
        df_out = df.copy(deep=True)
        for col_name in df_out.columns:
            q1 = df_out[col_name].quantile(0.25)
            q3 = df_out[col_name].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            df_out = df_out.loc[(df_out[col_name] > lower) & (df_out[col_name] < upper)]
        return df_out

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
        return df_out



    def return_result(self):
        # print(self.df_final)
        return self.df_final


# oh=OutlierHandling(df,col_map)
# oh.return_result()