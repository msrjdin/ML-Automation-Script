import copy
import pandas as pd
import numpy as np


class OutlierHandling:
    def __init__(self, df,colTypes,y,target_type, method):
        self.df = df.copy()
        self.y=y
        self.target_type=target_type
        self.colTypes = copy.deepcopy(colTypes)
        self.colTypes[self.target_type].remove(self.y)

        self.colTypes['Numeric'] = set(colTypes['Numeric']).intersection(set(df.columns))

        # self.all_dfs = []
        if method=='capping':
            self.capping_outlier(self.df)
        elif method=='removing':
            self.remove_outlier(self.df)
        elif method=='zscore':
            self.zscore_outlier(self.df)

    # capping the values to the predefined lower and upper percentiles
    def capping_outlier(self, df, lowerperc=0.01, higherperc=0.99):
        df_out = df.copy()
        df = df[list(self.colTypes['Numeric'])].copy()
        for col in df.columns:
            percentiles = df[col].quantile([lowerperc, higherperc]).values
            df[col][df[col] <= percentiles[0]] = percentiles[0]
            df[col][df[col] >= percentiles[1]] = percentiles[1]
            df_out[list(self.colTypes['Numeric'])] = df
        self.return_df=df_out


    # removing the values which are not within iqr range
    def remove_outlier(self, df):
        df_out = df[list(self.colTypes['Numeric'])].copy()
        df_in = df.copy(deep=True)
        df_in.drop(list(self.colTypes['Numeric']), axis=1, inplace=True)
        for col_name in df_out.columns:
            q1 = df_out[col_name].quantile(0.25)
            q3 = df_out[col_name].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            df_out = df_out.loc[(df_out[col_name] > lower) & (df_out[col_name] < upper)]
        df_final = pd.concat([df_in, df_out], axis=1, join='inner')
        self.return_df = df_final

    # removing values which are less than predefined zscore value
    def zscore_outlier(self, df, threshold=3):
        l = []
        df1 = df.copy(deep=True)
        df = df[list(self.colTypes['Numeric'])].copy(deep=True)
        df1.drop(self.colTypes['Numeric'], axis=1, inplace=True)

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
        df_final = pd.concat([df1, df_out], axis=1, join='inner')
        self.return_df=df_final

    def return_result(self):
        return self.return_df
