class NullHandling():
  
    def __init__(self, df):
        self.dict_isnull = (df.isnull().sum() / len(df)).to_dict()
        self.df=df
        self.colTypes=colTypes.copy()
		self.remove_columns()
 
    #Removing columns which have more than 75 percent nulls
    def remove_columns(self):
        cols_remove=[]
        for key in self.dict_isnull:
            if(self.dict_isnull[key]>0.75):
                cols_remove.append(key)        
        if not cols_remove:
            return self.colTypes
        else:
            for i in cols_remove:
                for j in self.colTypes.keys():
                    if i in self.colTypes[j]:
                        self.colTypes[j].remove(i)
        if y in cols_remove:
            cols_remove.remove(y)
        
        self.df.drop(cols_remove,axis=1, inplace=True)
        
#         return self.colTypes
        

    #Imputing the null values with the mean value of the column
    def continuous_impute_mean(self):
        df_temp=self.df.copy()
        imputer = SimpleImputer(strategy='mean')
        df_temp[self.colTypes['Numeric']] = imputer.fit_transform(df_temp[self.colTypes['Numeric']])
        return df_temp
		
    #Imputing the null values using KNN
    def continuous_impute_knn(self):
        df_temp=self.df.copy()
        imputer = KNNImputer(n_neighbors=5) 
        df_temp[self.colTypes['Numeric']] = imputer.fit_transform(df_temp[self.colTypes['Numeric']])
        return df_temp
    
     #Common method calling all the impute functions   
    def impute(self,strategy,fill_value = 0, fill_categorical = '-1'):
        df_temp=self.df
        
        #Dealing with Continuous cols
        if strategy is None:
            df_temp[self.colTypes['Numeric']]=df_temp[self.colTypes['Numeric']].fillna(fill_value)
        elif strategy == 'mean':
            self.continuous_impute_mean()
        elif strategy == 'knn':
            self.continuous_impute_knn()
            
        #dealing with categorical Cols 
        df_temp[self.colTypes['Categorical']]=df_temp[self.colTypes['Categorical']].fillna(fill_categorical)
        
        return df_temp
