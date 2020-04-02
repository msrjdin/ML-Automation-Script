

class ColumnTypeIdentification:
    
    def __init__(self, df,y):
        self.dtypes={}
        self.df=df 
        self.y=y
        #Getting Columns pandas datatype
        for i in self.df.columns:
            self.dtypes[i]=(self.df[i].dtypes)
#         print(self.dtypes)
        self.colTypes={'Categorical': [], 'Text':[], 'Numeric': []}
        
        #saving the final col type (Categorical, Text or Numeric)
        self.detecting_col_types()
        target_type=''
        for i in self.colTypes.keys():
            if  self.y in self.colTypes[i]:
                target_type=i
                break
        self.target_type=target_type
    
    
    #Categorical if the type is object and there are 20 distinct values in the first 95 %ile of the data else Text
    #Categorical if the type is not object and distinct values are 5% of all the total records
    #Else Numeric
    def detecting_col_types(self):
        for i in self.dtypes.keys():
            if self.dtypes[i]=='O':
                if (self.df[i].fillna('',axis=0).apply(lambda x: len(x))).quantile(q=0.95)<20:
                    self.colTypes['Categorical'].append(i)
                else:
                    self.colTypes['Text'].append(i)
            else: 
                distinctValues = self.df[i].nunique()
                if distinctValues < int((self.df[i].shape[0])*0.05):
                    self.colTypes['Categorical'].append(i)
                else:
                    self.colTypes['Numeric'].append(i)
    