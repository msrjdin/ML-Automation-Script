import inquirer

class ColumnTypeIdentification:
    def __init__(self, df,y):
        self.dtypes={}
        self.df=df 
        self.y=y
        #Getting Columns pandas datatype
        for i in self.df.columns:
            self.dtypes[i]=(self.df[i].dtypes)
#         print(self.dtypes)
        self.colTypes={'Categorical': [], 'Text':[], 'Numeric': [], 'Identity': []}
        
        #saving the final col type (Categorical, Text or Numeric)
        self.detecting_col_types()
        self.target_type= None
#        self.col_types_confirmation()


    
    
    #Categorical if the type is object and there are 20 distinct values in the first 95 %ile of the data else Text
    #Categorical if the type is not object and distinct values are 5% of all the total records
    #Identity column.. Which is uique to each row
    #Else Numeric
    def detecting_col_types(self):
        for i in self.dtypes.keys():
            if self.dtypes[i]=='O':
                if (self.df[i].fillna('',axis=0).apply(lambda x: len(x))).quantile(q=0.95)<20:
                    self.colTypes['Categorical'].append(i)
                    self.dtypes[i]='Categorical'
                #elif self.df[i].nunique() >= int(0.98*(self.df[i].shape[0])):
                    #self.colTypes['Identity'].append(i)
                else:
                    self.colTypes['Text'].append(i)
                    self.dtypes[i]='Text'
            else: 
                distinctValues = self.df[i].nunique()
                if distinctValues < int((self.df[i].shape[0])*0.05):
                    self.colTypes['Categorical'].append(i)
                    self.dtypes[i] = 'Categorical'
                #elif self.df[i].nunique() >= int(0.98 * (self.df[i].shape[0])):
                    #self.colTypes['Identity'].append(i)
                else:
                    self.colTypes['Numeric'].append(i)
                    self.dtypes[i] = 'Numeric'

    def col_types_confirmation(self):
        columnTypesConfirmation=[]
        print('Please Confirm/Map the columns with their type:\n')
        for col in self.dtypes.keys():
            colView=self.df[col].copy()
            colView=colView[colView.notna()].head(3).values
            columnTypesConfirmation.append(
                inquirer.List(col, message=str(col) + ':\nValues: {}'.format(colView),
                                                         choices=['Text', 'Categorical', 'Numeric', 'Identity'],
                                                         default=self.dtypes[col]))
        confirmedColTypes=inquirer.prompt(columnTypesConfirmation)

        self.colTypes = {'Categorical': [], 'Text': [], 'Numeric': [], 'Identity': []}
        for col in self.dtypes.keys():
            self.dtypes[col]=confirmedColTypes[col]
            self.colTypes[confirmedColTypes[col]].append(col)
            if self.y==col:
                self.target_type=confirmedColTypes[col]
