import pandas as pd

# df = pd.read_csv(r"C:\Users\SatyaSindhuMolleti\Downloads\sample_train.csv")

class DetectingColTypes:
    def __init__(self, df):
        self.dtypes = {}
        self.df = df

        for i in self.df.columns:
            self.dtypes[i] = (self.df[i].dtypes)

        self.detecting_col_types()

    # Categorical if the type is object and there are 20 distinct values in the first 95 %ile of the data else Text
    # Categorical if the type is not object and distinct values are 5% of all the total records
    # Else Numeric
    def detecting_col_types(self):
        for i in self.dtypes.keys():
            if self.dtypes[i] == 'O':
                if (self.df[i].fillna('', axis=0).apply(lambda x: len(x))).quantile(q=0.95) < 20:
                    self.dtypes[i] = 'Categorical'
                else:
                    self.dtypes[i] = 'Text'
            else:
                distinctValues = self.df[i].nunique()
                if distinctValues < int((self.df[i].shape[0]) * 0.05):
                    self.dtypes[i] = 'Categorical'
                else:
                    self.dtypes[i] = 'Numeric'
        self.returnValues()

    def returnValues(self):
        return self.dtypes


# class_ob = DetectingColTypes(df)
