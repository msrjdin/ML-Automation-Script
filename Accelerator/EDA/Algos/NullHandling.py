import copy

from sklearn.impute import  SimpleImputer

from missingpy import KNNImputer

#col_map={'remove columns':['Age'],'impute mean':['Fare'],'impute knn':['PassengerId'],'categorical impute':['Sex']}



class NullHandling:
    def __init__(self, df,col_map):
        self.df = df.copy()
        self.col_map=col_map

        for key,value in col_map.items():
            if key == 'remove columns':
                self.remove_columns(value)
            elif key == 'impute mean':
                self.impute_mean(value)
            elif key == 'impute knn':
                self.impute_knn(value)
            elif key == 'impute zero':
                self.impute_zero(value)
            elif key == 'categorical impute':
                self.categorical_impute(value)

    def remove_columns(self,value):
        self.df.drop(value,axis=1, inplace=True)

    def impute_mean(self,value):
        imputer = SimpleImputer(strategy='mean')
        self.df[value] = imputer.fit_transform(self.df[value])

    def impute_knn(self,value):
        imputer = KNNImputer(n_neighbors=5)
        self.df[value] = imputer.fit_transform(self.df[value])

    def impute_zero(self,value):
        self.df[value] = self.df[value].fillna(0)

    def categorical_impute(self,value):
        self.df[value]=self.df[value].fillna(-1)

    def return_result(self):
        return self.df


    

    