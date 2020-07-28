from EDA.Algos import *
from EDA.EDAFlow import EDAFlow
#from Initialisation.Algos import *
from Initialisation.InitFlow import *

if __name__ =="__main__":
    print(sys.path)
    app_name = 'App2'
    miniFlow = InitFlow(app_name, "dataframe.csv")
    miniFlow.detectingColType()
    miniFlow.targetCol('Survived')
    miniFlow.insights()
    miniFlow.save()
    nullcolTypes = {'impute mean':['Fare'],'impute knn':['PassengerId'],'categorical impute':['Sex']}
    outcolTypes = {'capping':['Fare','Age'],'zscore':['PassengerId']}
    enccolTypes = {'label':['Sex'] ,'one-hot':['Embarked']}
    #textcolTypes = {'ingredients': ['stemming', 'lemmetize'],'ingredient': ['stemming']} 
    textcolTypes = { }
    eda = EDAFlow(app_name,nullcolTypes,outcolTypes,enccolTypes,textcolTypes)
