from EDA.EDAFlow import EDAFlow
#from EDA.Algos import *
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
    nullcolTypes = {}
    outcolTypes = {}
    edaflow = EDAFlow(app_name,nullcolTypes,outcolTypes)
    
