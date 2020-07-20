import pandas as pd
df = pd.read_csv(r"C:\Users\SindhuKarnati\Desktop\MLAccelarator\train.csv")
df=df.head()


class Insights:
    def __init__(self, df,colTypes,targetName):
        self.colTypes=colTypes
        self.df=df
        self.targetName=targetName
        self.insights={}
        for key, value in colTypes.items():
            if key == self.targetName:
               self.targetType = value

        df_describe = self.df.describe()
        data = {'x': df_describe}
        plotting_data = {'plot_type': "Table", 'data': data}
        self.insights.update({'plot_describe_data': plotting_data})


        for key,value in colTypes.items():
            if key!=self.targetName:
                if(self.targetType == 'Numeric') :
                    if(value =='Numeric') :
                        self.insights.update(self.ShowScatterPlot(key))
                    elif(value == 'Categorical'):
                        self.insights.update(self.ShowViolinPlot(key,targetName))
                elif (self.targetType == 'Categorical'):
                    if (value == 'Numeric'):
                        self.insights.update(self.ShowViolinPlot(targetName,key))
                    elif (value == 'Categorical'):
                        self.insights.update(self.ShowStackedBar(key))

            else:
                if (self.targetType == 'Categorical'):
                    x=df.groupby([self.targetName]).count()
                    data = {'x': x}
                    plotting_data = {'plot_type': "Bar", 'data': data, 'x-axis': self.targetName}
                    self.insights.update({'plot_' + self.targetName: plotting_data})
                else:
                    x=df[self.targetName].values()
                    data = {'x': x}
                    plotting_data = {'plot_type': "Boxplot", 'data': data, 'x-axis': self.targetName}
                    self.insights.update({'plot_' + self.targetName: plotting_data})

        self.returnValues()

    def ShowScatterPlot(self,feature):
        x=self.df[self.targetName].values
        y=self.df[feature].values
        data = {'x' : x, 'y' : y}
        plotting_data={'plot_type':"Scatter", 'data' : data,'x-axis':self.targetName,'y-axis':feature}
        return {'plot_'+feature : plotting_data}


    def ShowViolinPlot(self,a,b):
        x = self.df[a].values
        y = self.df[b].values
        data = {'x' : x, 'y' : y}
        plotting_data = {'plot_type': "Violin", 'data': data,'x-axis':a,'y-axis':b}
        return {'plot_' + a + '_' + b: plotting_data}


    def ShowStackedBar(self, feature):
        x = self.df[self.targetName].values
        y = self.df[feature].values
        data = {'x' : x, 'y' : y}
        plotting_data = {'plot_type': "Stacked_Bar", 'data': data,'x-axis':self.targetName,'y-axis':feature}
        return {'plot_' + feature: plotting_data}

    def returnValues(self):
        print(self.insights)
        return self.insights



ob=Insights(df,{'PassengerId': 'Numeric', 'Survived': 'Categorical','Sex':'Categorical', 'Pclass': 'Categorical', 'Name': 'Text', 'Age': 'Numeric', 'SibSp': 'Categorical', 'Parch': 'Categorical', 'Fare': 'Numeric'}
,'Survived')







    # target distribution (bar)
    # df.describe()
    # graphs for more nulls (check on box plot)
    # numeric column and target numerical: scatter
    # numeric column and target categorical: violin
    #
    # categorical column and target numerical : violin
    # both cat : confusion matrix



