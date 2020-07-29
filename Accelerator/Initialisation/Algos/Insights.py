
import pandas as pd

df = pd.read_csv(r"C:\Users\SindhuKarnati\Desktop\MLAccelarator_v2\Accelerator\dataframe.csv")
df=df.head()


class Insights:
    def __init__(self, df, colTypes, targetName):
        self.colTypes = colTypes
        self.df = df
        self.targetName = targetName
        self.insights = {}


        for key, value in colTypes.items():
            if key == self.targetName:
                self.targetType = value

        describe_df=df.describe()
        data = {'x' : describe_df}
        plotting_data={'plot_type': "describe_table", 'data': data}
        self.insights.update({'plot_describe':plotting_data})


        for key, value in colTypes.items():
            if (self.targetType == 'Numeric'):
                if (value == 'Numeric'):
                    self.insights.update(self.ShowScatterPlot(key))
                elif (value == 'Categorical'):
                    self.insights.update(self.ShowViolinPlot(key))
            elif (self.targetType == 'Categorical'):
                if (value == 'Numeric'):
                    self.insights.update(self.ShowViolinPlot(key))
                elif (value == 'Categorical'):
                    self.insights.update(self.ShowStackedBarPlot(key))



    def ShowScatterPlot(self,feature): # {"scatter": [col, data]}
        x=self.df[self.targetName].values
        y=self.df[feature].values
        data = {'x' : x, 'y' : y}
        plotting_data={'plot_type':"Scatter", 'data' : data}
        return {'plot_'+feature : plotting_data}


    def ShowViolinPlot(self, feature):
        x = self.df[self.targetName].values
        y = self.df[feature].values
        data = {'x': x, 'y': y}
        plotting_data = {'plot_type': "Violin", 'data': data}
        return {'plot_' + feature: plotting_data}


    def ShowStackedBarPlot(self, feature):
        x = self.df[self.targetName].values
        y = self.df[feature].values
        data = {'x': x, 'y': y}
        plotting_data = {'plot_type': "Stacked_Bar", 'data': data}
        return {'plot_' + feature: plotting_data}


    # def returnValues(self):
    #     print(self.insights)
    #     return self.insights



ob=Insights(df,{'PassengerId': 'Numeric', 'Survived': 'Categorical', 'Pclass': 'Categorical', 'Name': 'Text', 'Sex': 'Categorical', 'Age': 'Numeric', 'SibSp': 'Categorical', 'Parch': 'Categorical', 'Ticket': 'Categorical', 'Fare': 'Numeric', 'Cabin': 'Categorical', 'Embarked': 'Categorical'}
,'Survived')
print(ob.insights)

    # target distribution (bar)
    # df.describe()
    # graphs for more nulls (check on box plot)
    # numeric column and target numerical: scatter
    # numeric column and target categorical: violin
    #
    # categorical column and target numerical : violin
    # both cat : confusion matrix



