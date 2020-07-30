import pandas as pd

# col_map = {'sub:['add':['Age','Fare','X']],'C']}
col_map = {'add':['Age','Fare','X']}
# {'add':['a','b']}, {'add':['c','d']},{'sub:['add':['Age','Fare','X']],'C']}
df=pd.read_csv('C://Users//SindhuKarnati//Desktop//MLAccelarator_v2//Accelerator//dataframe.csv')
df=df.head()
print(df)

[a+b-c, c+b]

class CombinationalTransform:

    def __init__(self, df, col_map):
        self.df = df
        self.col_map = col_map
        self.df_temp=pd.DataFrame()
        self.col_list=[]
        for key,value in self.col_map.items():
            if key == 'add':
                self.add(self.df[value])
            if key == 'sub':
                self.sub(self.df[value])
            if key == 'multiply':
                self.sub(self.df[value])
            if key == 'divide':
                self.sub(self.df[value])


    def add(self,df):
        df_=df.copy(deep=True)
        self.col_list = df.columns
        self.df_temp['add']=df[self.col_list[0]]
        col_name='add_'+self.col_list[0]
        for i in range(1,len(self.col_list)):
            col_name=col_name+'_'+self.col_list[i]
            self.df_temp['add'] = self.df_temp['add']+df[self.col_list[i]]
        self.df_temp.rename(columns={'add':col_name},inplace=True)
        print(self.df_temp)
        return self.df_temp


    def sub(self,df):
        df_=df.copy(deep=True)
        self.col_list = df.columns
        self.df_temp['sub']=df[self.col_list[0]]
        col_name='sub_'+self.col_list[0]
        for i in range(1,len(self.col_list)):
            col_name=col_name+'_'+self.col_list[i]
            self.df_temp['sub'] = df[self.col_list[i]]-self.df_temp['sub']
        self.df_temp.rename(columns={'sub':col_name},inplace=True)
        print(self.df_temp)
        return self.df_temp



    def multiply(self,df):
        df_=df.copy(deep=True)
        self.col_list = df.columns
        self.df_temp['multiply']=df[self.col_list[0]]
        col_name='multiply_'+self.col_list[0]
        for i in range(1,len(self.col_list)):
            col_name=col_name+'_'+self.col_list[i]
            self.df_temp['multiply'] = self.df_temp['multiply']*df[self.col_list[i]]
        self.df_temp.rename(columns={'multiply':col_name},inplace=True)
        return self.df_temp


    def divide(self,df):
        df_=df.copy(deep=True)
        self.col_list = df.columns
        self.df_temp['divide']=df[self.col_list[0]]
        col_name='divide_'+self.col_list[0]
        for i in range(1,len(self.col_list)):
            col_name=col_name+'_'+self.col_list[i]
            self.df_temp['divide'] = self.df_temp['divide']/df[self.col_list[i]]
        self.df_temp.rename(columns={'divide':col_name},inplace=True)
        return self.df_temp


ct=CombinationalTransform(df,col_map)





