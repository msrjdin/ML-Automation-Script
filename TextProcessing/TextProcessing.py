import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.corpus import stopwords

# nlp_data ='train_nlp.json'

#data = pd.read_json(r'C:\Users\SatyaSindhuMolleti\Desktop\train.json')

#y = 'cuisine'
##colTypes = {'Categorical': [], 'Text': ['ingredients'], 'Numeric': [], 'Identity': []}
#data = data.drop('id', axis=1)



class TextProcessing:
    def __init__(self, df, colTypes, method):
        self.df = df.copy()
        self.txtCols = colTypes['Text']
        self.method = method
        for col in self.txtCols:
            if method == 'BOW':
                cv = CountVectorizer()
            elif method == 'Tfidf':
                cv = TfidfVectorizer()

            self.corpusBuild(col, cv)

    def corpusBuild(self, col, cv):
        self.cv = cv
        text_col = col
        corpus=[]

        self.df[text_col] = self.df[text_col].apply(lambda x: str(x).strip('[]').split(','))
        print(self.df[text_col][0])
        self.df['text_manipulated'] = self.df[text_col].map(";".join)
        for row in self.df[text_col]:
            for word in row:
                if word not in corpus:
                    corpus.append(word)
        self.matrixBuild(cv, corpus, text_col)


    def matrixBuild(self, cv, corpus, col):
        cv.fit(corpus)
        X = cv.transform(self.df['text_manipulated'].values)
        X_df = pd.DataFrame(X.todense())
        self.df = self.df.merge(X_df, left_index=True, right_index=True)

    def return_result(self):
        self.df.drop('text_manipulated', axis=1,inplace=True)
        self.df.drop(self.txtCols, axis=1, inplace=True)
        return self.df

# tp = TextProcessing(data, y, colTypes, "BOW")
# print(tp.return_dfs().head())