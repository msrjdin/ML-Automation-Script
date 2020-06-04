import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.corpus import stopwords

# nlp_data ='train_nlp.json'

data = pd.read_json(r'C:\Users\SatyaSindhuMolleti\Desktop\train.json')

y = 'cuisine'
colTypes = {'Categorical': [], 'Text': ['ingredients'], 'Numeric': [], 'Identity': []}
data = data.drop('id', axis=1)



class TextProcessing:
    def __init__(self, df, y, colTypes, method):
        self.df = df.copy()
        self.txtCols = colTypes['Text']
        self.method = method
        for self.col in self.txtCols:
            if method == 'BOW':
                self.cv = CountVectorizer()
            elif method == 'Tfidf':
                self.cv = TfidfVectorizer()
            self.corpus = []
            self.corpusBuild(self.col, self.cv)

    def corpusBuild(self, col, cv):
        self.cv = cv
        text_col = self.col
        processed_texts = []

        #         if (type(self.df[text_col][0]) != list):
        #             for row in self.df[text_col] :
        #                 # Remove all the special characters
        #                 processed_text = re.sub(r'\W', ' ', str(row))
        #                 # remove all single characters
        #                 processed_text = re.sub(r'\s+[a-zA-Z]\s+', ' ', processed_text)
        #                 # Substituting multiple spaces with single space
        #                 processed_text = re.sub(r'\s+', ' ', processed_text, flags=re.I)
        #                 # Converting to Lowercase
        #                 processed_text = processed_text.lower()
        #                 processed_texts.append(processed_text)
        #                 # data[text_col] = data[text_col].apply(lambda x: re.sub("[^\w]", " ", x).split())
        #                 corpus.add(row.str.split())
        #         elif  (type(self.df[text_col][0]) == list):

        self.df['text_manipulated'] = self.df[text_col].map(";".join)
        for row in self.df[text_col]:
            for word in row:
                if word not in self.corpus:
                    self.corpus.append(word)
        self.matrixBuild(self.cv)

    def matrixBuild(self, cv):
        cv.fit(self.corpus)
        X = cv.fit_transform(self.df['text_manipulated'].values)
        print(X.shape)
        X_df = pd.DataFrame(X.todense())
        df1 = self.df
        df1 = df1.merge(X_df, left_index=True, right_index=True)
        self.df = df1

    def return_dfs(self):
        self.df = self.df.drop('text_manipulated', axis=1)
        self.df = self.df.drop(self.txtCols, axis=1)
        return self.df

tp = TextProcessing(data, y, colTypes, "BOW")
print(tp.return_dfs().head())
