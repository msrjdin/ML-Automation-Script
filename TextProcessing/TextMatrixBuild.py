import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.corpus import word_tokenize, stopwords

class TextMatrixBuild:
    def __init__(self, df, colTypes, method, stopwordsRemoval=False):
        self.df = df.copy()
        self.txtCols = colTypes['Text']

        self.stopwordsRemoval = stopwordsRemoval
        self.corpus = {}
        self.word2idx = {}

        self.corpusBuild()

        if method=='BOW':
            cv=CountVectorizer()
            self.matrixBuild(cv)
        elif method=='Tfidf':
            cv=TfidfVectorizer
            self.matrixBuild(cv)

    def corpusBuild(self):
        for col in self.txtCols:
            self.df[col] = self.df[col].apply(lambda x: word_tokenize(x.lower()))
            if self.stopwordsRemoval:
                self.df[col] = self.df[col].apply(lambda x: [i for i in x if i not in stopwords.words('english')])
            # self.df[col]=self.df[col].apply(lambda x: [self.word2idx[col][i] for i in x])
            corpus=[]
            for i in self.df[col].values:
                corpus.extend(i)
            self.corpus[col]=set(corpus)
            self.word2idx[col]={wrd : idx for idx,wrd in enumerate(self.corpus[col])}

    def matrixBuild(self, cv):
        for col in self.txtCols:
            # cv = CountVectorizer()
            mtrx = cv.fit_transform(self.df[col])
            mtrx=pd.DataFrame(mtrx)
            mtrx.columns=['col_'+str(i) for i in mtrx.columns]
            self.df=self.df.merge(mtrs, how='left', left_index=True, right_index=True)

    def return_dfs(self):
        return self.df

