from time import ctime
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.corpus import stopwords
import re
import spacy
from nltk.stem.snowball import SnowballStemmer
import collections
from wordcloud import WordCloud
import seaborn as sns


# nlp_data ='train_nlp.json'

#data = pd.read_json(r'C:\Users\SatyaSindhuMolleti\Desktop\train.json')

#y = 'cuisine'
##colTypes = {'Categorical': [], 'Text': ['ingredients'], 'Numeric': [], 'Identity': []}
#data = data.drop('id', axis=1)



class TextProcessing:
    def __init__(self, df, colTypes, method, preprossessing_steps):
        self.df = df.copy()
        self.txtCols = colTypes['Text']
        self.method = method
        self.preprossessing_steps = preprossessing_steps
        self.top_tokens = pd.DataFrame(columns=['word', 'freq', 'column'])
        for self.col in self.txtCols:
            if method == 'BOW':
                print("bow begins")
                print(ctime())
                self.cv = CountVectorizer()
                print("bow ends")
                print(ctime())
            elif method == 'Tfidf':
                self.cv = TfidfVectorizer()

            print("corpus build begins begins")
            print(ctime())
            self.corpus_build(self.col, self.cv, self.preprossessing_steps, self.top_tokens)
            print(ctime())
            print("corpus ends")


    def corpus_build(self, col, cv, preprossessing_steps, top_tokens):
        self.cv = cv
        text_col = self.col
        self.top_tokens = top_tokens
        self.preprossessing_steps = preprossessing_steps
        cleaned_text = []
        self.df['text_manipulated'] = self.df[text_col]
        for row in self.df['text_manipulated']:
            # Remove special chars
            text_manipulated = re.sub(r'\W', ' ', str(row))

            # Replace multiple spaces with single space
            text_manipulated = re.sub(r'\s+', ' ', text_manipulated, flags=re.I)

            # Converting to Lowercase
            text_manipulated = text_manipulated.lower()

            cleaned_text.append(text_manipulated)

        # lemmetize
        if "lemmetize" in self.preprossessing_steps:
            cleaned_text = self.lemmetizer(cleaned_text)

        # stemming
        if "stem" in self.preprossessing_steps:
            cleaned_text = self.stemmer(cleaned_text)

        # remove digits
        if "remove_numericals" in self.preprossessing_steps:
            cleaned_text = self.remove_digits(cleaned_text)

        # remove stop qords
        if "remove_stop_words" in self.preprossessing_steps:
            cleaned_text = self.remove_stop_words(cleaned_text)

        X = self.matrix_build(self.cv, cleaned_text)

        self.top_tokens = pd.concat([self.top_tokens, self.top_features(self.cv, text_col, X)])

        self.word_cloud(cleaned_text, 'black', 'Common Words',text_col)

    def lemmetizer(self, cleaned_text):
        lemmetized_text = []
        nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
        for row in cleaned_text:
            doc = nlp(row)
            s = " ".join([token.lemma_ for token in doc])
            lemmetized_text.append(s)
        return lemmetized_text

    def stemmer(self, cleaned_text):
        stemmed_text = []
        stemmer = SnowballStemmer(language='english')
        nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
        for row in cleaned_text:
            s = " ".join([stemmer.stem(i) for i in row.split(" ")])
            stemmed_text.append(s)
        return stemmed_text

    def remove_digits(self, cleaned_text):
        removed_digits = []
        for row in cleaned_text:
            removed_digits.append(re.sub('\d+', '', row))
        return removed_digits

    def remove_stop_words(self, cleaned_text):
        remove_stop_words = []
        for row in cleaned_text:
            tokens = [word for word in row.split(" ") if not word in stopwords.words('english')]
            remove_stop_words.append((" ").join(tokens))
        return remove_stop_words

    def matrix_build(self, cv, cleaned_text):
        #         cv.fit(cleaned_text)
        print("matrix build starts")
        print(ctime())
        X = cv.fit_transform(cleaned_text)
        X_df = pd.DataFrame(X.todense())
        df1 = self.df
        df1 = df1.merge(X_df, left_index=True, right_index=True)
        self.df = df1
        print(ctime())
        print("matric builds ends")
        return X

    def top_features(self, cv, text_col, X):
        word_freq = dict(zip(cv.get_feature_names(), np.asarray(X.sum(axis=0)).ravel()))
        word_counter = collections.Counter(word_freq)
        word_counter_df = pd.DataFrame(word_counter.most_common(10), columns=['word', 'freq'])
        word_counter_df['column'] = text_col
        return word_counter_df

    def word_cloud(self, cleaned_text, bgcolor, title,text_col):
        print("building wc graphs")
        print(ctime())
        data = [word for word in cleaned_text if not word.isnumeric()]
        # plt.clf()
        #plt.figure(figsize=(100, 100))
        wc = WordCloud(background_color=bgcolor, max_words=100, max_font_size=50)
        wc.generate(' '.join(data))
        wc.to_file('static\\'+text_col+'.png')
        print(ctime())
        print("end wc graphs")

    def return_result(self):
        self.df.drop('text_manipulated', axis=1, inplace=True)
        self.df.drop(self.txtCols, axis=1, inplace=True)
        return self.df

#tp = TextProcessing(data, colTypes, "BOW",['lemmetize','stem','remove_numericals','remove_stop_words'])