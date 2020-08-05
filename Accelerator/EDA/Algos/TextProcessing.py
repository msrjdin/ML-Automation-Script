import pandas as pd
import numpy as np
# from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.corpus import stopwords
import re
import spacy
from nltk.stem.snowball import SnowballStemmer
import collections
from wordcloud import WordCloud
# import seaborn as sns
# import matplotlib.pyplot as plt


# data = pd.read_csv(r"C:\Users\SatyaSindhuMolleti\whats_cooking_csv.csv")
# print(data.shape)
# data['ingredient'] = data['ingredients']
# # print(data.head())
# col_map = {'ingredients': ['stemming', 'lemmetize'],'ingredient': ['stemming']}




class TextProcessing:
    def __init__(self, df, col_map):
        self.df = df.copy()
        self.col_map=col_map
        self.transaformed_data={}
        print(col_map,'..------------')
        for col,methods in self.col_map.items() :
            processed_text = self.preprocessText(self.df[col].values)
            # lemmetize
            if "lemmetize" in methods:
                print('------------')
                processed_text = self.lemmetizer(processed_text)

            # stemming
            if "stemming" in methods:
                processed_text = self.stemmer(processed_text)

            # remove digits
            if "remove_numericals" in methods:
                processed_text = self.remove_digits(processed_text)

            # remove stop qords
            if "remove_numericals" in methods:
                processed_text = self.remove_stop_words(processed_text)

            self.df[col]=processed_text

            # self.transaformed_data.update({col:processed_text})
        self.return_result()


    def preprocessText(self, text_data):
        cleaned_text = []
        for row in text_data:
            # Remove special chars
            text_manipulated = re.sub(r'\W', ' ', row)

            # Replace multiple spaces with single space
            text_manipulated = re.sub(r'\s+', ' ', text_manipulated, flags=re.I)

            # Converting to Lowercase
            text_manipulated = text_manipulated.lower()

            cleaned_text.append(text_manipulated)

        return cleaned_text


    def lemmetizer(self, cleaned_text):
        print("LEMMMATIZE*******************")
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

    # def matrix_build(self, cv, cleaned_text):
    #     #         cv.fit(cleaned_text)
    #     X = cv.fit_transform(cleaned_text)
    #     X_df = pd.DataFrame(X.todense())
    #     df1 = self.df
    #     df1 = df1.merge(X_df, left_index=True, right_index=True)
    #     self.df = df1
    #     return X


    # def top_features(self, cv, text_col, X):
    #     word_freq = dict(zip(cv.get_feature_names(), np.asarray(X.sum(axis=0)).ravel()))
    #     word_counter = collections.Counter(word_freq)
    #     word_counter_df = pd.DataFrame(word_counter.most_common(10), columns=['word', 'freq'])
    #     word_counter_df['column'] = text_col
    #     return word_counter_df
    #
    # def word_cloud(self, cleaned_text, bgcolor, title,text_col):
    #     data = [word for word in cleaned_text if not word.isnumeric()]
    #     # plt.clf()
    #     plt.figure(figsize=(100, 100))
    #     wc = WordCloud(background_color=bgcolor, max_words=100, max_font_size=50)
    #     wc.generate(' '.join(data))
    #     wc.to_file('C:\\Users\\SatyaSindhuMolleti\\Desktop\\'+text_col+'.png')

    def return_result(self):
        # print(self.df)
        return self.df

# tp = TextProcessing(data, col_map)
# print(tp.return_dfs().head())