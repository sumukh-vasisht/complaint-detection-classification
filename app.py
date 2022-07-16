import re,string
import itertools
import io
import numpy as np
import pandas as pd
import csv
from collections import Counter
from symspellpy.symspellpy import SymSpell, Verbosity
from textblob import TextBlob
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import nltk
from stemming.porter2 import stem

import io
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn import model_selection, naive_bayes, svm
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from collections import Counter

class PreprocessText:

    def __init__(self, dataFrame, textColumn):
        self.dataFrame = dataFrame
        self.textColumn = textColumn
        self.dico = {}
        # self.targetColumn = targetColumn

    def setup_dictionary(self):
        dico1 = open('dicos/dico1.txt', 'rb')
        for word in dico1:
            word = word.decode('utf8')
            word = word.split()
            self.dico[word[1]] = word[3]
        dico1.close()
        dico2 = open('dicos/dico2.txt', 'rb')
        for word in dico2:
            word = word.decode('utf8')
            word = word.split()
            self.dico[word[0]] = word[1]
        dico2.close()
        dico3 = open('dicos/dico2.txt', 'rb')
        for word in dico3:
            word = word.decode('utf8')
            word = word.split()
            self.dico[word[0]] = word[1]
        dico3.close()

    def correct_spell(self, text):
        text = text.split()
        for word in text:
            if(word in self.dico.keys()):
                word = self.dico[word]
        text = ' '.join(text)
        return text

    def remove_link(self, text):
        text = re.sub(r'http.?://[^\s]+[\s]?','', text)
        text = re.sub(r'^https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
        return text

    def lemmatize_stemming(self, text):
        return stem(WordNetLemmatizer().lemmatize(text, pos='v'))

    def clean_dataset(self, text):
        text = self.remove_link(text)
        text = re.sub(r"i'm", "i am", text)
        text = re.sub(r"he's","he is",text)
        text = re.sub(r"she's","she is",text)
        text = re.sub(r"that's","that is",text)
        text = re.sub(r"what's","what is",text)
        text = re.sub(r"where's","where is",text)
        text = re.sub(r"\'ll","will",text)
        text = re.sub(r"\'re","are",text)
        text = re.sub(r"\'d","would",text)
        text = re.sub(r"won't","will not",text)
        text = re.sub(r"can't","can not",text)
        text = re.sub(r"[-()\";:<>{}+=?,]","",text)
        text = re.sub('\d+','', text)
        text = re.sub(r'[^\x00-\x7F]+','',text)

        punct=string.punctuation
        transtab=str.maketrans(punct,len(punct)*' ')

        text = self.correct_spell(text)

        return text.strip()

    def remove_hashtag(self, text):
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#\w+','', text)
        text = self.remove_link(text)
        text = self.clean_dataset(text)
        return(text.strip())

    def remove_stopwords(self, text):
        result = []
        for token in gensim.utils.simple_preprocess(text):
            if token not in gensim.parsing.preprocessing.STOPWORDS and len(token)>3:
                result.append(self.lemmatize_stemming(token))
        result = ' '.join(str(x) for x in result)
        return str(result)

    def preprocess(self, text):
        result = []
        for token in gensim.utils.simple_preprocess(text):
            if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
                result.append(self.lemmatize_stemming(token))

        result=" ".join(str(x) for x in result)
        return str(result)

    def no_propernoun(self, text):
        text = list(text.split(' '))
        for i in text:
            if i not in self.dico.values():
                text.remove(i)
        text = ' '.join(str(x) for x in text)
        return str(text)

    def get_processed_dataframe(self):
        # dataFrame = pd.read_csv(self.dataFrame, encoding="ISO-8859-1")
        dataset = self.dataFrame.copy()
        # print('CLEANING DATASET')
        dataset['cleaned']=dataset[self.textColumn].astype(str).apply(self.clean_dataset)
        # print('REMOVING HASHTAG')
        dataset['hashtag'] = dataset['cleaned'].astype(str).apply(self.remove_hashtag)
        # print("REMOVING STOPWORDS")
        dataset['without_stopword']=dataset['hashtag'].astype(str).apply(self.remove_stopwords)
        # print("LEMMATIZING")
        dataset['lemmatize'] = dataset['without_stopword'].astype(str).apply(self.preprocess)
        # print("REMOVING PROPER NOUNS")
        dataset['without_propernoun'] = dataset['lemmatize'].astype(str).apply(self.no_propernoun)
        self.dataFrame['preprocessed'] = dataset['without_propernoun']
        return self.dataFrame

class BinaryClassClassification():

    def __init__(self):
        self.Encoder = LabelEncoder()
        self.model = ''  
        self.lr = LogisticRegression()
        self.Naive = naive_bayes.MultinomialNB()
        self.SVM = SVC(C=1.0, kernel='linear', degree=3, gamma='auto')

    def fit(self, dataFrame, featureColumn, targetColumn):
        X = dataFrame[featureColumn].values
        y = dataFrame[targetColumn].values

        split_size = 0.20
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = split_size, random_state=42)  

        y_train = self.Encoder.fit_transform(y_train.ravel())
        y_test = self.Encoder.fit_transform(y_test.ravel())

        Tfidf_vector = TfidfVectorizer(max_features=5000)
        Tfidf_vector.fit(X.ravel().astype('U'))
        Train_X_Tfidf = Tfidf_vector.transform(X_train.ravel().astype('U'))
        Test_X_Tfidf = Tfidf_vector.transform(X_test.ravel().astype('U'))

        # lr = LogisticRegression()
        self.lr.fit(Train_X_Tfidf,y_train)
        predictions = self.lr.predict(Test_X_Tfidf)
        # print("Logistic Regression Accuracy Score with columns(",featureColumn, targetColumn,") -> ",accuracy_score(predictions, y_test)*100)
        lrAccuracy = accuracy_score(predictions, y_test)*100

        # Naive = naive_bayes.MultinomialNB()
        self.Naive.fit(Train_X_Tfidf, y_train)
        predictions = self.Naive.predict(Test_X_Tfidf)
        # print("Naive Bayes Accuracy Score with columns(",featureColumn, targetColumn,") -> ",accuracy_score(predictions, y_test)*100)
        nbAccuracy = accuracy_score(predictions, y_test)*100

        # SVM = SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
        self.SVM.fit(Train_X_Tfidf, y_train)
        predictions = self.SVM.predict(Test_X_Tfidf)
        # print("SVM Accuracy Score with columns(",featureColumn, targetColumn,") -> ",accuracy_score(predictions, y_test)*100)
        svmAccuracy = accuracy_score(predictions, y_test)*100

        if(lrAccuracy>=nbAccuracy and lrAccuracy>=svmAccuracy):
            self.model = 'lr'
        elif(nbAccuracy>=lrAccuracy and nbAccuracy>=svmAccuracy):
            self.model = 'nb'
        else:
            self.model = 'svm'

    def predict(self, dataFrame, featureColumn):

        X = dataFrame[featureColumn].values

        Tfidf_vector = TfidfVectorizer(max_features=5000)
        Tfidf_vector.fit(X.ravel().astype('U'))
        X_Tfidf = Tfidf_vector.transform(X.ravel().astype('U'))

        if(self.model == 'lr'):
            predictions = self.lr.predict(X_Tfidf)
        elif(self.model == 'nb'):
            predictions = self.Naive.predict(X_Tfidf)
        else:
            predictions = self.SVM.predict(X_Tfidf)

        # print(predictions)
        return predictions

class MultiClassClassification:

    def __init__(self):

        self.Encoder = LabelEncoder()
        self.model = ''  
        self.lr = LogisticRegression()
        self.Naive = naive_bayes.MultinomialNB()
        self.SVM = SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
        self.categories = 1
        self.knn = KNeighborsClassifier(n_neighbors=self.categories)

    def fit(self, dataFrame, featureColumn, targetColumn):
        X = dataFrame[featureColumn].values
        y = dataFrame[targetColumn].values

        items = Counter(y).keys()
        self.categories = len(items)
        self.knn = KNeighborsClassifier(n_neighbors=self.categories)

        split_size = 0.20
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = split_size, random_state=42)  

        y_train = self.Encoder.fit_transform(y_train.ravel())
        y_test = self.Encoder.fit_transform(y_test.ravel())

        Tfidf_vector = TfidfVectorizer(max_features=5000)
        Tfidf_vector.fit(X.ravel().astype('U'))
        Train_X_Tfidf = Tfidf_vector.transform(X_train.ravel().astype('U'))
        Test_X_Tfidf = Tfidf_vector.transform(X_test.ravel().astype('U'))

        # lr = LogisticRegression()
        self.lr.fit(Train_X_Tfidf,y_train)
        predictions = self.lr.predict(Test_X_Tfidf)
        # print("Logistic Regression Accuracy Score with columns(",featureColumn, targetColumn,") -> ",accuracy_score(predictions, y_test)*100)
        lrAccuracy = accuracy_score(predictions, y_test)*100

        # Naive = naive_bayes.MultinomialNB()
        self.Naive.fit(Train_X_Tfidf, y_train)
        predictions = self.Naive.predict(Test_X_Tfidf)
        # print("Naive Bayes Accuracy Score with columns(",featureColumn, targetColumn,") -> ",accuracy_score(predictions, y_test)*100)
        nbAccuracy = accuracy_score(predictions, y_test)*100

        # SVM = SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
        self.SVM.fit(Train_X_Tfidf, y_train)
        predictions = self.SVM.predict(Test_X_Tfidf)
        # print("SVM Accuracy Score with columns(",featureColumn, targetColumn,") -> ",accuracy_score(predictions, y_test)*100)
        svmAccuracy = accuracy_score(predictions, y_test)*100

        self.knn.fit(Train_X_Tfidf, y_train)
        predictions = self.knn.predict(Test_X_Tfidf)
        # print("KNeighborsClassifier Accuracy Score with columns(", featureColumn, targetColumn, ") -> ", accuracy_score(predictions, y_test)*100)
        knnAccuracy = accuracy_score(predictions, y_test)*100

        if(lrAccuracy>=nbAccuracy and lrAccuracy>=svmAccuracy and lrAccuracy>=knnAccuracy):
            self.model = 'lr'
        elif(nbAccuracy>=lrAccuracy and nbAccuracy>=svmAccuracy and nbAccuracy>=knnAccuracy):
            self.model = 'nb'
        elif(svmAccuracy>=lrAccuracy and svmAccuracy>=nbAccuracy and svmAccuracy>=knnAccuracy):
            self.model = 'svm'
        else:
            self.model = 'knn'

    def predict(self, dataFrame, featureColumn):

        # print(self.model)

        X = dataFrame[featureColumn].values

        Tfidf_vector = TfidfVectorizer(max_features=5000)
        Tfidf_vector.fit(X.ravel().astype('U'))
        X_Tfidf = Tfidf_vector.transform(X.ravel().astype('U'))

        if(self.model == 'lr'):
            predictions = self.lr.predict(X_Tfidf)
        elif(self.model == 'nb'):
            predictions = self.Naive.predict(X_Tfidf)
        elif(self.model == 'svm'):
            predictions = self.SVM.predict(X_Tfidf)
        else:
            predictions = self.knn.predict(X_Tfidf)

        # print(predictions)
        return predictions

class HotEncoding:

    # def __init__(self):

    def encode(self, dataFrame, column):

        columnValues = dataFrame[column].values
        distinctValues = []
        for value in columnValues:
            if value not in distinctValues:
                distinctValues.append(value)
        valueDict = {}
        count = 0
        for value in distinctValues:
            valueDict[count] = value
            count += 1
        invValueDict = {v: k for k, v in valueDict.items()}
        dataFrame[column].replace(invValueDict, inplace = True)
        return dataFrame

df = pd.read_csv('tweets_dataset.csv')
preprocessModel = PreprocessText(df, 'tweet')
finalDf = preprocessModel.get_processed_dataframe()
print(finalDf) 

model = BinaryClassClassification()
model.fit(finalDf, 'preprocessed', 'complaint')

model.predict(finalDf, 'preprocessed')    

model = MultiClassClassification(categories = 17)
model.fit(finalDf, 'preprocessed', 'category')

model.predict(finalDf, 'preprocessed')   

#####################################################################################################

df = pd.read_csv('tweets_dataset.csv')
preprocessModel = PreprocessText(df, 'tweet')
finalDf = preprocessModel.get_processed_dataframe()
print(finalDf) 

model = BinaryClassClassification()
model.fit(finalDf, 'preprocessed', 'complaint')

model.predict(finalDf, 'preprocessed')    

model = MultiClassClassification(categories = 17)
model.fit(finalDf, 'preprocessed', 'category')

model.predict(finalDf, 'preprocessed')   

#####################################################################################################

df = pd.read_csv('Tweets.csv')

model = HotEncoding()
df = model.encode(df, 'airline_sentiment')
print(df.head())

preprocessModel = PreprocessText(df, 'text')
finalDf = preprocessModel.get_processed_dataframe()

model = MultiClassClassification()
model.fit(finalDf, 'preprocessed', 'airline_sentiment')

model.predict(finalDf, 'preprocessed')   

    