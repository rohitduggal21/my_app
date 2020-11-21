import pandas as pd
import re
import pickle
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import WordNetLemmatizer
stop_words = set(stopwords.words('english'))

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

class myModel:	
	def __init__(self, df, to_process, target):	
		self.to_process = to_process
		self.target = target
		self.df = df[[to_process, target]]
		self.df[to_process] = self.df[to_process].apply(lambda x: self.process_words(re.sub(r"[^a-zA-Z0-9]+", " ", x).lower()))
		self.model = None
		self.train_score = None
		self.test_score = None
	def process_words(self,text):	
		result = []
		lem = WordNetLemmatizer()
		word_tokens = word_tokenize(text)
		for word in word_tokens:	
			if word not in stop_words:	
				result.append(lem.lemmatize(word))
		return ','.join(result).replace(',',' ')
	def train(self):	
		X_train, X_test, y_train, y_test = train_test_split(self.df[self.to_process], self.df[self.target])
		pipe = Pipeline([('vector',TfidfVectorizer()),('classifier',LogisticRegression())])
		self.model = GridSearchCV(pipe,cv=3,param_grid={'classifier__C':[0.01,0.001]})
		self.model.fit(X_train,y_train)
		self.train_score = self.model.score(X_train,y_train)
		self.test_score = self.model.score(X_test,y_test)
	def predict(self,data):	
		text = self.process_words(re.sub(r"[^a-zA-Z0-9]+", " ", data).lower())
		if len(text)!=0:	
			return {'status':True,'value':int(self.model.predict([text])[0])}
		else:	
			return {'status':False,'value':'Invalid Text!'}
	def dump(self):	
		dbfile = open('model.rd', 'ab')
		pickle.dump(self, dbfile)
		dbfile.close()