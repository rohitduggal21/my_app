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

'''
myModel
	
	__init__(self, df, to_process, target)
	df: input data (pandas dataframe)
	to_process: text column to process (string)
	target: target column (string)
	=> return: None

	process_words(self, text)
	text: text to be processed {remove stop words, lemmatize} (string)
	=> return: processed text (string)

	train(self)
	train the model
	=> return: None

	predict(self, data)
	data: text for which sentiment is required (string)
	=> return: dict (keys=status,value)

	dump()
	dump the trained model as model.rd
	=> return: None
'''
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
		#remove special characters and convert to lower case
		text = self.process_words(re.sub(r"[^a-zA-Z0-9]+", " ", data).lower())
		if len(text)!=0:	
			return {'status':True,'value':int(self.model.predict([text])[0])}
		else:	
			return {'status':False,'value':'Invalid Text!'}
	def dump(self):	
		dbfile = open('model.rd', 'ab')
		pickle.dump(self, dbfile)
		dbfile.close()