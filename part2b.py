import re
import nltk
import pandas as pd

from nltk import word_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve

from functions import *

if __name__ == "__main__":
	data = pd.read_csv('datasets/part2/adult_notes.gz', compression='gzip')
	data.chartext = data.chartext.fillna('')

	for index, row in data.iterrows():
		data.at[index, 'chartext'] = tokenize(row['chartext']) 

	drop_cols = ['mort_icu', 'train']
	X_train_doc, y_train, X_test_doc, y_test = split_train_test(data, 'mort_icu', drop_cols=drop_cols)

	tfidf_vectorizer = TfidfVectorizer()

	print("Extracting features...")
	X_train = tfidf_vectorizer.fit_transform(X_train_doc['chartext'])
	X_test = tfidf_vectorizer.transform(X_test_doc['chartext'])

	model, y_predict, y_predict_prob = logistic_regression(X_train, y_train, X_test, y_test, penalty='l1')

	roc_stats(y_test, y_predict, y_predict_prob, 'ROC_part2b')
	f1_score_stats(y_test, y_predict)
	mortality_factors(model.coef_, tfidf_vectorizer.get_feature_names())