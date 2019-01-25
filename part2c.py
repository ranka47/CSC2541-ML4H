import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from scipy.sparse import hstack

from functions import *

data_icu = pd.read_csv('datasets/part2/adult_icu.gz', compression='gzip')
data_notes = pd.read_csv('datasets/part2/adult_notes.gz', compression='gzip')

# Extracting feature and splitting for adult_icu.gz
drop_cols = ['train', 'subject_id', 'hadm_id', 'icustay_id', 'mort_icu']
binary_cols = ['first_hosp_stay', 'first_icu_stay', 'adult_icu', 'eth_asian', 'eth_black', 'eth_hispanic', 'eth_other', 'eth_white', 'admType_ELECTIVE', 'admType_EMERGENCY', 'admType_NEWBORN', 'admType_URGENT']

X_train_icu, y_train_icu, X_test_icu, y_test_icu = split_train_test(data_icu, 'mort_icu', drop_cols=drop_cols, binary_cols=binary_cols, normalize='minmax')

# Extracting feature and splitting for adult_notes.gz
drop_cols = ['mort_icu', 'train']
data_notes.chartext = data_notes.chartext.fillna('')

for index, row in data_notes.iterrows():
	data_notes.at[index, 'chartext'] = tokenize(row['chartext']) 

X_train_doc, y_train_notes, X_test_doc, y_test_notes = split_train_test(data_notes, 'mort_icu', drop_cols=drop_cols)

tfidf_vectorizer = TfidfVectorizer()

print("Extracting features...")
X_train_notes = tfidf_vectorizer.fit_transform(X_train_doc['chartext'])
X_test_notes = tfidf_vectorizer.transform(X_test_doc['chartext'])

# Under the assumption to take the split from one of the files, i.e., adult_icu
X_train = hstack([X_train_icu, X_train_notes])
X_test = hstack([X_test_icu, X_test_notes])

model, y_predict, y_predict_prob = logistic_regression(X_train, y_train_icu, X_test, y_test_icu, penalty='l2', max_iter=1000)

roc_stats(y_test_icu, y_predict, y_predict_prob, 'ROC_part2c')
f1_score_stats(y_test_icu, y_predict)
mortality_factors(model.coef_, list(X_train_icu.columns) + list(tfidf_vectorizer.get_feature_names()))

"""
AUC Score =>  0.8471296737212014 [295]
"""