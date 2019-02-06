from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, f1_score

from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords

import pandas as pd
import matplotlib.pyplot as plt

import re

from operator import itemgetter

SPACE = " "
stopwords_list = set(stopwords.words('english'))

def plot_roc_graph(fpr, tpr, filename):
    plt.figure()
    plt.plot(fpr, tpr, color='red', lw=2, label='ROC Curve')
    plt.plot([0, 1], [0, 1], color='blue', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.savefig(filename + '.png')

def mean_normalize(data):
    # Changes scale of the distribution
    # Brings value both +ve and -ve but binary values have [0,1]
    return (data - data.mean())/data.std()

def minmax_normalize(data):
    return (data - data.min())/(data.max() - data.min())

def split_train_test(data, target_label, drop_cols = [''], binary_cols = [''], target_values=[0,1], normalize='default'):
    if normalize == 'default': 
        print("No valid normalization technique specified. Skipping normalization")

    for col in data.columns:
        if ((col not in binary_cols) and (col not in drop_cols)):
            if normalize == 'mean':
                data[col] = mean_normalize(data[col])
            elif normalize == 'minmax':
                data[col] = minmax_normalize(data[col])

    X_train = data[data['train'] == 1]
    X_test = data[data['train'] == 0]
    y_train = X_train[target_label]
    y_test = X_test[target_label]

    X_train = X_train.drop(drop_cols, axis=1)
    X_test = X_test.drop(drop_cols, axis=1)

    return X_train, y_train, X_test, y_test

def mortality_factors(coeff, feature_names, n=5):
    coeff_to_feature = []
    for index in range(len(feature_names)):
        coeff_to_feature.append([coeff[0][index], feature_names[index]])

    coeff_to_feature = sorted(coeff_to_feature, key=itemgetter(0))
    
    print("Lowest " + str(n) + ": ")
    for coefficient, feature_name in coeff_to_feature[0:n]:
        print(coefficient, feature_name)
    
    print("Top " + str(n) + ": ")
    for coefficient, feature_name in coeff_to_feature[-n:]:
        print(coefficient, feature_name)

    print("Absolute Coefficient values ...")

    coeff_to_feature = []
    for index in range(len(feature_names)):
        coeff_to_feature.append([abs(coeff[0][index]), feature_names[index]])

    coeff_to_feature = sorted(coeff_to_feature, key=itemgetter(0))
    
    print("Lowest " + str(n) + ": ")
    for coefficient, feature_name in coeff_to_feature[0:n]:
        print(coefficient, feature_name)
    
    print("Top " + str(n) + ": ")
    for coefficient, feature_name in coeff_to_feature[-n:]:
        print(coefficient, feature_name)


def logistic_regression(X_train, y_train, X_test, y_test, penalty='l2', max_iter=100, solver='liblinear', class_weight=None):
    model = LogisticRegression(max_iter=max_iter, penalty=penalty, solver=solver, class_weight=class_weight).fit(X_train, y_train)
    y_predict = model.predict(X_test)
    y_predict_prob = model.predict_proba(X_test)
    print("No. of iterations to converge: ", model.n_iter_)
    return model, y_predict, y_predict_prob

def roc_stats(y_test, y_predict, pos_scores, name):
    fpr, tpr, thresholds = roc_curve(y_test, pos_scores, pos_label = 1)
    plot_roc_graph(fpr, tpr, name)
    print("AUC Score => ", roc_auc_score(y_test, pos_scores))

def f1_score_stats(y_test, y_predict):
    print("F1 Score is: ", f1_score(y_test, y_predict))

def tokenize(note, return_as_list = False, lowercase = False, regex=re.compile(r"\w+")):
    tokens = RegexpTokenizer(regex).tokenize(note)
    
    if lowercase:
        tokens = [token.lower() for token in tokens]

    if return_as_list:
        return [token for token in tokens if (token.lower() not in stopwords_list and token != '')]
    else:
        return SPACE.join([token for token in tokens if (token.lower() not in stopwords_list and token != '')])