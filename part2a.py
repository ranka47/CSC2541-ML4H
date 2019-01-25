import pandas as pd
import matplotlib.pyplot as plt

from functions import *

drop_cols = ['train', 'subject_id', 'hadm_id', 'icustay_id', 'mort_icu']
binary_cols = ['first_hosp_stay', 'first_icu_stay', 'adult_icu', 'eth_asian', 'eth_black', 'eth_hispanic', 'eth_other', 'eth_white', 'admType_ELECTIVE', 'admType_EMERGENCY', 'admType_NEWBORN', 'admType_URGENT']

if __name__ == "__main__":
    data = pd.read_csv('datasets/part2/adult_icu.gz', compression='gzip')
    X_train, y_train, X_test, y_test = split_train_test(data, 'mort_icu', drop_cols=drop_cols, binary_cols=binary_cols, normalize='minmax')

    model, y_predict, y_predict_prob = logistic_regression(X_train, y_train, X_test, y_test)
    roc_stats(y_test, y_predict, y_predict_prob, 'ROC_part2a')
    f1_score_stats(y_test, y_predict)
    mortality_factors(model.coef_, X_train.columns)
