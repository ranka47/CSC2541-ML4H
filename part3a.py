from sklearn.datasets import load_svmlight_file
from sklearn.linear_model import LogisticRegression
import pandas as pd
from enum import Enum
from functions import *

"""
Columns for hypertension_charts
Index(['subject_id', 'hadm_id', 'charttime', 'itemid', 'valuenum'], dtype='object')
    => Heart rate in bpm (itemid = 220045)
    => Respiratory rate in breaths / minute (itemid = 220210)
    => O2 saturation in % (itemid = 220277)
    => Blood pressure in mmHg (itemid = 220181)

Columns for hypertension_patients
Index(['subject_id', 'hadm_id', 'hypertension', 'train'], dtype='object')
"""

class items(Enum):
    HEART_RATE = 220045
    RESPIRATORY_RATE = 220210
    O2_SATURATION = 220277
    BLOOD_PRESSURE = 220181


data = pd.read_csv("datasets/part3/hypertension_charts.gz", compression='gzip')
patients_data = pd.read_csv("datasets/part3/hypertension_patients.gz", compression='gzip')

for item in items:
    print("For measurement:", item.name)

    filtered_data = data[data['itemid'] == item.value]
    
    # Hospital Admission ID taken because hypertension depends only on the current time states
    drop_hadm_id = [id for id, group in filtered_data.groupby('hadm_id') if (group.shape[0] < 2)]
    print("No. of patients removed: ", len(drop_hadm_id))
    filtered_data = filtered_data[~filtered_data['hadm_id'].isin(drop_hadm_id)]

    filtered_data = pd.merge(filtered_data, patients_data, how='left', on=['subject_id', 'hadm_id'])
    temp_X_train, temp_y_train, temp_X_test, temp_y_test = split_train_test(filtered_data, 'hypertension', ['charttime', 'itemid', 'subject_id', 'train'])
    
    X_train = temp_X_train.groupby('hadm_id')['valuenum'].agg([pd.np.min, pd.np.max, pd.np.mean])
    y_train = temp_X_train.groupby('hadm_id')['hypertension'].agg([pd.np.min])
    X_test = temp_X_test.groupby('hadm_id')['valuenum'].agg([pd.np.min, pd.np.max, pd.np.mean])
    y_test = temp_X_test.groupby('hadm_id')['hypertension'].agg([pd.np.min])

    model, y_predict, y_predict_prob = logistic_regression(X_train, y_train.values.ravel(), X_test, y_test.values.ravel())

    roc_stats(y_test.values.ravel(), y_predict, y_predict_prob[:,1], 'ROC_part3a_' + item.name)
    f1_score_stats(y_test, y_predict)
    print("==============================================")

"""
For measurement: HEART_RATE
No valid normalization technique specified. Skipping normalization
AUC Score =>  0.5257705376792808 [19]

For measurement: RESPIRATORY_RATE
No valid normalization technique specified. Skipping normalization
AUC Score =>  0.5274909367989626 [33]

For measurement: O2_SATURATION
No valid normalization technique specified. Skipping normalization
AUC Score =>  0.513004689276241 [21]

For measurement: BLOOD_PRESSURE
No valid normalization technique specified. Skipping normalization
AUC Score =>  0.543442301845312 [49]
"""