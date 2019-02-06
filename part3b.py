from enum import Enum

import pandas as pd
import numpy as np

from keras.layers import LSTM, Dense
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

from functions import *

def input_generator(input, labels = []):
    while True:
        if len(labels) > 0:
            for sequence, label in zip(input, labels):
                # print(sequence.reshape((1, sequence.shape[0], 1)).shape, label.shape)
                yield(sequence.reshape((1, sequence.shape[0], 1)), label.reshape((1,1)))
        else:
            for sequence in input:
                yield(sequence.reshape((1, sequence.shape[0], 1)))

def create_sequential_input(df, col_name, groupby_col):
    groups = df.groupby(groupby_col)
    sequences = np.array([np.array(col_df[col_name]) for col_value, col_df in groups])
    return sequences

class items(Enum):
    # HEART_RATE = 220045
    # RESPIRATORY_RATE = 220210
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
    filtered_data = filtered_data.sort_values('charttime')
    filtered_data = pd.merge(filtered_data, patients_data, how='left', on=['subject_id', 'hadm_id'])
    temp_X_train, temp_y_train, temp_X_test, temp_y_test = split_train_test(filtered_data, 'hypertension', ['charttime', 'itemid', 'subject_id', 'train'])

    filtered_data = []

    print("Creating input....")
    X_train = create_sequential_input(temp_X_train, 'valuenum', 'hadm_id')
    y_train = temp_X_train.groupby('hadm_id')['hypertension'].agg([pd.np.min])
    y_train = np.array(y_train)
    X_test = create_sequential_input(temp_X_test, 'valuenum', 'hadm_id')
    y_test = temp_X_test.groupby('hadm_id')['hypertension'].agg([pd.np.min])
    y_test = np.array(y_test)
    print("Input created!!!!")

    model = Sequential()
    model.add(LSTM(8, input_shape=(None, 1)))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit_generator(input_generator(X_train, y_train), steps_per_epoch=X_train.shape[0], epochs=1)

    # print(model.summary())
    y_predict_prob = model.predict_generator(input_generator(X_test), steps=X_test.shape[0])

    y_predict = np.array([0 if row < 0.5 else 1 for row in y_predict_prob])

    roc_stats(y_test, y_predict, y_predict_prob, 'ROC_part3b_' + item.name)
    f1_score_stats(y_test, y_predict)

"""
For measurement: HEART_RATE
15360/15360 [==============================] - 802s 52ms/step - loss: 0.6855 - acc: 0.5643  
-------------Another Attempt--------------------
Epoch 1/3
15360/15360 [==============================] - 772s 50ms/step - loss: 0.6864 - acc: 0.5586  
Epoch 2/3
15360/15360 [==============================] - 726s 47ms/step - loss: 0.6841 - acc: 0.5680
Epoch 3/3
15360/15360 [==============================] - 767s 50ms/step - loss: 0.6842 - acc: 0.5680
AUC Score =>  0.5080431907081346
F1 Score is:  0.0

For measurement: RESPIRATORY_RATE
15354/15354 [==============================] - 708s 46ms/step - loss: 0.6847 - acc: 0.5653
AUC Score =>  0.5373868638011491
F1 Score is:  0.0006891798759476222

For measurement: O2_SATURATION
15343/15343 [==============================] - 704s 46ms/step - loss: 0.6845 - acc: 0.5670  
AUC Score =>  0.4995157730200833
F1 Score is:  0.0006891798759476222

For measurement: BLOOD_PRESSURE
15071/15071 [==============================] - 326s 22ms/step - loss: 0.6842 - acc: 0.5680  
AUC Score =>  0.5099836479017081
F1 Score is:  0.0
"""