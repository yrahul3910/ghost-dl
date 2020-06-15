import numpy as np
import pandas as pd
import os
from collections import Counter
import matplotlib.pyplot as plt
import statistics

import time
import warnings

# From https://machinelearningmastery.com/sequence-classification-lstm-recurrent-neural-networks-python-keras/
# and https://towardsdatascience.com/multi-class-text-classification-with-lstm-1590bee1bd17
from sklearn.model_selection import train_test_split
from keras.layers import Dense, BatchNormalization
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import label_binarize
from keras.utils import to_categorical
from keras.models import Sequential
from keras.optimizers import SGD

from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score, recall_score
from keras import backend as K
import tensorflow as tf

warnings.filterwarnings('ignore')


def get_auc(actual, preds, classes):
    return roc_auc_score(label_binarize(actual, classes), label_binarize(preds, classes))


def get_fpr(actual, preds):
    tn, fp, fn, tp = confusion_matrix(actual, preds, labels=[0,1]).ravel()
    fpr = fp * 1.0 / (tn + fp) if (tn + fp) != 0 else 0
    
    return fpr


def subtotal(x):
    xx = [0]
    for i, t in enumerate(x):
        xx += [xx[-1] + t]
    return xx[1:]


def get_recall(true):
    total_true = float(len([i for i in true if i == 1]))
    hit = 0.0
    recall = []
    for i in range(len(true)):
        if true[i] == 1:
            hit += 1
        recall += [hit / total_true if total_true else 0.0]
    return recall


def get_popt20(data):
    data.sort_values(by=["bug", "loc"], ascending=[0, 1], inplace=True)
    x_sum = float(sum(data['loc']))
    x = data['loc'].apply(lambda t: t / x_sum)
    xx = subtotal(x)

    # get  AUC_optimal
    yy = get_recall(data['bug'].values)
    xxx = [i for i in xx if i <= 0.2]
    yyy = yy[:len(xxx)]
    s_opt = round(auc(xxx, yyy), 3)

    # get AUC_worst
    xx = subtotal(x[::-1])
    yy = get_recall(data['bug'][::-1].values)
    xxx = [i for i in xx if i <= 0.2]
    yyy = yy[:len(xxx)]
    try:
        s_wst = round(auc(xxx, yyy), 3)
    except:
        # print "s_wst forced = 0"
        s_wst = 0
    
    # get AUC_prediction
    data.sort_values(by=["prediction", "loc"], ascending=[0, 1], inplace=True)
    x = data['loc'].apply(lambda t: t / x_sum)
    xx = subtotal(x)
    yy = get_recall(data['bug'].values)
    xxx = [k for k in xx if k <= 0.2]
    yyy = yy[:len(xxx)]
    try:
        s_m = round(auc(xxx, yyy), 3)
    except:
        return 0
    
    Popt = (s_m - s_wst) / (s_opt - s_wst)
    return round(Popt,3)


base_path = '../data/defect/'


file_dic = {"ivy": ["ivy-1.1.csv", "ivy-1.4.csv", "ivy-2.0.csv"],
            "lucene": ["lucene-2.0.csv", "lucene-2.2.csv", "lucene-2.4.csv"],
            "poi": ["poi-1.5.csv", "poi-2.0.csv", "poi-2.5.csv", "poi-3.0.csv"],
            "synapse": ["synapse-1.0.csv", "synapse-1.1.csv", "synapse-1.2.csv"],
            "velocity": ["velocity-1.4.csv", "velocity-1.5.csv", "velocity-1.6.csv"],
            "camel": ["camel-1.0.csv", "camel-1.2.csv", "camel-1.4.csv", "camel-1.6.csv"],
            "jedit": ["jedit-3.2.csv", "jedit-4.0.csv", "jedit-4.1.csv", "jedit-4.2.csv", "jedit-4.3.csv"],
            "log4j": ["log4j-1.0.csv", "log4j-1.1.csv", "log4j-1.2.csv"],
            "xalan": ["xalan-2.4.csv", "xalan-2.5.csv", "xalan-2.6.csv", "xalan-2.7.csv"],
            "xerces": ["xerces-1.2.csv", "xerces-1.3.csv", "xerces-1.4.csv"]
           }


def run_on_dataset(filename, metric='d2h', epochs=10, layers=4, draw_roc=False, weighted=False):
    paths = [os.path.join(base_path, file_name) for file_name in file_dic[filename]]
    train_df = pd.concat([pd.read_csv(path) for path in paths[:-1]], ignore_index=True)
    test_df = pd.read_csv(paths[-1])
    
    train_df, test_df = train_df.iloc[:, 3:], test_df.iloc[:, 3:]
    train_size = train_df["bug"].count()
    df = pd.concat([train_df, test_df], ignore_index=True)
    df['bug'] = df['bug'].apply(lambda x: 0 if x == 0 else 1)
    
    train_data = df.iloc[:train_size, :]
    test_data = df.iloc[train_size:, :]
    
    X_train = train_data[train_data.columns[:-2]]
    y_train = train_data['bug']
    X_test = test_data[test_data.columns[:-2]]
    y_test = test_data['bug']
    
    frac = sum(y_train) * 1.0 / len(y_train)
    if weighted:
        weights = np.array([1., 0.1 / frac])
    else:
        weights = np.array([1., 1.])
                
    model = Sequential()
    model.add(Dense(20, input_shape=(X_train.shape[1],), activation='relu', name='layer1'))
    
    for i in range(layers - 2):
        model.add(Dense(20, activation='relu', name='layer'+str(i+2)))
        
    model.add(Dense(1, activation='sigmoid', name='layer'+str(layers)))
    model.compile(loss=weighted_categorical_crossentropy(weights), optimizer='adam', metrics=['accuracy'])

    batch_size = 64

    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,validation_split=0.1, verbose=0, callbacks=[EarlyStopping(monitor='val_loss', patience=20, min_delta=0.0001)])
    
    y_pred = model.predict_classes(X_test)
    
    if metric == 'fpr':
        metric_ = get_fpr(y_test, y_pred)
    elif metric == 'recall':
        metric_ = recall_score(y_test, y_pred)
    elif metric == 'auc':
        metric_ = get_auc(y_test, y_pred, classes=[0,1])
    elif metric == 'popt20':
        test_data.loc[:,"prediction"]=y_pred
        metric_ = get_popt20(test_data)
    
    if draw_roc:
        fpr, tpr, _ = roc_curve(y_test, y_pred)
        print('AUC =', auc(fpr, tpr))
        print(metric, '=', metric_)
        plt.plot(fpr, tpr, color='darkorange')
        plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    
    return history, metric_


for dataset in file_dic.keys():
    print(dataset)
    print('=' * len(dataset))
    for metric in ['fpr']:
        values = []
        for i in range(20):
            _, metric_ = run_on_dataset(filename=dataset, metric=metric, epochs=10, layers=4)
            values.append(metric_)
        
        print(metric, '-', statistics.median(values))
    
    print()
