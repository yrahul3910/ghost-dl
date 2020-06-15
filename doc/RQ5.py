import os
import warnings
import numpy as np
import pandas as pd
import time 
import sys

warnings.filterwarnings("ignore")

sys.path.append(os.path.abspath("../src/"))
from helper.ML import *


data_path = "../data/defect/"
file_dic = {"ivy":     ["ivy-1.1.csv", "ivy-1.4.csv", "ivy-2.0.csv"],\
        "lucene":  ["lucene-2.0.csv", "lucene-2.2.csv", "lucene-2.4.csv"],\
        "poi":     ["poi-1.5.csv", "poi-2.0.csv", "poi-2.5.csv", "poi-3.0.csv"],\
        "synapse": ["synapse-1.0.csv", "synapse-1.1.csv", "synapse-1.2.csv"],\
        "velocity":["velocity-1.4.csv", "velocity-1.5.csv", "velocity-1.6.csv"], \
        "camel": ["camel-1.0.csv", "camel-1.2.csv", "camel-1.4.csv", "camel-1.6.csv"], \
        "jedit": ["jedit-3.2.csv", "jedit-4.0.csv", "jedit-4.1.csv", "jedit-4.2.csv", "jedit-4.3.csv"], \
        "log4j": ["log4j-1.0.csv", "log4j-1.1.csv", "log4j-1.2.csv"], \
        "xalan": ["xalan-2.4.csv", "xalan-2.5.csv", "xalan-2.6.csv", "xalan-2.7.csv"], \
        "xerces": ["xerces-1.2.csv", "xerces-1.3.csv", "xerces-1.4.csv"]
        }
file_inc = {"ivy": 0, "lucene": 1, "poi":  2, "synapse":3, "velocity":4, "camel": 5,"jedit": 6,
            "log4j": 7, "xalan": 8,"xerces": 9}


def _test(res=''):
    paths = [os.path.join(data_path, file_name) for file_name in file_dic[res]]
    train_df = pd.concat([pd.read_csv(path) for path in paths[:-1]], ignore_index=True)
    test_df = pd.read_csv(paths[-1])

    train_df, test_df = train_df.iloc[:, 3:], test_df.iloc[:, 3:]
    train_size=train_df["bug"].count()
    df=pd.concat([train_df,test_df],ignore_index=True)
    df['bug']=df['bug'].apply(lambda x: 0 if x == 0 else 1)

    metric="d2h"

    x = []
    for i in range(20):
        learner, _ = DeepLearner()
        a = time.process_time()
        run_model(df.iloc[:train_size, :], df.iloc[train_size:, :], learner, metric, -1)
        b = time.process_time()
        x.append(b - a) 
    print(np.median(x))


if __name__ == '__main__':
    for i in file_dic.keys():
        print(i)
        _test(i)   
