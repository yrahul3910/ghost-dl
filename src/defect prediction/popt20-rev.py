from __future__ import print_function, division
import os
cwd = os.getcwd()
import_path=os.path.abspath(os.path.join(cwd, '..'))
import sys
sys.path.append(import_path)

from helper.transformation import *
from random import seed
from helper.utilities import _randchoice, unpack
from helper.ML import *
from itertools import product
import numpy as np
import pandas as pd
from sklearn.metrics import auc
from concurrent.futures import ProcessPoolExecutor
import time
import pickle
from collections import OrderedDict
from operator import itemgetter


metrics=["d2h","popt","popt20"]
data_path = os.path.join(cwd, "..","..", "data","defect")

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
#file_inc = {"ivy": 0, "lucene": 1, "poi":  2, "synapse":3, "velocity":4, "camel": 5,"jedit": 6,
#            "log4j": 7, "xalan": 8,"xerces": 9}
file_inc = {x: i for (i, x) in enumerate(file_dic.keys())}

def _test(res=''):
    paths = [os.path.join(data_path, file_name) for file_name in file_dic[res]]
    train_df = pd.concat([pd.read_csv(path) for path in paths[:-1]], ignore_index=True)
    test_df = pd.read_csv(paths[-1])

    train_df, test_df = train_df.iloc[:, 3:], test_df.iloc[:, 3:]
    train_size=train_df["bug"].count()
    df=pd.concat([train_df,test_df],ignore_index=True)
    df['bug']=df['bug'].apply(lambda x: 0 if x == 0 else 1)

    metric="popt20"

    final = {}
    final_auc={}
    e_value = [0.2]
    start_time=time.time()
    dic={}
    dic_func={}
    for mn in range(500+file_inc[res]*10,520+file_inc[res]*10):
        for e in e_value:
            preprocess = [standard_scaler, minmax_scaler, [normalizer]*10]  # ,[polynomial]*5
            MLs = [[DeepLearner] * 20]  # [SVM]*100,
            preprocess_list = unpack(preprocess)
            MLs_list = unpack(MLs)
            combine = [[r[0], r[1]] for r in product(preprocess_list, MLs_list)]

            if e not in final_auc.keys():
                final_auc[e]=[]
                dic[e] = {}


            func_str_dic = {}
            func_str_counter_dic = {}
            lis_value = []
            dic_auc={}
            for i in combine:
                scaler, tmp1 = i[0]()
                model, tmp2 = i[1]()
                string1 = tmp1 + "|" + tmp2
                func_str_dic[string1] = [scaler, model]
                func_str_counter_dic[string1] = 0

            counter=0
            while counter!=30:
                if counter not in dic_func.keys():
                    dic_func[counter]=[]
                try:
                    keys = [k for k, v in func_str_counter_dic.items() if v == 0]
                    key = _randchoice(keys)
                    scaler,model=func_str_dic[key]
                    df1=transform(df,scaler)

                    train_data, test_data = df1.iloc[:train_size,:], df1.iloc[train_size:,:]
                    measurement = run_model(train_data, test_data, model, metric,training=-2)

                    if all(abs(t - measurement) > e for t in lis_value):
                        lis_value.append(measurement)
                        func_str_counter_dic[key] += 1
                    else:
                        func_str_counter_dic[key] += -1

                    if counter not in dic[e].keys():
                        dic[e][counter] = []
                        dic_func[counter]=[]
                    if e == 0.05:
                        dic_func[counter].append(key)
                    dic[e][counter].append(max(lis_value))
                    dic_auc[counter]=max(lis_value)

                    counter+=1
                except:
                    raise

            dic1 = OrderedDict(sorted(dic_auc.items(), key=itemgetter(0))).values()
            area_under_curve = round(auc(list(range(len(dic1))), list(dic1)), 3)
            print("AUC: ", area_under_curve)
            final[e]=dic_auc
            final_auc[e].append(area_under_curve)
    total_run=time.time()-start_time
    final_auc["temp"]=final
    final_auc["time"] = total_run
    final_auc["counter_full"]=dic
    final_auc["settings"]=dic_func
    print(final_auc)

if __name__ == '__main__':
    for i in file_inc.keys():
        if file_inc[i] >= 5: continue
        print(i)
        with ProcessPoolExecutor(max_workers=4) as executor:
            result = executor.submit(_test, i).result()
