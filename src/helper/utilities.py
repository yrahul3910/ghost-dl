from __future__ import print_function, division

from random import randint, uniform, choice, sample
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
import math
import numpy as np

PRE, REC, SPEC, FPR, NPV, ACC, F1 = 7, 6, 5, 4, 3, 2, 1

def _randint(a=0,b=0):
    return randint(a,b)

def _randchoice(a=[]):
    return choice(a)

def _randuniform(a=0.0,b=0.0):
    return uniform(a,b)

def _randsample(a=[],b=1):
    return sample(a,b)

def unpack(l):
    tmp=[]
    for i in l:
        if list!=type(i):
            tmp.append(i)
        else:
            for x in i:
                tmp.append(x)
    return tmp

def get_performance(tn, fp, fn, tp):
    pre = 1.0 * tp / (tp + fp) if (tp + fp) != 0 else 0
    rec = 1.0 * tp / (tp + fn) if (tp + fn) != 0 else 0
    spec = 1.0 * tn / (tn + fp) if (tn + fp) != 0 else 0
    fpr = 1 - spec
    npv = 1.0 * tn / (tn + fn) if (tn + fn) != 0 else 0
    acc = 1.0 * (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) != 0 else 0
    f1 = 2.0 * tp / (2.0 * tp + fp + fn) if (2.0 * tp + fp + fn) != 0 else 0
    return [round(x, 3) for x in [pre, rec, spec, fpr, npv, acc, f1]]

def get_score(criteria, prediction, test_labels,data):
    tn, fp, fn, tp = confusion_matrix(test_labels,prediction, labels=[0,1]).ravel()
    pre, rec, spec, fpr, npv, acc, f1 = get_performance(tn, fp, fn, tp)
    all_metrics = [tp, fp, tn, fn, pre, rec, spec, fpr, npv, acc, f1]
    if criteria == "Accuracy":
        score = -all_metrics[-ACC]
    elif criteria == "d2h":
        score = all_metrics[-FPR] ** 2 + (1 - all_metrics[-REC]) ** 2
        score = math.sqrt(score) / math.sqrt(2)
    elif criteria=="Pf_Auc":
        score=auc_measure(prediction,test_labels)
    elif criteria=="popt":
        score=get_auc(data)
    elif criteria=="popt20":
        score=get_popt20(data)
    elif criteria == "Gini":
        p1 = all_metrics[-PRE]  # target == 1 for the positive split
        p0 = 1 - all_metrics[-NPV]  # target == 1 for the negative split
        score = 1 - p0 ** 2 - p1 ** 2
    else:  # Information Gain
        P, N = all_metrics[0] + all_metrics[3], all_metrics[1] + all_metrics[2]
        p = 1.0 * P / (P + N) if P + N > 0 else 0  # before the split
        p1 = all_metrics[-PRE]  # the positive part of the split
        p0 = 1 - all_metrics[-NPV]  # the negative part of the split
        I, I0, I1 = (-x * np.log2(x) if x != 0 else 0 for x in (p, p0, p1))
        I01 = p * I1 + (1 - p) * I0
        score = -(I - I01)  # the smaller the better.
    return round(score, 3)


def auc_measure(prediction, test_labels):
    fpr, tpr, _ = roc_curve(test_labels, prediction, pos_label=1)
    auc1 = auc(fpr, tpr)
    return auc1

def subtotal(x):
    xx = [0]
    for i, t in enumerate(x):
        xx += [xx[-1] + t]
    return xx[1:]


def get_recall(true):
    total_true = float(len([i for i in true if i == 1]))
    hit = 0.0
    recall = []
    for i in xrange(len(true)):
        if true[i] == 1:
            hit += 1
        recall += [hit / total_true if total_true else 0.0]
    return recall


def get_auc(data):
    """The smaller the better"""
    if len(data) == 1:
        return 0
    x_sum = float(sum(data['loc']))
    x = data['loc'].apply(lambda t: t / x_sum)
    xx = subtotal(x)
    yy = get_recall(data['bug'].values)
    try:
        ret = round(auc(xx, yy), 3)
    except:
        #print"?"
        ret = 0
    return ret

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
