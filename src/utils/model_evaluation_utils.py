import csv, pandas as pd, sklearn as sk
from sklearn import metrics


def compute_metrics(list_of_preds_and_exps, threshold, unknownIntent):
    # schema is [[prediction, expected]]
    count, correct, incorrect, missed, oos = 0, 0, 0, 0, 0
    for pred_exp in list_of_preds_and_exps:
        if (pred_exp[0] == pred_exp[1]) & (pred_exp[0] != unknownIntent) & (pred_exp[1] != unknownIntent):
            correct += 1
        elif (pred_exp[0] == unknownIntent) & (pred_exp[1] != unknownIntent):
            missed += 1
        elif (pred_exp[0] != unknownIntent) & (pred_exp[0] != pred_exp[1]):
            incorrect += 1
        elif (pred_exp[0] == unknownIntent) & (pred_exp[1] == unknownIntent):
            oos += 1
        count+=1
    metric_counts = {'count': count, 'threshold': threshold, 'correct': correct, 'incorrect': incorrect, 'missed': missed, 'oos': oos,
                     'correct%': round(100*correct/count, 3), 'incorrect%': round(100*incorrect/count, 3), 'missed%': round(100*missed/count, 3), 'oos%': round(100*oos/count, 3),
                    'point_metric': round(correct-(.65*incorrect)-(0.15*missed), 2)}
    return metric_counts

