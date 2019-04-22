import math
import numpy 
from .utils import class_counts

'''
here rows is a list of indecies for the dataframe (sample indecies)
'''

def gini(row_indecies, data_df):
    #square loss
    counts = class_counts(row_indecies, data_df)
    impurity = 1
    k = float(len(row_indecies))
    for lbl in counts:
        prob_of_lbl = counts[lbl]/k
        impurity -= prob_of_lbl**2
    return impurity

def entropy(row_indecies, data_df):
    counts = class_counts(row_indecies, data_df)
    impurity = 1
    k = float(len(row_indecies))
    for lbl in counts:
        prob_of_lbl = counts[lbl]/k*math.log((counts[lbl]/k), 2)
        impurity -= prob_of_lbl
    return impurity


def std(rows_indecies, data_df):
    g = np.array([data_df.iloc[i, -1] for i in rows_indecies])
    return np.std(g)**2


def info_gain(left, right, current_uncertainty, method, data_df):
    #left andf right are the true_rows and false_rows indecies
    p = float(len(left)) / (len(left) + len(right))
    if method == 'gini':
        return current_uncertainty - p * gini(left, data_df) - (1 - p) * gini(right, data_df)
    elif method == 'entropy':
        return current_uncertainty - p * entropy(left, data_df) - (1 - p) * entropy(right, data_df)
    elif method == 'std':
        return current_uncertainty -  std(left, data_df) -  std(right, data_df)