import pandas as pd


def unique_vals(rows, col):
    return set([row[col] for row in rows])

def class_counts(row_indecies, data_df):
    #Counts the number of classes and their distribution.
    counts = {}  
    for row in xrange(len(row_indecies)):
        label = data_df.iloc[row_indecies[row], -1]
        if label not in counts:
            counts[label] = 0
        counts[label] += 1
    return counts

def is_numeric(value):
    return isinstance(value, int) or isinstance(value, float)


def import_data(filepath):
    data = pd.read_csv(filepath)
    return data