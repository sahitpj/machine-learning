from trees import decisionTree, randomForest
from utils import import_data, K_fold_split, normal_split


import pandas as pd

filepath = 'datasets/tennis.csv'

data = import_data(filepath)

dt = decisionTree('classification', 'gini', data, 0, 100)
tree = dt.train()
dt.construct_tree(tree, "")


filepath_2 = 'datasets/iris.csv'
from utils import import_data, K_fold_split, normal_split, transform_pd, transform_question
train, validate, test = normal_split(import_data(filepath_2), 0.7)

print transform_question(1, import_data(filepath_2))