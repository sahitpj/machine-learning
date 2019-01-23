from utils import import_data
from trees import decisionTree, randomForest

import pandas as pd

filepath = 'datasets/tennis.csv'

data = import_data(filepath)

# dt = decisionTree('classification', 'gini', data, 0, 100)
rf = randomForest('classification', 'gini', data, 2, 0, 100)
forest = rf.train()
rf.construct_forest()
# tree = dt.train()
# dt.construct_tree(tree, "")
# test = ['Overcast','Cold','Normal','Strong']
# a,b = dt.predict(test, tree)
# print a, b
