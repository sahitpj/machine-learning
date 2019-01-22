from utils import import_data
from trees import decisionTree

import pandas as pd

filepath = 'trees/data.csv'

data = import_data(filepath)

dt = decisionTree('classification', 'gini', data, 0, 100)
tree = dt.train()
test = ['Overcast','Cold','Normal','Strong']
a,b = dt.predict(test, tree)
print a, b
