from utils import class_counts, unique_vals, is_numeric, import_data
from entropy import gini, info_gain, std, entropy
from nodes import Decision_Node, Leaf
from dt import Question, decisionTree

filepath = 'data.csv'

data = import_data(filepath)

dt = decisionTree('gini', data, 10)
dt.train()
dt.tree.show()