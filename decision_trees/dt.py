from utils import class_counts, unique_vals, is_numeric
from entropy import gini, info_gain
from nodes import Decision_Node, Leaf

import pandas as pd



'''
Input data is in the form of pandas DataFrame. (Table structure).
The last feature is assumed to be the Y-festure, ie the feature to be predicted.
'''


class Question(object):
    def __init__(self, column, value, df):
        self.column = column
        self.value = value
        self.df = df

    def match(self, row_num):
        val = self.df.iloc[row_num, self.column]
        if is_numeric(val):
            return val >= self.value
        else:
            return val == self.value

    def __repr__(self):
        condition = "=="
        if is_numeric(self.value):
            condition = ">="
        return "Is %s %s %s?" % (
            list(self.df)[self.column], condition, str(self.value))


class decisionTree(object):
    def __init__(self, method, data, max_depth=2):
        self.max_depth = max_depth
        self.method = method
        self.tree = None
        self.data = data
        self.samples = data.shape[0]
        self.features = data.shape[1]-1
        self.current_depth = 1
        self.main_rows = range(self.samples)


    def partition(self, rows, question):
        true_rows, false_rows = [], []
        for row_num in xrange(len(rows)):
            if question.match(row_num):
                true_rows.append(row_num) #appending the row index
            else:
                false_rows.append(row_num)
        return true_rows, false_rows


    def find_best_split(self, rows):
        best_gain = 0  
        best_question = None  
        current_uncertainty = gini(rows, self.data)

        for col in range(self.features): 
            values = set([ self.data.iloc[i, col] for i in rows])  # unique values in the column
            for val in values:  
                question = Question(col, val, self.data)
                true_rows, false_rows = self.partition(rows, question)
                if len(true_rows) == 0 or len(false_rows) == 0:
                    continue
                gain = info_gain(true_rows, false_rows, current_uncertainty, self.method, self.data)
                if gain >= best_gain:
                    best_gain, best_question = gain, question

        return best_gain, best_question


    def build_tree(self, rows):
        if self.current_depth == self.max_depth:
            return 
        gain, question = self.find_best_split(rows)
        if gain == 0:
            return Leaf(rows, self.data)

        true_rows, false_rows = self.partition(rows, question)
        true_branch = self.build_tree(true_rows)
        false_branch = self.build_tree(false_rows)
        self.current_depth += 1
        return Decision_Node(question, true_branch, false_branch)

    def print_tree(self, node, spacing=""):
        """World's most elegant tree printing function."""
        if isinstance(node, Leaf) == 1:
            print (spacing + "Predict", node.predictions)
            return
        print (spacing + str(node.question))
        print (spacing + '--> True:')
        print_tree(node.true_branch, spacing + "  ")
        print (spacing + '--> False:')
        print_tree(node.false_branch, spacing + "  ")

    def classify(self, row, node):
        """See the 'rules of recursion' above."""

        # Base case: we've reached a leaf
        if isinstance(node, Leaf):
            return node.predictions

        if node.question.match(row):
            return classify(row, node.true_branch)
        else:
            return classify(row, node.false_branch)

    def print_leaf(self, counts):
        """A nicer way to print the predictions at a leaf."""
        total = sum(counts.values()) * 1.0
        probs = {}
        for lbl in counts.keys():
            probs[lbl] = str(int(counts[lbl] / total * 100)) + "%"
        return probs



    def train(self):
        self.tree = self.build_tree(self.main_rows)
        print 'Tree built successfully'










