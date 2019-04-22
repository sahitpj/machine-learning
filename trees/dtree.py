from .utils import class_counts, unique_vals, is_numeric
from .entropy import gini, info_gain, std, entropy
from .nodes import Decision_Node, Leaf, Question

import pandas as pd

'''
Input data is in the form of pandas DataFrame. (Table structure), however input to methods 
is taken as list of row indecies for the dataframe
The last feature is assumed to be the Y-feature, ie the feature to be predicted.
'''


class decisionTree(object):
    '''
    - resanpling status, tells whether features can be taken again for splitting
    - type here means either a classification - distinct classes or regression - continous classes
    - method here is minimization function - gini, entropy or std
    '''
    def __init__(self, type_, method, data, resampling_status=0, max_depth=2 ): 
        self.max_depth = 2
        self.counter = 0
        self.method = method 
        self.type = type_ 
        self.tree = None
        self.data = data
        self.samples = data.shape[0]
        self.features = data.shape[1]-1
        self.current_depth = 1
        self.main_rows = range(self.samples)
        self.features_done = []
        self.resampling_status = resampling_status
        if self.type == 'regression':
            assert(self.method == 'std')


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
        if self.type == 'classification':
            if self.method == 'gini':
                current_uncertainty = gini(rows, self.data)
            elif self.method == 'entropy':
                current_uncertainty = entropy(rows, self.data)
        elif self.type == 'regression':
            current_uncertainty = std(rows, self.data)
        for col in range(self.features): 
            if self.is_feature_done(col):
                continue #goes to the next value of the loop
            values = set([ self.data.iloc[i, col] for i in rows])  # unique values in the column
            for val in values:  
                question = Question(col, val, self.data)
                true_rows, false_rows = self.partition(rows, question)
                if len(true_rows) == 0 or len(false_rows) == 0:
                    continue
                gain = info_gain(true_rows, false_rows, current_uncertainty, self.method, self.data)
                if gain >= best_gain:
                    best_gain, best_question = gain, question
        try:
            self.features_done.append(best_question.column) #adds used column to the feature done list
        except:
            None
        return best_gain, best_question


    def is_feature_done(self, col):
        if not self.resampling_status:
            if col in self.features_done:
                return 1
            else:
                return 0
        else:
            return 0


    def build_tree(self, rows):
        # self.counter += 1
        # if self.counter >= 2**(self.max_depth+1) - 1:
        #     return Leaf(rows, self.data)
        gain, question = self.find_best_split(rows)
        if gain == 0:
            return Leaf(rows, self.data)

        true_rows, false_rows = self.partition(rows, question)
        true_branch = self.build_tree(true_rows)
        false_branch = self.build_tree(false_rows)
        return Decision_Node(question, true_branch, false_branch)

    def construct_tree(self, node, spacing=""):
        if isinstance(node, Leaf) == 1:
            print(spacing + "Predict", node.predictions)
            return
        print(spacing + str(node.question))
        print(spacing + '--> True:')
        self.construct_tree(node.true_branch, spacing + "  ")
        print(spacing + '--> False:')
        self.construct_tree(node.false_branch, spacing + "  ")

    def classify(self, row, node):
        # Base case: we've reached a leaf
        if isinstance(node, Leaf):
            return node.predictions, node.predicted_value
        # node.predictions is the dictionary of predictions at that leaf
        # node.predicted_value is the predicted value, basically the most probable prediction from argmax(node.predictions)
        if node.question.match(row):
            return self.classify(row, node.true_branch)
        else:
            return self.classify(row, node.false_branch)


    def predict(self, data_row, node): #data_row is a list of the test data.
        if isinstance(node, Leaf):
            return node.predictions, node.predicted_value

        if node.question.predict(data_row):
            return self.predict(data_row, node.true_branch)
        else:
            return self.predict(data_row, node.false_branch)

    def print_leaf(self, counts):
        """A nicer way to print the predictions at a leaf."""
        total = sum(counts.values()) * 1.0
        probs = {}
        for lbl in counts.keys():
            probs[lbl] = str(int(counts[lbl] / total * 100)) + "%"
        return probs


    def train(self):
        # learns data tree
        self.tree = self.build_tree(self.main_rows)
        print('Tree built successfully')
        return self.tree

    def print_tree(self):
        self.construct_tree(self.tree, "")


