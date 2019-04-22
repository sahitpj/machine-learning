from .dtree import  decisionTree
from .utils import class_counts, unique_vals, is_numeric
from .entropy import gini, info_gain, entropy, std
from .nodes import Decision_Node, Leaf, Question

import math

def transform_question(row, df):
    df_row = df.iloc[row, :]
    return list(df_row)[:-1]

class ADABoost(decisionTree):
    def __init__(self, type_, method, data, iterations, resampling_status=0, max_depth=2):
        super(ADABoost, self).__init__(type_, method, data, resampling_status, max_depth)
        self.type = 'classification'
        self.max_depth = 1 #decision stumps
        self.counter = 0
        self.iterations = iterations
        self.weights = None
        self.model_stage_values = list()
        self.trees = list()

    def stage(self, error):
        return math.log((1-error)/error)

    def initialize_weights(self):
        weights = list()
        for i in xrange(self.samples):
            weights.append(1.0/self.samples)
        self.weights = weights
        # return weights

    def update_weights(self, terrors, stage_error):
        for i in xrange(self.samples):
            self.weights[i] = self.weights[i]*math.exp(stage_error*terrors[i])
        
    
    def iterate(self):
        tree = self.build_tree(self.main_rows)
        self.construct_tree(tree, "")
        count = 0 
        terror = list()
        for i in xrange(self.samples):
            p = (self.predict(transform_question(i, self.data), tree)[-1] != self.data.iloc[i, -1])
            count += p*self.weights[i]
            terror.append(p)
        self.trees.append(tree)
        error = count/sum(self.weights)
        if error == 0:
            return 0
        stage_error = self.stage(error)
        self.model_stage_values.append(stage_error)
        self.update_weights(terror, stage_error)
        return 1


    def build_tree(self, rows):
        self.counter += 1
        if self.counter > 1:
            return Leaf(rows, self.data)
        gain, question = self.find_best_split(rows)
        if gain == 0:
            return Leaf(rows, self.data)

        true_rows, false_rows = self.partition(rows, question)
        true_branch = self.build_tree(true_rows)
        false_branch = self.build_tree(false_rows)
        return Decision_Node(question, true_branch, false_branch)


    def train(self):
        # learns data tree
        self.initialize_weights()
        for i in xrange(self.iterations):
            if not self.iterate():
                break
            self.counter = 0
        print(self.model_stage_values)
        print('ADABoost done successfully')


    def predict_adaboost(self, data_row):
        k = class_counts(self.main_rows, self.data).keys()
        classes = {
            k[0]: 1,
            k[1]: -1
        }
        t = 0.0
        for i in xrange(len(self.trees)):
            tree = self.trees[i]
            a,prediction = self.predict(data_row, tree)
            t += classes[prediction]* self.model_stage_values[i]
        if t >= 0:
            return k[0]
        else:
            return k[1]

