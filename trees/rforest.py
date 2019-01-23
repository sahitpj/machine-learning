from dtree import  decisionTree
from utils import class_counts, unique_vals, is_numeric
from entropy import gini, info_gain, entropy, std
from nodes import Decision_Node, Leaf, Question

import random, math, operator
from multiprocessing import Process, Queue


class randomForest(decisionTree):
    def __init__(self, type_, method, data, n_fold, resampling_status=0, max_depth=2):
        super(randomForest, self).__init__(type_, method, data, resampling_status, max_depth)
        self.n_fold = n_fold
        self.forest = []


    def find_best_split(self, rows, cols):
        best_gain = 0  
        best_question = None
        current_uncertainty = None
        if self.type == 'classification':
            if self.method == 'gini':
                current_uncertainty = gini(rows, self.data)
            elif self.method == 'entropy':
                current_uncertainty = entropy(rows, self.data)
        elif self.type == 'regression':
            current_uncertainty = std(rows, self.data)
        for col in cols: 
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


    def build_tree(self, rows, cols):
        # print self.current_depth
        if self.current_depth == self.max_depth:
            return 
        gain, question = self.find_best_split(rows, cols)
        if gain == 0:
            return Leaf(rows, self.data)

        true_rows, false_rows = self.partition(rows, question)
        true_branch = self.build_tree(true_rows, cols)
        false_branch = self.build_tree(false_rows, cols)
        self.current_depth += 1
        return Decision_Node(question, true_branch, false_branch)


    def indi_tree(self, q): #individual tree building with construction of fatures and samples
        feature_count = int(math.floor(self.features**0.5)) #number of features for each tree
        sample_count = int(self.samples)/feature_count #number of samples for each tree (re-sampling allowed)
        cols = []       #generate features set for each forest
        rows = []       #generate sample set for each forest
        count_1 = 0
        count_2 = 0
        while count_1 < feature_count:
            index_1 = random.randrange(self.features)
            if index_1 not in cols:
                cols.append(index_1)
                count_1 += 1
        while count_2 < sample_count:
            index_1 = random.randrange(self.samples)
            rows.append(index_1)
            count_2 += 1
        tree = self.build_tree(rows, cols)
        q.put(tree)


    
    def train(self): #multiprocessing enabled forest generation
        trees = []
        for i in xrange(self.n_fold):
            q = Queue()
            p = Process(target=self.indi_tree, args=(q, ))
            p.start()
            trees.append(q.get())
            p.join()
        self.forest = trees
        print 'forest successfully built'
        return self.forest


    def construct_forest(self):
        for i in xrange(len(self.forest)):
            print '### Tree No  '+str(i+1)+'  ######'
            print ' '
            print ' '
            self.construct_f_tree(i, spacing="")
            print ' '
            print ' '


    def construct_f_tree(self, tree_no, spacing=""):
        self.construct_tree(self.forest[tree_no], spacing="")

    
    def predict_f(self, data_row):
        predictions_forest = {}
        for i in xrange(len(self.forest)):
            prediction_list, prediction = self.predict(data_row, self.forest[i])
            try:
                predictions_forest[prediction] += 1
            except:
                predictions_forest[prediction] = 1
        return max(predictions_forest.iteritems(), key=operator.itemgetter(1))[0]