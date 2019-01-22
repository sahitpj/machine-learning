from dtree import  decisionTree
import random, math, operator


class randomForest(decisionTree):
    def __init__(self, type_, method, data, n_fold, resampling_status=0, max_depth=2):
        super().__init__(type_, method, data, resampling_status, max_depth)
        self.n_fold = n_fold
        self.forest = []


    def find_best_split(self, rows, cols):
        best_gain = 0  
        best_question = None  
        current_uncertainty = gini(rows, self.data)

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
        if self.current_depth == self.max_depth:
            return 
        gain, question = self.find_best_split(rows, cols)
        if gain == 0:
            return Leaf(rows, self.data)

        true_rows, false_rows = self.partition(rows, question)
        true_branch = self.build_tree(true_rows)
        false_branch = self.build_tree(false_rows)
        self.current_depth += 1
        return Decision_Node(question, true_branch, false_branch)


    
    def generate_forest(self, rows):
        trees = []
        for i in xrange(self.n_fold):
            t = int(math.floor(self.features**0.5))
            cols = []       #generate features set for each forest
            rows = []       #generate sample set for each forest
            count = 0
            while count < t:
                index_1 = random.randrange(self.features)
                index_2 = random.randrange(self.samples)
                if index not in cols:
                    cols.append(index_1)
                    count += 1
                rows.append(index_2)
            tree = self.build_tree(rows, cols)
            trees.append(tree)
        self.forest = trees


    def construct_forest(self):
        for i in xrange(len(self.forest)):
            print '### Tree No  '+str(i+1)+'######'
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