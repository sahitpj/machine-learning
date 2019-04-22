from .utils import class_counts, is_numeric

class Decision_Node:
    """A Decision Node asks a question.

    This holds a reference to the question, and to the two child nodes.
    """

    def __init__(self,
                 question,
                 true_branch,
                 false_branch):
        self.question = question
        self.true_branch = true_branch
        self.false_branch = false_branch

    def show(self):
        print(self.question)


class Leaf:
    def __init__(self, rows, data_df):
        self.predictions = class_counts(rows, data_df)
        index = None
        count = 0
        for key in self.predictions.keys():
            if self.predictions[key] > count:
                count = self.predictions[key]
                index = key
        self.predicted_value = index


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

    def predict(self, data_row):
        val = data_row[self.column]
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