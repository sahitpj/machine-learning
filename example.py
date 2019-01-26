from trees import decisionTree, randomForest, ADABoost
from utils import import_data, K_fold_split, normal_split, transform_pd, transform_question


import pandas as pd

filepath = 'datasets/tennis.csv'

data = import_data(filepath)

# dt = decisionTree('classification', 'gini', data, 0, 100)
# tree = dt.train()
# dt.construct_tree(tree, "")

filepath_2 = 'datasets/iris.csv'
iris_data = import_data(filepath_2)



train, validate, test = normal_split(iris_data, 0.7)
train_data, validate_data, test_data = transform_pd(train, iris_data), transform_pd(validate, iris_data), transform_pd(test, iris_data)
train_validate = transform_pd(train+validate, iris_data)
# rf = randomForest('classification', 'gini', train_validate, 20)

# forest = rf.train()
# rf.construct_forest()

# count = 0.0
# for row in test:
#     count += (rf.predict_f(list(iris_data.iloc[row, :-1])) == iris_data.iloc[row, -1])
# print count/len(test)

# for i in xrange(n_folds, K_fold_data):
#     test = K_fold_data[i]
#     train = []
#     for j in xrange(n_folds):
#         if i != j:
#             train.extend(K_fold_data[j])
#     train_data = transform_pd(train, iris_data)
#     dt = decisionTree('classification', 'gini', train_data, 0, 100)
#     tree = dt.train()
#     count = 0.0
#     for row in test:
#         count += (dt.predict(transform_question(row, iris_data), tree)[-1] == iris_data.iloc[row, -1])
#     print count/len(test)

# n_folds = 5
# K_fold_data = K_fold_split(iris_data, n_folds)
# for i in xrange(n_folds):
#     test = K_fold_data[i]
#     train = []
#     for j in xrange(n_folds):
#         if i != j:
#             train.extend(K_fold_data[j])
#     train_data = transform_pd(train, iris_data)
#     rf = randomForest('classification', 'gini', train_validate, 5)
#     forest = rf.train()
#     count = 0.0
#     for row in test:
#         count += (rf.predict_f(list(iris_data.iloc[row, :-1])) == iris_data.iloc[row, -1])
#     print count/len(test)



ab = ADABoost('classification', 'gini', iris_data, 8)
ab.train()