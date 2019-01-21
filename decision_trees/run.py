from main import decisionTree

filepath = 'data.csv'

dt = decisionTree('classification', 10, 'Yes')
dt.import_data(filepath, 'Play')

X_train, Y_train, X_val, Y_val, X_test, Y_test = dt.shuffle(0.6, 0.2, 0)

tree = dt.train_tree(X_train, Y_train)
print tree