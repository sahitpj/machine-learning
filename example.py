from neuralnet import NeuralNetwork
import numpy as np 
from sklearn.model_selection import train_test_split
from utils.data import import_data, convertToOneHot

# x_t_d = [[1,2], [3,4], [4,5], [5,6]]
# y_t_d = [3,7,9,11]

# x_t_d = np.array(x_t_d)
# y_t_d = np.array(y_t_d)

# x_t_d = np.concatenate((x_t_d, np.ones((4,1))), axis=1)

# nn = NeuralNetwork([20, 5,1], ['relu', 'relu', 'relu'], x_t_d, y_t_d)

# nn.train(2, 1, 0.01)


# print('\n')
# print(nn.predict(x_t_d))




filepath2 = 'datasets/mnist.csv'
mnist_data = import_data(filepath2)

X = mnist_data.iloc[:, 1:]
Y = mnist_data.iloc[:, 0]


Y = convertToOneHot(Y.values)

X_train, X_test, y_train, y_test = train_test_split(X.values/255, Y, test_size=0.2, shuffle=True)


nn = NeuralNetwork([1000, 512, 10], ['softmax', 'softmax', 'softmax'], X_train, y_train, 'classification')

nn.train(10,1000, 0.5)


preds = nn.predict(X_test)

print(preds) 
count = 0
for i in range(preds.shape[0]):
    r = np.argmax(preds[i])
    if 1 == y_test[i][r]:
        count += 1
print(count/preds.shape[0])
