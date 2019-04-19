import numpy as np

def getActivationFunction(name):
    if name == 'sigmoid':
        return lambda x : 1/(1+np.exp(-x))
    elif name == 'linear':
        return lambda x : x
    elif name == 'relu':
        def relu(x):
            y = np.copy(x)
            y[y<0] = 0
            return y
        return relu
    elif name == 'softmax':
        def softmax(x):
            y = np.copy(x)
            y = np.exp(y)
            s = np.sum(y, axis=1)
            # print(x, y, s)
            for i in range(s.shape[0]):
                y[i] = y[i]/s[i]
            return y
        return softmax
    else:
        print('Unknown activation function. linear is used')
        return lambda x: x


def getDerivitiveActivationFunction(name):
    if name == 'sigmoid':
        sig = lambda x : 1/(1+np.exp(-x))
        return lambda x : sig(x)*(1-sig(x)) 
    elif name == 'linear':
        return lambda x: 1
    elif name == 'relu':
        def relu_diff(x):
            y = np.copy(x)
            y[y>=0] = 1
            y[y<0] = 0
            return y
        return relu_diff
    elif name == 'softmax':
        def softmax(x):
            y = np.copy(x)
            y = np.exp(y)
            s = np.sum(y, axis=1)
            for i in range(s.shape[0]):
                y[i] = y[i]/s[i]
            return y
        return lambda x : softmax(x)*(1 - softmax(x))
    else:
        print('Unknown activation function. linear is used')
        return lambda x: 1
