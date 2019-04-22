from .layer import Dense 
from .activations import getActivationFunction, getDerivitiveActivationFunction
import numpy as np
import sys

class Model(object):
    def __init__(self, X_train, y_train, t):
        self.layers = list()
        self.X_train = X_train
        self.y_train = y_train
        self.samples = self.X_train.shape[0]
        self.features = self.X_train.shape[1]
        self.input_size = self.features
        self.batch_size = None
        self.epochs = None
        self.learning_rate = None
        self.t = t

    def add_layer(self, layer_type, size,  activation):
        if layer_type == 'dense':
            input_size = 0
            if len(self.layers) == 0:
                input_size = self.input_size
            else:
                input_size = self.layers[-1].output_size
            layer = Dense(input_size, size, activation)
            self.layers.append(layer)
        else:
            raise NotImplemented


    def forward_prop(self, x):
        a = x
        z_s = list()
        a_s = [x]
        count = 0
        for layer in self.layers:
            a_func = getActivationFunction(layer.activation_function)
            # print(layer.parameters)
            z_s.append(a.dot(layer.parameters))
            # print(z_s[-1], "z_s")
            # print(z_s[-1])
            a = a_func(z_s[-1])
            # print("------")
            # print(a)
            a_s.append(a)
            # print(a, "a")
        # print(a_s[-1])
        return z_s, a_s


    def backward_prop(self, y, z_s, a_s):
        dw = list()
        deltas = None
        ad_func = getDerivitiveActivationFunction(self.layers[-1].activation_function)
        if self.t == 'regression':
            deltas = [(y-a_s[-1]) * ad_func(z_s[-1])]
        elif self.t == 'classification':
            l = np.zeros(y.shape)
            for i in range(y.shape[0]):
                r = np.argmax(a_s[-1][i])
                l[i][r] = 1
            print(l)
            deltas = [(y-l) * ad_func(z_s[-1])]
        for i in range(len(self.layers)-1):
            ad_func = getDerivitiveActivationFunction(self.layers[-i-2].activation_function)
            deltas = [ deltas[0].dot(self.layers[-i-1].parameters.T) * ad_func(z_s[-i-2]) ] + deltas
        dw = [a_s[i].T.dot(d)/self.batch_size for i,d in enumerate(deltas)]
        return dw


    def train(self):
        for e in range(self.epochs):
            i = 0
            for batch in range(int(self.samples/self.batch_size)):
                X_batch = self.X_train[i:i + self.batch_size]
                y_batch = self.y_train[i: i + self.batch_size]
                i += self.batch_size

                z_s, a_s = self.forward_prop(X_batch)
                k = np.array(y_batch)
                k = k.reshape((self.batch_size, -1))
                dw = self.backward_prop(k, z_s, a_s)
                for layer_num in range(len(self.layers)):
                    self.layers[layer_num].parameters += self.learning_rate*dw[layer_num]

                # sys.stdout.flush()
                # print(np.sum(a_s[-1]))
                sys.stdout.write("\r epoch number: {} , batch number: {}/{} --- loss = {} ".format(e+1, batch+1 , int(self.samples/self.batch_size),  np.linalg.norm(a_s[-1]-y_batch)/self.batch_size ))
            sys.stdout.write("\n")

    





