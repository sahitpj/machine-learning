from .layer import Dense 
from .activations import getActivationFunction, getDerivitiveActivationFunction


class Model(object):
    def __init__(self, X_train, y_train):
        self.layers = list()
        self.X_train = X_train
        self.y_train = y_train
        self.samples = self.X_train.shape[0]
        self.features = self.X_train.shape[1]
        self.input_size = self.features
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate

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
        for layer in self.layers:
            a_func = getActivationFunction(layer.activation_function)
            z_s.append(a.dot(layer.parameters))
            a = a_func(z_s[-1])
            a_s.append(a)
        return z_s, a_s


    def backward_prop(self, y, z_s, a_s):
        dw = list()
        ad_func = getDerivitiveActivationFunction(self.layers[-1].activation_function)
        deltas = [(y-a_s[-1]) * ad_func(z_s[-1])]

        for i in range(len(self.layers)-1):
            ad_func = getDerivitiveActivationFunction(self.layers[-i-2].activation_function)
            deltas = [ deltas[0].dot(self.layers[-i-1].parameters.T) * ad_func(z_s[-i-2]) ] + deltas

        dw = [d.dot(a_s[i].T)/self.batch_size for i,d in enumerate(deltas)]

        return dw



    
    def train(self):
        for e in range(self.epochs):
            i = 0
            for batch in range(int(self.samples/self.batch_size)):
                X_batch = self.X_train[i:i+batch_size]
                y_batch = self.y_train[i:, i+batch_size]
                i += batch_size

                z_s, a_s = self.forward_prop(X_batch)

                dw = self.backward_prop(y_batch, z_s, a_s)
                for layer_num in range(len(self.layers)):
                    self.layers[layer_num].parameters += self.learning_rate*dw[layer_num]

                print("epoch number: {} , batch number: {} --- loss = {}".format(e+1, batch+1 , np.linalg.norm(a_s[-1]-y_batch) ))

    





