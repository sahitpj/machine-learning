from .model import Model

class NeuralNetwork(object):
    def __init__(self, layers_list, activations_list, X_train, y_train):
        self.layers_list = layers_list
        self.activations_list = activations_list
        self.nn_length = len(self.layers_list)
        self.X_train = X_train
        self.y_train = y_train

        self.nn_model = Model(self.X_train, self.y_train)
        self.build_model()


    def build_model(self):
        for i in range(self.nn_length):
            self.nn_model.add_layer('dense', self.layers_list[i], self.activations_list[i])

        
    def train(self, epochs, batch_size, learning_rate):
        self.nn_model.epochs = epochs
        self.nn_model.batch_size = batch_size
        self.nn_model.learning_rate = learning_rate

        nn_model.train()