import numpy as np 


class Layer(object):
    def __init__(self, input_size, output_size, activation_function):
        self.input_size = input_size
        self.output_size = output_size
        self.activation_function = activation_function
        self.parameters_size = (input_size, output_size)
        self.parameters = None
        

    def initialise_parameters(self):
        self.parameters = np.random.rand(self.input_size, self.output_size)
        return self.parameters

    
class Dense(Layer):
    def __init__(self, input_size, output_size, activation_function):
        super().__init__(input_size, output_size, activation_function)
        