import numpy as np 
import pandas as pd 
from autograd import grad


class SupportVectorMachine_D(object):
    def __init__(self, X, y, alpha, lambda_, epochs_num):
        "remember to add a bias column to the dataset"
        self.X_train = X 
        self.features = X.shape[1]
        self.samples = X.shape[0]
        self.y_train = y
        self.lambda_ = lambda_
        self.alpha = alpha
        self.epochs_num = epochs_num
        self.params = None
        self.initialise_params()
        self.gradient_func = grad(self.hinge_loss)

    def hinge_loss(self, w):
        loss = 0.5*(np.sum(w**2))
        for i in range(self.samples):
            loss += self.alphe*max(0, 1 - self.y_train[i]*(self.X_train[i].dot(self.params)))
        return loss

    def initialise_params(self):
        params = np.random.rand(self.features, 1)
        self.params = params
        
    
    def epoch_update(self):
        current_params = self.params
        current_params -= self.gradient_func(current_params)*self.lambda_
        self.params = current_params
            

    def train(self):
        for i in range(self.epochs_num):
            self.epoch_update()
        print("training complete")
        