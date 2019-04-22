from .linear_torch import TorchGradientDescentAutogradRegression
import torch, math, random

class stochasticGradientDescent(TorchGradientDescentAutogradRegression):
    def __init__(self,  X, Y, alpha, **kwargs):
        super(stochasticGradientDescent, self).__init__(X, Y, alpha, **kwargs)
        try:
            h = kwargs['batch_size']
            self.iterations = int(self.Y.shape[0])/h
            self.batch_size = int(self.Y.shape[0])/self.iterations
        except:
            self.iterations = int(self.Y.shape[0])
            self.batch_size = 1
        try:
            self.epochs_no = kwargs['epochs_no']
        except:
            self.epochs_no = 1
        self.batches = None

    def assign_batchs(self):
        r = range(int(self.Y.shape[0]))
        random.shuffle(r, random.random)
        batches = list()
        for i in xrange(self.iterations):
            batches.append(r[i:i+self.batch_size])
        self.batches = batches
        return batches

    def ForwardFunction(self, i):
        X = self.X[self.batches[i]]
        Y = self.Y[self.batches[i]]
        p = torch.mean((Y-X.mm(self.theta.double()))**2) #Loss function forward function
        self.objective = p
        return p

    def get_grads(self, i):
        self.initialise_theta()
        k = self.ForwardFunction(i)
        self.objective.backward()
        self.gradients = self.theta.grad
        return self.gradients

    def epoch(self):
        for i in xrange(self.iterations):
            self.update_theta(i)
        return self.theta

    def update_theta(self, i):
        h = self.get_grads(i)
        current_theta = self.theta.clone() #cloing theta so that we don't update in-place values
        current_theta -= self.gradients*self.alpha
        self.theta = current_theta
        return current_theta

    def train(self):
        self.initialise_theta()
        error = 0.0001
        for i in xrange(self.epochs_no):
            self.assign_batchs()
            print('')
            theta = self.epoch().double()
            print('Epoch -  '+ str(i+1))
            print('')
            return theta
            print(self.MSE(theta))
            if self.MSE(theta) <= error:
                break
        print('### Training complete')