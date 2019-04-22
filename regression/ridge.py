from .linear_torch import TorchGradientDescentAutogradRegression
from .linear_torch import TorchNormalEquationRegression
import torch, math, random

class normalEquationRidgeRegression(TorchNormalEquationRegression):
    def __init__(self, X, Y, lambda_):
        super(normalEquationRidgeRegression, self).__init__(X, Y)
        self.lambda_ = lambda_

    def train(self):
        p = torch.inverse(self.X.t().mm(self.X) + (self.lambda_**2)*torch.eye(self.X.shape[1]))
        k = p.mm(self.X.t())
        theta = k.mm(self.Y)
        assert(theta.shape[0] == self.features)
        assert(theta.shape[1] == 1)
        self.theta = theta
        return theta



class TorchridgeRegression(TorchGradientDescentAutogradRegression):
    def __init__(self,  X, Y, alpha, lambda_, **kwargs):
        super(ridgeRegression, self).__init__(X, Y, alpha, **kwargs)
        self.lambda_ = lambda_
        self.objective = None
        self.gradients = None

    
    def ForwardFunction(self):
        j = torch.ones(self.features, 1)
        j[0,0] = 0
        p = torch.mean((self.Y-self.X.mm(self.theta.double()))**2) + self.lambda_*(self.theta.t().mm(self.theta*j)) #Loss function forward function
        self.objective = p
        return p

    
