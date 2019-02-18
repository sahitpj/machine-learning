import torch 
from errors import MSE_torch, SSE_torch

class coordinateDescent(object):
    def __init__(self, X, Y, **kwargs):
        self.X = X
        self.Y = Y
        self.theta = None
        self.features = X.shape[1]
        self.samples = X.shape[0]
        try:
            self.iterations = kwargs['iterations']
        except:
            self.iterations = 100

        
    def training_loss(self, theta):
        return self.MSE(theta)

    def initialise_theta(self):
        theta = torch.rand(self.features, 1)
        self.theta = theta.double()
        return theta

    def update_theta(self):
        for i in xrange(self.features):
            self.update_theta_indi(i)
        return self.theta
    
    def update_theta_indi(self, i):
        current_theta = self.theta
        r = torch.ones(self.features, 1).double()
        r[i, 0] = 0
        rho = torch.sum((self.Y-self.X.mm(current_theta*r))*torch.reshape(self.X[:, i], (self.samples,1)))
        z = torch.sum(self.X[:, i]**2)
        current_theta[i, 0] = rho/z
        self.theta = current_theta
        return current_theta

    def train(self):
        self.initialise_theta()
        error = 0.00001
        for i in xrange(self.iterations):
            print ''
            theta = self.update_theta()
            print 'Iteration -  '+ str(i+1)
            print ''
            print self.MSE(theta)
            if self.MSE(theta) <= error:
                break
        print '### Training complete'
        

    def predict(self, X_test):
        '''
        Assumes thats X_test has the bias festure added to it, however an assert is done
        '''
        assert(X_test.shape[1] == self.theta.shape[0])
        predictions = X_test.mm(self.theta)
        self.predictions = predictions
        return predictions

    def accuracy(self, Y_test, metric):
        if metric == 'MSE':
            return MSE_torch(self. predictions, Y_test)
        elif metric == 'SSE':
            return SSE_torch(self. predictions, Y_test)

    def MSE(self, theta):
        Yy = self.X.mm(theta)
        assert(self.Y.shape[0] == Yy.shape[0])
        return torch.sum((self.Y-Yy)**2)/self.samples