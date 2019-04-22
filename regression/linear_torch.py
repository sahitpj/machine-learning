import torch
from .errors import MSE, SSE, MSE_torch, SSE_torch
from autograd import grad

'''
Assumes all data to be in pytorch format
'''

class NotTrained(Exception):
    def __init__(self):
        self.message = "Theta has not been trained yet to predict"
    def __str__(self):
        return self.message


class TorchNormalEquationRegression(object):
    '''
    Uses matrix equation solving method to solve regression problem
    The following assumes that a bias column has been added to X (1st column), before sending it into the regressor.

    The following convention is following : rows -> samples ; columns -> features
    '''
    def __init__(self, X, Y):
        self.X = X #X is a numpy array of nxm 
        self.Y = Y #Y is a numpy array of nx1
        self.samples = X.shape[0]
        self.features = X.shape[1]
        self.theta = None

    def train(self):
        p = torch.inverse(self.X.t().mm(self.X))
        k = p.mm(self.X.t())
        theta = k.mm(self.Y)
        assert(theta.shape[0] == self.features)
        assert(theta.shape[1] == 1)
        self.theta = theta
        return theta

    def predict(self, X_test):
        if self.theta == None:
            raise NotTrained()
        else:
            return X_test.mm(self.theta) #return nx1 matrix of predicted values


class TorchGradientDescentRegression(object):
    '''
    Gradient descent using gradient descnet, by conventional methods
    '''
    def __init__(self, X, Y, alpha, **kwargs):
        self.X = X #X is a numpy array of nxm 
        self.Y = Y #Y is a numpy array of nx1
        self.samples = X.shape[0]
        self.features = X.shape[1]
        self.theta = None
        self.iterations = None
        self.alpha = alpha #learning rate
        try:
            self.iterations = kwargs['iterations']
        except:
            self.iterations = 100
        self.predictions = None

    def training_loss(self, theta):
        return self.MSE(theta)

    def initialise_theta(self):
        theta = torch.rand(self.features, 1)
        self.theta = theta
        return theta
    
    def update_theta(self):
        current_theta = self.theta
        gradients = torch.zeros((self.features, 1)).double()
        for i in xrange(self.features):
            if i == 0:
                j = torch.sum(self.Y-self.X.mm(current_theta))*(2.0/self.samples)
                gradients[0, 0] = -j
            else:
                j = torch.sum((self.Y-self.X.mm(current_theta))*torch.reshape(self.X[:, i], (self.samples,1)))*(2.0/self.samples)
                gradients[i, 0] = -j
        current_theta -= gradients*self.alpha
        self.theta = current_theta
        return current_theta

    def train(self):
        self.initialise_theta()
        error = 10
        for i in xrange(self.iterations):
            print('')
            theta = self.update_theta()
            print('Iteration -  '+ str(i+1))
            print('')
            if self.MSE(theta) <= error:
                break
        print('### Training complete')
        

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


    
class TorchGradientDescentAutogradRegression(TorchGradientDescentRegression):
    '''
    Using Pytorch's built in Autograd to compute the gradient. Recommended for computing gradients of complex networks
    '''
    def __init__(self,  X, Y, alpha, **kwargs):
        super(TorchGradientDescentAutogradRegression, self).__init__(X, Y, alpha, **kwargs)
        self.objective = None
        self.gradients = None

    def initialise_theta(self):
        try:
            theta = torch.tensor(self.theta, requires_grad=True) #using previous theta and adding gradient function
        except:
            theta = torch.rand(self.features, 1, requires_grad=True) #otherwise initialising theta to a random value
        self.theta = theta
        return theta    

    def ForwardFunction(self):
        p = torch.mean((self.Y-self.X.mm(self.theta.double()))**2) #Loss function forward function
        self.objective = p
        return p
    
    def get_grads(self):
        self.initialise_theta()
        k = self.ForwardFunction()
        self.objective.backward()
        self.gradients = self.theta.grad
#         self.theta = self.theta.clone()
        return self.gradients

    def update_theta(self):
        h = self.get_grads()
        current_theta = self.theta.clone() #cloing theta so that we don't update in-place values
        current_theta -= self.gradients*self.alpha
        self.theta = current_theta
        return current_theta

    def train(self):
        self.initialise_theta()
        error = 0.0001
        for i in xrange(self.iterations):
            print('')
            theta = self.update_theta()
            print('Iteration -  '+ str(i+1))
            print('')
            print(self.MSE(theta))
            if self.MSE(theta) <= error:
                break
        print('### Training complete')
    