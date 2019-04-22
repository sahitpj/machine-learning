import autograd.numpy as np
from .errors import MSE, SSE
from autograd import grad

'''
Assumes all data to be in numpy format
'''

class NotTrained(Exception):
    def __init__(self):
        self.message = "Theta has not been trained yet to predict"
    def __str__(self):
        return self.message


class normalEquationRegression(object):
    '''
    Uses Matrix multiplication to find the solution for the Regression problem
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
        p = np.linalg.inv(self.X.T.dot(self.X))
        k = p.dot(self.X.T)
        theta = k.dot(self.Y)
        assert(theta.shape[0] == self.features)
        assert(theta.shape[1] == 1)
        self.theta = theta
        return theta

    def predict(self, X_test):
        if self.theta == None:
            raise NotTrained()
        else:
            return X_test.dot(self.theta) #return nx1 matrix of predicted values


class gradientDescentRegression(object):
    '''
    Using Gradient descent to solve the regression problem
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
        theta = np.random.rand(self.features, 1)
        self.theta = theta
        return theta
    
    def update_theta(self):
        current_theta = self.theta
        gradients = np.zeros((self.features, 1))
        for i in xrange(self.features):
            if i == 0:
                j = np.sum(self.Y-self.X.dot(current_theta))*(2.0/self.samples)
                gradients[0, 0] = -j
            else:
                j = np.sum((self.Y-self.X.dot(current_theta))*np.reshape(self.X[:, i], (self.samples,1)))*(2.0/self.samples)
                gradients[i, 0] = -j
        current_theta -= gradients*self.alpha
        self.theta = current_theta
        return current_theta

    def train(self):
        theta = None
        self.initialise_theta()
        error = 0.00001
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
        predictions = X_test.dot(self.theta)
        self.predictions = predictions
        return predictions

    def accuracy(self, Y_test, metric):
        if metric == 'MSE':
            return MSE(self. predictions, Y_test)
        elif metric == 'SSE':
            return SSE(self. predictions, Y_test)

    
    def MSE(self, theta):
        Yy = self.X.dot(theta)
        assert(self.Y.shape[0] == Yy.shape[0])
        return np.sum((self.Y-Yy)**2)/self.samples


    
class gradientDescentAutogradRegression(gradientDescentRegression):
    '''
    Using the autograd function to find gradient for the Regression problem
    '''
    def __init__(self,  X, Y, alpha, **kwargs):
        super(gradientDescentAutogradRegression, self).__init__(X, Y, alpha, **kwargs)
        self.gradient_func = grad(self.training_loss)

    def update_theta(self):
        current_theta = self.theta
        current_theta -= self.gradient_func(current_theta)*self.alpha
        self.theta = current_theta
        return current_theta




            
    

        

    
