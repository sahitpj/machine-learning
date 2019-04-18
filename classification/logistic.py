import autograd.numpy as np
from autograd import grad

def sigmoid(x):
    return 1/(1+np.exp(-x))


class LogisticRegression(object):
    def __init__(self, X_train, y_train, learning_rate):
        self.X_train = X_train
        self.y_train = y_train
        self.samples = self.X_train.shape[0]
        self.features = self.X_train.shape[1]
        self.learning_rate = learning_rate
        self.iterations = 100
        self.theta = None
        self.gradient_func = grad(self.training_loss)

    def training_loss(self, theta):
        return self.cost(theta)


    def initialise_theta(self):
        params = np.random.rand(self.features, 1)
        self.theta = params

    
    def cost(self, theta):
        Yy = sigmoid(self.X_train.dot(theta))
        cost = 0.
        for i in range(self.samples):
            cost -= (self.y_train[i]*np.log10(Yy[i]) + (1-self.y_train[i])*np.log10(1 - Yy[i]))
        return cost/self.samples

    def update_theta(self):
        current_theta = self.theta
        current_theta -= self.gradient_func(current_theta)*self.learning_rate
        self.theta = current_theta
        return current_theta

    def train(self):
        theta = None
        self.initialise_theta()
        error = 0.00001
        for i in xrange(self.iterations):
            print ''
            theta = self.update_theta()
            print 'Iteration -  '+ str(i+1)
            print ''
            if self.MSE(theta) <= error:
                break
        print '### Training complete'


    def predict(self, X_test):
        '''
        Assumes thats X_test has the bias festure added to it, however an assert is done
        '''
        assert(X_test.shape[1] == self.theta.shape[0])
        predictions = X_test.dot(self.theta)
        self.predictions = predictions
        return predictions