import numpy as np 
from scipy.stats import multivariate_normal
import operator

class GaussianNB(object):
    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.features = X_train.shape[1]
        self.samples = X_train[0]
        self.y_train = y_train
        # self.type_ = type_
        self.params = dict()
        self.classes = None
        self.find_params()
        # if self.type_ != 'continous':
        #     raise NotImplemented()


    def find_params(self):
        classes = np.unique(np.array(self.y_train))
        self.classes = classes
        for i in classes:
            inds = np.where(self.y_train == i)[0]
            self.params[i] = dict()
            self.params[i]['mean'] = np.mean(self.X_train[inds], axis = 0)
            self.params[i]['var'] = np.var(self.X_train[inds], axis = 0)
            self.params[i]['size'] = len(inds)


    def test_prob(self, X_test, class_):
        x = np.reshape(X_test, (self.features, ))
        prob = self.params[class_]['size']/self.samples
        for i in range(self.features):
            prob *= multivariate_normal.pdf(x[i], mean=self.params[class_]['mean'][i], cov=self.params[class_]['var'][i])
        return prob

    def predict(X_test):
        probabilities = dict()
        for i in classes:
            probabilities[i] = self.test_prob(X_test, i)
        return max(probabilities.items(), key=operator.itemgetter(1))[0]






