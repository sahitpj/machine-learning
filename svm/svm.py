import numpy as np 
import pandas as pd 
import cvxpy as cp 


class SupportVectorMachine(object):
    def __init__(self, X, y):
        self.X_train = X 
        self.features = X.shape[1]
        self.samples = X.shape[0]
        self.y_train = y
        self.params = None
        self.c = None
        self.optimal_params = list()
        self.optimal_c = None
        self.inialise_params()

    def inialise_params(self):
        params = list()
        for i in range(self.features):
            params.append(cp.Variable())
        self.params = params
        self.c = cp.Variable()


    def get_constraints(self):
        constraints = list()
        for i in range(self.samples):
            j = self.y_train[i] * (self.X_train[i].dot(self.params) + self.c) - 1
            constraints.append(j >= 0)
        return constraints

    def train_soft(self):
        constraints = self.get_constraints()
        W = sum([p*p for p in self.params])
        obj = cp.Minimize(0.5*W)

        prob = cp.Problem(obj, constraints)
        prob.solve()  # Returns the optimal value.
        print("status:", prob.status)
        print("optimal value", prob.value)

        for i in range(self.features):
            self.optimal_params.append(self.params[i].value)
        self.optimal_c = self.c.value
        print("Training complete!")

    def predict(X_test):
        "Assuming the X_test value is a numpy array"
        y = X_test.dot(self.optimal_params) + self.optimal_c
        return (y >= 0)

