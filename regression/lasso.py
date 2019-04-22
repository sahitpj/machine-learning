import torch 
from .linear_np import gradientDescentAutogradRegression
from autograd import grad

class coordinateDescentLASSO(object):
    def __init__(self, X, Y, lambda_, **kwargs):
        super(coordinateDescentLASSO, self).__init__(X, Y, **kwargs)
        self.lambda_ = lambda_

    def update_theta_indi(self, i):
        current_theta = self.theta
        r = torch.ones(self.features, 1).double()
        r[i, 0] = 0
        rho = torch.sum((self.Y-self.X.mm(current_theta*r))*torch.reshape(self.X[:, i], (self.samples,1)))
        z = torch.sum(self.X[:, i]**2)
        d = self.lambda_**2/2
        if rho < -d:
            current_theta[i, 0] = (rho+d)/z
        elif rho > d:
            current_theta[i, 0] = (rho-d)/z
        else:
            current_theta[i, 0] = 0
        self.theta = current_theta
        return current_theta


class coordinateDescentLASSOAutoGrad(gradientDescentAutogradRegression):
    def __init__(self, X, Y, alpha, lambda_, **kwargs):
        super(gradientDescentAutogradRegression, self).__init__(X, Y, alpha, **kwargs)
        self.lambda_ = lambda_


    def training_loss(self, theta):
        return self.MSE(theta) + (self.lambda_**2)*np.sum(abs(theta))