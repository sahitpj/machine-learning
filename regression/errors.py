import numpy as np
import torch

def MSE(Y_predict, Y):
    assert(Y_predict.shape[0] == Y.shape[0])
    return np.sum((Y_predict-Y)**2)/Y.shape[0]

def MSE_torch(Y_predict, Y):
    assert(Y_predict.shape[0] == Y.shape[0])
    return torch.sum((Y_predict-Y)**2)/Y.shape[0]

def SSE(Y_predict, Y):
    assert(Y_predict.shape[0] == Y.shape[0])
    return np.sum((Y_predict-Y)**2)

def SSE_torch(Y_predict, Y):
    assert(Y_predict.shape[0] == Y.shape[0])
    return torch.sum((Y_predict-Y)**2)