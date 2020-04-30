# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 21:42:28 2020

@author: Dell
"""

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import warnings
warnings.filterwarnings("ignore")

def sigmoid_der(x):
    w, h = x.shape
    z = 1/(1+np.exp(-x))
    z = z*(1-z)
    
    assert (z.shape == (w, h))
    return (z)

def relu_der(x):
    w, h = x.shape
    x[x<=0] = 0
    x[x>0] = 1
    
    assert (x.shape == (w, h))
    return x

def sigmoid_backward(dA, cache):
    dZ = np.multiply(dA, sigmoid_der(cache))
    return dZ

def relu_backward(dA, cache):
    dZ = np.multiply(dA, relu_der(cache))
    return dZ
    

def linear_backward(dZ, cache):
    """
    dZ -> Gradient of cost w.r.t linear output
    cache -> (A_prev, W, b)
    dW -> gradient of cost w.r.t W
    db -> gradient of cost w.r.t b
    dA_prev -> gradient of cost w.r.t activation of previous layer
    """
    A_prev, W, b = cache
    
    m = A_prev.shape[1]
    t = np.dot(dZ, A_prev.T)
    dW = np.multiply(1/m, t)
    db = (1/m)*np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)
    return dA_prev, dW, db

def linear_activation_backward(dA, cache, activation):
    """
    dA -> post activation gradient
    cache -> (linear_cache, activation_cache)
    activation -> sigmoid or relu
    """
    linear_cache, activation_cache = cache
    if activation == 'relu':
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    elif activation == 'sigmoid':
        dZ = sigmoid_backward(dA, activation_cache)
        
        assert (dZ.shape == (1, 8000))
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    return dA_prev, dW, db

def L_model_backward(AL, Y, caches):
    grads = {}
    L = len(caches)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)
    dAL = -(np.divide(Y, AL) - np.divide(1-Y, 1-AL))
    assert (dAL.shape == AL.shape)
    # Lth layer
    current_cache = caches[L-1]
    
    grads["dA"+str(L-1)], grads["dW"+str(L)], grads["db"+str(L)] = linear_activation_backward(dAL, current_cache, activation='sigmoid')
    
    # loop from L-2 to 0
    for l in reversed(range(L-1)):
        # Relu -> Linear
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA"+str(l+1)], current_cache, activation='relu')
        grads["dA"+str(l)] = dA_prev_temp
        grads["dW"+str(l+1)] = dW_temp
        grads["db"+str(l+1)] = db_temp
    return grads


"""
from linear_activation_forward import L_model_forward
from parameter_initialize import para_init
from data_init import dat_init
from cost_func import compute_cost
X_train, y_train, X_test, y_test = dat_init()
layers = [11, 6, 6, 1]
par = para_init(layers, 0.01)
AL, caches = L_model_forward(X_train, par)
cost = compute_cost(AL, y_train)
print (cost)
gradients = L_model_backward(AL, y_train, caches)
"""
