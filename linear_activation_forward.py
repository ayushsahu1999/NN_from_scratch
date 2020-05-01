# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 19:22:35 2020

@author: Dell
"""

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from data_init import dat_init
X_train, y_train, X_test, y_test = dat_init()
# Sigmoid Function
def sigmoid(x):
    z = 1/(1+np.exp(-x))
    return z

def relu(x):
    return x * (x>0)

def lin_act_forward(A_prev, W, b, activation):
    """
    A_prev -> Activation from previous layer
    W -> Weights
    b -> Bias
    activation -> Sigmoid or Relu
    """
    if activation == 'sigmoid':
        Z, linear_cache = np.dot(W, A_prev)+b, (A_prev, W, b)
        A, activation_cache = sigmoid(Z), Z
        
    elif activation == 'relu':
        Z, linear_cache = np.dot(W, A_prev)+b, (A_prev, W, b)
        A, activation_cache = relu(Z), Z
        
    assert(A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, activation_cache)
    return A, cache

def L_model_forward(X, parameters):
    caches = []
    A = X
    L = len(parameters) // 2  # number of layers 
    for l in range(1, L):
        A_prev = A
        A, cache = lin_act_forward(A_prev, parameters["W"+str(l)], parameters["b"+str(l)],
                                                      activation = 'relu')
        caches.append(cache)
    
    AL, cache = lin_act_forward(A, parameters["W"+str(L)], parameters["b"+str(L)],
                                                  activation = 'sigmoid')
    caches.append(cache)
    
    #assert (AL.shape == (y_train.shape[0], y_train.shape[1]))
    return AL, caches

#AL, caches = L_model_forward(X_train, par)

