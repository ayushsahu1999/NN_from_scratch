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
from batch_norm import forward_prop, batch_norm_init
from parameter_initialize import para_init

X_train, y_train, X_test, y_test, X_val, y_val = dat_init()


# Sigmoid Function
def sigmoid(x):
    z = 1 / (1 + np.exp(-x))
    return z


def relu(x):
    return x * (x > 0)


def lin_act_forward(A_prev, W, b, b_par, gamma, beta, batch_norm, l, activation):
    """
    A_prev -> Activation from previous layer
    W -> Weights
    b -> Bias
    activation -> Sigmoid or Relu
    """
    
    if activation == 'sigmoid':
        Z, linear_cache = np.dot(W, A_prev) + b, (A_prev, W, b)
#        if batch_norm:
#            Z_telda, cache, b_par = forward_prop(Z, gamma, beta, b_par, l)
#            Z_norm, Z_centered, std, gamma = cache
#
#            A, activation_cache = sigmoid(Z_telda), (Z, Z_norm, Z_centered, std, gamma)
#        else:
        A, activation_cache = sigmoid(Z), Z

    elif activation == 'relu':
        Z, linear_cache = np.dot(W, A_prev) + b, (A_prev, W, b)
        if batch_norm:
            Z_telda, cache, b_par = forward_prop(Z, gamma, beta, b_par, l)
            Z_norm, Z_centered, std, gamma = cache
            
            A, activation_cache = relu(Z_telda), (Z, Z_norm, Z_centered, std, gamma)
        else:
            A, activation_cache = relu(Z), Z

    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, activation_cache)
    return A, cache, b_par


def L_model_forward(X, parameters, b_par, batch_norm):
    caches = []
    A = X
    L = len(parameters) // 2  # number of layers 
    for l in range(1, L):
        A_prev = A
#        print (b_par["gamma"+str(l)].shape)
        gamma = b_par["gamma"+str(l)]
        beta = b_par["beta"+str(l)]
        
        A, cache, b_par = lin_act_forward(A_prev, parameters["W" + str(l)], parameters["b" + str(l)], 
                                        b_par, gamma, beta, batch_norm, l, activation='relu')
        if b_par['mode'] == 'train':
            caches.append(cache)
        
#    print (b_par["gamma"+str(L)].shape)
    gamma = b_par["gamma"+str(L)]
    beta = b_par["beta"+str(L)]
    
    AL, cache, b_par = lin_act_forward(A, parameters["W" + str(L)], parameters["b" + str(L)],
                                                     b_par, gamma, beta, batch_norm, L, activation='sigmoid')
    
    if b_par['mode'] == 'train':
        caches.append(cache)

    # assert (AL.shape == (y_train.shape[0], y_train.shape[1]))
    return AL, caches, b_par


'''
layers_dims = [11, 20, 10, 10, 1]
par = para_init(layers_dims)
b_par = batch_norm_init(layers_dims)
AL, caches = L_model_forward(X_train, par, b_par)
'''


def forward_prop_check(X, Y, parameters):
    caches = []
    A = X
    L = len(parameters) // 2  # number of layers 
    for l in range(1, L):
        A_prev = A
        A, cache = lin_act_forward(A_prev, parameters["W" + str(l)], parameters["b" + str(l)],
                                   activation='relu')
        caches.append(cache)

    AL, cache = lin_act_forward(A, parameters["W" + str(L)], parameters["b" + str(L)],
                                activation='sigmoid')
    caches.append(cache)

    m = Y.shape[1]

    logprobs = np.multiply(-np.log(AL), Y) + np.multiply(-np.log(1 - AL), 1 - Y)
    cost = 1. / m * np.sum(logprobs)
    cost = np.squeeze(cost)  # [[10]] --> 10
    assert (cost.shape == ())

    # assert (AL.shape == (y_train.shape[0], y_train.shape[1]))
    return cost, caches
