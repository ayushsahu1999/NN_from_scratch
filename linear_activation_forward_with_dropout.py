# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 11:51:52 2020

@author: Dell
"""

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


def lin_act_forward(A_prev, W, b, b_par, gamma, beta, batch_norm, D_prev, keep_prob, l, activation):
    """
    A_prev -> Activation from previous layer
    W -> Weights
    b -> Bias
    activation -> Sigmoid or Relu
    """
    
    if activation == 'sigmoid':
        Z = np.dot(W, A_prev) + b
        if batch_norm:
            Z_telda, cache, b_par = forward_prop(Z, gamma, beta, b_par, l)
            Z_norm, Z_centered, std, gamma = cache

            A, activation_cache = sigmoid(Z_telda), (Z, Z_norm, Z_centered, std, gamma)
            
        else:
            A, activation_cache = sigmoid(Z), Z
            
        D = D_prev
        linear_cache = (A_prev, W, b, D_prev)
    elif activation == 'relu':
        Z = np.dot(W, A_prev) + b
        if batch_norm:
            Z_telda, cache, b_par = forward_prop(Z, gamma, beta, b_par, l)
            Z_norm, Z_centered, std, gamma = cache
        
            A, activation_cache = relu(Z_telda), (Z, Z_norm, Z_centered, std, gamma)
            
        else:
            A, activation_cache = relu(Z), Z
            
            
        linear_cache = (A_prev, W, b, D_prev)
        D = np.random.rand(A.shape[0], A.shape[1])
        D = (D < keep_prob).astype(int)
        A = np.multiply(A, D)
        A = A / keep_prob

    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, activation_cache)
    return A, cache, b_par, D


def L_model_forward(X, parameters, b_par, batch_norm, keep_prob=0.5):
    caches = []
    A = X
    L = len(parameters) // 2  # number of layers 
    D = np.random.rand(A.shape[0], A.shape[1])
    D = (D < keep_prob).astype(int)
    for l in range(1, L):
        A_prev = A
        D_prev = D
#        print (b_par["gamma"+str(l)].shape)
        gamma = b_par["gamma"+str(l)]
        beta = b_par["beta"+str(l)]
        
        
        A, cache, b_par, D = lin_act_forward(A_prev, parameters["W" + str(l)], parameters["b" + str(l)], 
                                        b_par, gamma, beta, batch_norm, D_prev, keep_prob, l, activation='relu')
        if b_par['mode'] == 'train':
            caches.append(cache)
        
#    print (b_par["gamma"+str(L)].shape)
    gamma = b_par["gamma"+str(L)]
    beta = b_par["beta"+str(L)]
    D_prev = D
    
    AL, cache, b_par, D = lin_act_forward(A, parameters["W" + str(L)], parameters["b" + str(L)],
                                                     b_par, gamma, beta, batch_norm, D_prev, keep_prob, L, activation='sigmoid')
    
    if b_par['mode'] == 'train':
        caches.append(cache)

    # assert (AL.shape == (y_train.shape[0], y_train.shape[1]))
    return AL, caches, b_par