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
    z = 1/(1+np.exp(-x))
    return z

def relu(x):
    return x * (x>0)

def lin_act_forward(A_prev, W, b, activation, v_mu, v_sigma, gamma, beta):
    """
    A_prev -> Activation from previous layer
    W -> Weights
    b -> Bias
    activation -> Sigmoid or Relu
    """
    b = 0.9
    if activation == 'sigmoid':
        Z, linear_cache = np.dot(W, A_prev)+b, (A_prev, W, b)
        Z_telda, cache = forward_prop(Z, gamma, beta)
        mu, sigma, Z_norm = cache
        v_mu = b*v_mu + (1-b)*mu
        v_sigma = b*v_sigma + (1-b)*sigma
        A, activation_cache = sigmoid(Z_telda), (Z, mu, sigma, Z_norm, gamma, beta)
        
    elif activation == 'relu':
        Z, linear_cache = np.dot(W, A_prev)+b, (A_prev, W, b)
        Z_telda, cache = forward_prop(Z, gamma, beta)
        mu, sigma, Z_norm = cache
        v_mu = b*v_mu + (1-b)*mu
        v_sigma = b*v_sigma + (1-b)*sigma
        A, activation_cache = relu(Z_telda), (Z, mu, sigma, Z_norm, gamma, beta)
        
    assert(A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, activation_cache)
    return A, cache, v_mu, v_sigma

def L_model_forward(X, parameters, b_par, v_mu, v_sigma):
    caches = []
    A = X
    L = len(parameters) // 2  # number of layers 
    for l in range(1, L):
        A_prev = A
        t_mu = v_mu[l]
        t_sigma = v_sigma[l]
        A, cache, a, b = lin_act_forward(A_prev, parameters["W"+str(l)], parameters["b"+str(l)],
                                                      activation = 'relu', v_mu=t_mu, v_sigma=t_sigma,
                                                      gamma=b_par["gamma"+str(l)],
                                                      beta=b_par["beta"+str(l)])
        v_mu[l] = a
        v_sigma[l] = b
        caches.append(cache)
    t_mu = v_mu[L]
    t_sigma = v_sigma[L]
    AL, cache, v_mu[L], v_sigma[L] = lin_act_forward(A, parameters["W"+str(L)], parameters["b"+str(L)],
                                                  activation = 'sigmoid', v_mu=t_mu, v_sigma=t_sigma,
                                                  gamma=b_par["gamma"+str(L)],
                                                  beta=b_par["beta"+str(L)])
    caches.append(cache)
    
    
    
    #assert (AL.shape == (y_train.shape[0], y_train.shape[1]))
    return AL, caches, v_mu, v_sigma

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
        A, cache = lin_act_forward(A_prev, parameters["W"+str(l)], parameters["b"+str(l)],
                                                      activation = 'relu')
        caches.append(cache)
    
    AL, cache = lin_act_forward(A, parameters["W"+str(L)], parameters["b"+str(L)],
                                                  activation = 'sigmoid')
    caches.append(cache)
    
    m = Y.shape[1]
    
    logprobs = np.multiply(-np.log(AL),Y) + np.multiply(-np.log(1 - AL), 1 - Y)
    cost = 1./m * np.sum(logprobs)
    cost = np.squeeze(cost)  # [[10]] --> 10
    assert (cost.shape == ())
    
    
    #assert (AL.shape == (y_train.shape[0], y_train.shape[1]))
    return cost, caches


