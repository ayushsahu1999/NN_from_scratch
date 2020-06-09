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

def sigmoid_backward(dA, cache, batch_norm):
    if batch_norm:
        Z, _, a, b, g = cache
    else:
        Z = cache
    
    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)
    
    assert (dZ.shape == Z.shape)
    
    return dZ

def relu_backward(dA, cache, batch_norm):
    if batch_norm:
        Z, _, a, b, g = cache
    else:
        Z = cache
    dZ = np.array(dA, copy=True) # just converting dz to a correct object.
    
    # When z <= 0, you should set dz to 0 as well. 
    dZ[Z <= 0] = 0
    
    assert (dZ.shape == Z.shape)
    
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

def linear_activation_backward(dA, cache, batch_norm, activation):
    """
    dA -> post activation gradient
    cache -> (linear_cache, activation_cache)
    activation -> sigmoid or relu
    """
    from batch_norm import forward_prop, batch_norm_init, back_prop
    linear_cache, activation_cache = cache
    if activation == 'relu':
        dZ = relu_backward(dA, activation_cache, batch_norm)
        #Z, Z_norm, Z_centered, std, gamma = activation_cache
        if batch_norm:
            dZ, dgamma, dbeta = back_prop(dZ, activation_cache)
        else:
            dgamma, dbeta = 0, 0
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    elif activation == 'sigmoid':
        dZ = sigmoid_backward(dA, activation_cache, batch_norm)
        #Z, mu, sigma, Z_norm, gamma, beta = activation_cache
        if batch_norm:
            dZ, dgamma, dbeta = back_prop(dZ, activation_cache)
        else:
            dgamma, dbeta = 0, 0
        #assert (dZ.shape == (1, 8000))
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    return dA_prev, dW, db, dgamma, dbeta

def L_model_backward(AL, Y, batch_norm, caches):
    from batch_norm import forward_prop, batch_norm_init, back_prop
    grads = {}
    L = len(caches)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)
    dAL = -(np.divide(Y, AL) - np.divide(1-Y, 1-AL))
    assert (dAL.shape == AL.shape)
    # Lth layer
    current_cache = caches[L-1]
    
    grads["dA"+str(L-1)], grads["dW"+str(L)], grads["db"+str(L)], grads["dgamma"+str(L)], grads["dbeta"+str(L)] = linear_activation_backward(dAL, current_cache, batch_norm, activation='sigmoid')
    
    # loop from L-2 to 0
    for l in reversed(range(L-1)):
        # Relu -> Linear
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp, dgamma_temp, dbeta_temp = linear_activation_backward(grads["dA"+str(l+1)], current_cache, batch_norm, activation='relu')
        grads["dA"+str(l)] = dA_prev_temp
        grads["dW"+str(l+1)] = dW_temp
        grads["db"+str(l+1)] = db_temp
        grads["dgamma"+str(l+1)] = dgamma_temp
        grads["dbeta"+str(l+1)] = dbeta_temp
    return grads


'''
from linear_activation_forward import L_model_forward
from parameter_initialize import para_init
from data_init import dat_init
from cost_func import compute_cost
from batch_norm import back_prop, batch_norm_init
X_train, y_train, X_test, y_test, X_val, y_val = dat_init()
layers_dims = [11, 20, 10, 10, 1]
par = para_init(layers_dims)
b_par = batch_norm_init(layers_dims)
AL, caches = L_model_forward(X_train, par, b_par)
cost = compute_cost(AL, y_train)
print (cost)
gradients = L_model_backward(AL, y_train, caches)
'''