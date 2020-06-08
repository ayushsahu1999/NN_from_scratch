# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 12:00:23 2020

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
    Z, _, a, b, g = cache
    
    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)
    
    assert (dZ.shape == Z.shape)
    
    return dZ

def relu_backward(dA, cache):
    Z, _, a, b, g = cache
    dZ = np.array(dA, copy=True) # just converting dz to a correct object.
    
    # When z <= 0, you should set dz to 0 as well. 
    dZ[Z <= 0] = 0
    
    assert (dZ.shape == Z.shape)
    
    return dZ
    

def linear_backward(dZ, cache, keep_prob):
    """
    dZ -> Gradient of cost w.r.t linear output
    cache -> (A_prev, W, b)
    dW -> gradient of cost w.r.t W
    db -> gradient of cost w.r.t b
    dA_prev -> gradient of cost w.r.t activation of previous layer
    """
    A_prev, W, b, D = cache
    
    m = A_prev.shape[1]
    t = np.dot(dZ, A_prev.T)
    dW = np.multiply(1/m, t)
    db = (1/m)*np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)
    dA_prev = np.multiply(dA_prev, D)
    dA_prev = dA_prev / keep_prob
    return dA_prev, dW, db

def linear_activation_backward(dA, cache, keep_prob, activation):
    """
    dA -> post activation gradient
    cache -> (linear_cache, activation_cache)
    activation -> sigmoid or relu
    """
    from batch_norm import forward_prop, batch_norm_init, back_prop
    linear_cache, activation_cache = cache
    if activation == 'relu':
        dout = relu_backward(dA, activation_cache)
        #Z, Z_norm, Z_centered, std, gamma = activation_cache
        dZ, dgamma, dbeta = back_prop(dout, activation_cache)
        dZ = np.multiply(dA, np.int64(dA>0))
        
        
        
        dA_prev, dW, db = linear_backward(dZ, linear_cache, keep_prob)
    elif activation == 'sigmoid':
        dout = sigmoid_backward(dA, activation_cache)
        #Z, mu, sigma, Z_norm, gamma, beta = activation_cache
        dZ, dgamma, dbeta = back_prop(dout, activation_cache)
        #assert (dZ.shape == (1, 8000))
        dA_prev, dW, db = linear_backward(dZ, linear_cache, keep_prob)
        
        
    return dA_prev, dW, db, dgamma, dbeta

def L_model_backward(AL, Y, caches, keep_prob):
    from batch_norm import forward_prop, batch_norm_init, back_prop
    grads = {}
    L = len(caches)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)
    dAL = -(np.divide(Y, AL) - np.divide(1-Y, 1-AL))
    assert (dAL.shape == AL.shape)
    # Lth layer
    current_cache = caches[L-1]
    
    grads["dA"+str(L-1)], grads["dW"+str(L)], grads["db"+str(L)], grads["dgamma"+str(L)], grads["dbeta"+str(L)] = linear_activation_backward(dAL, current_cache, keep_prob, activation='sigmoid')
    
    # loop from L-2 to 0
    for l in reversed(range(L-1)):
        # Relu -> Linear
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp, dgamma_temp, dbeta_temp = linear_activation_backward(grads["dA"+str(l+1)], current_cache, keep_prob, activation='relu')
        grads["dA"+str(l)] = dA_prev_temp
        grads["dW"+str(l+1)] = dW_temp
        grads["db"+str(l+1)] = db_temp
        grads["dgamma"+str(l+1)] = dgamma_temp
        grads["dbeta"+str(l+1)] = dbeta_temp
    return grads
