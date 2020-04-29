# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 19:04:25 2020

@author: Dell
"""

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Initializing the parameters
def para_init(layers_dims):
    from data_init import dat_init
    X_train, y_train, X_test, y_test = dat_init()
    parameters = {}
    L = len(layers_dims)
    for l in  range(1, L):
        parameters["W"+str(l)] = np.random.randn(layers_dims[l], layers_dims[l-1]) * 0.01
        parameters["b"+str(l)] = np.zeros((layers_dims[l], 1))
        
        assert(parameters["W"+str(l)].shape == (layers_dims[l], layers_dims[l-1]))
        assert(parameters["b"+str(l)].shape == (layers_dims[l], 1))
    return parameters

"""
layers = [11, 6, 6, 1]
par = para_init(layers)
print (par["W1"].shape)
print (par["b1"].shape)
print (par["W2"].shape)
print (par["b2"].shape)
print (par["W3"].shape)
print (par["b3"].shape)
"""