# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 15:14:07 2020

@author: Dell
"""

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def update_params(parameters, grads, learning_rate):
    L = len(parameters) // 2
    for l in range(L):
        parameters["W"+str(l+1)] = parameters["W"+str(l+1)] - learning_rate * grads["dW"+str(l+1)]
        parameters["b"+str(l+1)] = parameters["b"+str(l+1)] - learning_rate * grads["db"+str(l+1)]
        
    return parameters

"""
from linear_activation_forward import L_model_forward
from parameter_initialize import para_init
from data_init import dat_init
from cost_func import compute_cost
from backpropagation import L_model_backward
X_train, y_train, X_test, y_test = dat_init()
layers = [11, 6, 6, 1]
par = para_init(layers, 0.01)
AL, caches = L_model_forward(X_train, par)
cost = compute_cost(AL, y_train)
print (cost)
gradients = L_model_backward(AL, y_train, caches)
p = update_params(par, gradients, learning_rate=0.0075)
"""