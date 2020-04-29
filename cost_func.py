# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 19:50:52 2020

@author: Dell
"""

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def compute_cost(AL, Y):
    """
    AL -> predictions
    Y -> true
    """
    m = Y.shape[1]
    
    cost = -(1/m)*(np.dot(Y, np.log(AL).T)+np.dot((1-Y), np.log(1-AL).T))
    cost = np.squeeze(cost)  # [[10]] --> 10
    assert (cost.shape == ())
    return cost

"""
from linear_activation_forward import L_model_forward
from parameter_initialize import para_init
from data_init import dat_init
X_train, y_train, X_test, y_test = dat_init()
layers = [11, 6, 6, 1]
par = para_init(layers, 0.01)
AL, caches = L_model_forward(X_train, par)
cost = compute_cost(AL, y_train)
print (cost)
"""
