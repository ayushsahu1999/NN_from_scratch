# -*- coding: utf-8 -*-
"""
Created on Fri May  1 12:12:48 2020

@author: Dell
"""

# Importing the libraries
from main import L_layer_model
from data_init import dat_init
from linear_activation_forward import L_model_forward
X_train, y_train, X_test, y_test = dat_init()
layers_dims = [11, 10, 8, 1]
parameters = L_layer_model(X_train, y_train, layers_dims, learning_rate=0.01, num_iterations=2500, print_cost=True)


def pred(X, Y, parameters):
    y_pred, caches = L_model_forward(X, parameters)
    return y_pred

y_pred = pred(X_test, y_test, parameters)
y_pred[y_pred<=0.5] = 0
y_pred[y_pred>0.5] = 1
