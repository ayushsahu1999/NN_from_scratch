# -*- coding: utf-8 -*-
"""
Created on Fri May  1 12:12:48 2020

@author: Dell
"""

# Importing the libraries
from main import L_layer_model
from data_init import dat_init
from linear_activation_forward import L_model_forward
import numpy as np
X_train, y_train, X_test, y_test = dat_init()
#layers_dims = [11, 10, 8, 1]
layers_dims = [11, 6, 6, 1]
parameters = L_layer_model(X_train, y_train, layers_dims, learning_rate=0.1, num_iterations=3500, print_cost=True)


def pred(X, Y, parameters):
    y_pred, caches = L_model_forward(X, parameters)
    return y_pred

y_pred = pred(X_test, y_test, parameters)
y_pred[y_pred<=0.5] = 0
y_pred[y_pred>0.5] = 1

n = 0
cm = np.zeros((2, 2))
total = y_pred.shape[1]
assert (y_pred.shape == y_test.shape)
for i in range(len(y_pred[0])):
    if (int(y_pred[0][i]) == int(y_test[0][i])):
        n = n + 1
    if (int(y_pred[0][i]) != int(y_test[0][i]) and int(y_pred[0][i]==0)):
        cm[0][1] = cm[0][1] + 1
    if (int(y_pred[0][i]) != int(y_test[0][i]) and int(y_pred[0][i]==1)):
        cm[1][0] = cm[1][0] + 1
    if (int(y_pred[0][i]) == int(y_test[0][i]) and int(y_pred[0][i]==0)):
        cm[0][0] = cm[0][0] + 1
    if (int(y_pred[0][i]) != int(y_test[0][i]) and int(y_pred[0][i]==1)):
        cm[1][1] = cm[1][1] + 1
    
accuracy = n / total

#layers_dims = [11, 10, 8, 1] --> accuracy = 86.25%
#layers_dims = [11, 6, 6, 1] --> accuracy = 86.35%