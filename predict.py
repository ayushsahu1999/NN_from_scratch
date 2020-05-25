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
layers_dims = [11, 10, 10, 5, 1]
parameters = L_layer_model(X_train, y_train, layers_dims,
                           learning_rate=0.4, num_iterations=5000, print_cost=True)

def pred(X, Y, parameters):
    y_pred, caches = L_model_forward(X, parameters)
    return y_pred

## Test Accuracy
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

## Train Accuracy
y_pred = pred(X_train, y_train, parameters)
y_pred[y_pred<=0.5] = 0
y_pred[y_pred>0.5] = 1

n = 0
cm = np.zeros((2, 2))
total = y_pred.shape[1]
assert (y_pred.shape == y_train.shape)

for i in range(len(y_pred[0])):
    if (int(y_pred[0][i]) == int(y_train[0][i])):
        n = n + 1
    if (int(y_pred[0][i]) != int(y_train[0][i]) and int(y_pred[0][i]==0)):
        cm[0][1] = cm[0][1] + 1
    if (int(y_pred[0][i]) != int(y_train[0][i]) and int(y_pred[0][i]==1)):
        cm[1][0] = cm[1][0] + 1
    if (int(y_pred[0][i]) == int(y_train[0][i]) and int(y_pred[0][i]==0)):
        cm[0][0] = cm[0][0] + 1
    if (int(y_pred[0][i]) != int(y_train[0][i]) and int(y_pred[0][i]==1)):
        cm[1][1] = cm[1][1] + 1
 
accuracy = n / total

'''
layers_dims = [11, 10, 8, 1]
Using alpha = 0.1:
    train accuracy: 86.225
    test accuracy: 86.15
    --> UnderFit (High Bias)
'''
'''
Using alpha = 0.01:
    num_iterations = 3500
    layers_dims = [11, 10, 8, 1]
    train accuracy: 86.225
    test accuracy: 82.55
    --> UnderFit (High Bias)
'''
'''
Using alpha = 0.2:
    num_iterations = 3500
    layers_dims = [11, 10, 10, 1]
    train accuracy: 86.5
    test accuracy: 86.2
    --> UnderFit (High Bias)
'''
'''
Using alpha = 0.3:
    num_iterations = 3500
    layers_dims = [11, 10, 10, 5, 1]
    train accuracy: 86.76
    test accuracy: 86.25
    --> UnderFit (High Bias)
'''
'''
Using alpha = 0.4:
    num_iterations = 4000
    layers_dims = [11, 10, 10, 5, 1]
    train accuracy: 87.07
    test accuracy: 86.05
    --> UnderFit (High Bias)
'''
'''
Using alpha = 0.35:
    num_iterations = 4000
    layers_dims = [11, 10, 10, 5, 1]
    train accuracy: 86.85
    test accuracy: 85.65
    --> UnderFit (High Bias)
'''
'''
Using alpha = 0.4:
    num_iterations = 5000
    layers_dims = [11, 10, 10, 5, 1]
    train accuracy: 86.82
    test accuracy: 86.6
    --> UnderFit (High Bias)
'''
'''
Using alpha = 0.3:
    num_iterations = 5000
    layers_dims = [11, 10, 10, 5, 5, 1]
    train accuracy: 87.13
    test accuracy: 85.05
    --> UnderFit (High Bias)
'''
'''
Using alpha = 0.1:
    num_iterations = 5000
    layers_dims = [11, 10, 10, 5, 5, 1]
    train accuracy: 86.16
    test accuracy: 86.25
    --> UnderFit (High Bias)
'''


#layers_dims = [11, 10, 8, 1] --> accuracy = 86.15%
#layers_dims = [11, 10, 8, 1] --> (test) --> accuracy = 86.15%
#layers_dims = [11, 10, 8, 1] --> (train) --> accuracy = 86.225%
#layers_dims = [11, 6, 6, 1] --> accuracy = 86.05%