# -*- coding: utf-8 -*-
"""
Created on Fri May  1 12:12:48 2020

@author: Dell
"""

'''
According to me, human level accuracy of this problem is approx 90%.
So, not a bad model if above statement is correct.
'''
# https://wiseodd.github.io/techblog/2016/07/04/batchnorm/


# Importing the libraries
from main import L_layer_model, adam_model
from data_init import dat_init
from linear_activation_forward import L_model_forward
import numpy as np
X_train, y_train, X_test, y_test, X_val, y_val = dat_init()
#layers_dims = [11, 10, 8, 1]
#layers_dims = [11, 20, 10, 10, 1]
#layers_dims = [11, 10, 5, 5, 3, 3, 3, 1]
layers_dims = [11, 8, 5, 4, 2, 1, 3, 5, 3, 1]
parameters, b_par = adam_model(X_train, y_train, layers_dims, optimizer='momentum', decay_rate=0.002,
                           learning_rate=0.001, mini_batch_size=64, num_epochs=1000,
                           beta1=0.9, print_cost=True, decay=False)

#00024371157127187917
                           
'''
parameters = L_layer_model(X_train, y_train, layers_dims,
                           learning_rate=0.4, num_iterations=5000, print_cost=True)
'''


def pred(X, Y, parameters):
    b_par['mode'] = 'test'
    y_pred, c, b = L_model_forward(X, parameters, b_par)
    return y_pred

## Test Accuracy
y_pred_test = pred(X_test, y_test, parameters)
y_pred_test[y_pred_test<=0.5] = 0
y_pred_test[y_pred_test>0.5] = 1

n = 0
cm_test = np.zeros((2, 2))
total = y_pred_test.shape[1]
assert (y_pred_test.shape == y_test.shape)

for i in range(len(y_pred_test[0])):
    if (int(y_pred_test[0][i]) == int(y_test[0][i])):
        n = n + 1
    if (int(y_pred_test[0][i]) != int(y_test[0][i]) and int(y_pred_test[0][i]==0)):
        cm_test[0][1] = cm_test[0][1] + 1
    if (int(y_pred_test[0][i]) != int(y_test[0][i]) and int(y_pred_test[0][i]==1)):
        cm_test[1][0] = cm_test[1][0] + 1
    if (int(y_pred_test[0][i]) == int(y_test[0][i]) and int(y_pred_test[0][i]==0)):
        cm_test[0][0] = cm_test[0][0] + 1
    if (int(y_pred_test[0][i]) == int(y_test[0][i]) and int(y_pred_test[0][i]==1)):
        cm_test[1][1] = cm_test[1][1] + 1
        
accuracy_test = cm_test[1][1] / (cm_test[0][1] + cm_test[1][1] + cm_test[1][0])
accuracy_test = str(accuracy_test*100)+'%'

## Validation Accuracy
y_pred_val = pred(X_val, y_val, parameters)
y_pred_val[y_pred_val<=0.5] = 0
y_pred_val[y_pred_val>0.5] = 1

n = 0
cm_val = np.zeros((2, 2))
total = y_pred_val.shape[1]
assert (y_pred_val.shape == y_test.shape)

for i in range(len(y_pred_val[0])):
    if (int(y_pred_val[0][i]) == int(y_val[0][i])):
        n = n + 1
    if (int(y_pred_val[0][i]) != int(y_val[0][i]) and int(y_pred_val[0][i]==0)):
        cm_val[0][1] = cm_val[0][1] + 1
    if (int(y_pred_val[0][i]) != int(y_val[0][i]) and int(y_pred_val[0][i]==1)):
        cm_val[1][0] = cm_val[1][0] + 1
    if (int(y_pred_val[0][i]) == int(y_val[0][i]) and int(y_pred_val[0][i]==0)):
        cm_val[0][0] = cm_val[0][0] + 1
    if (int(y_pred_val[0][i]) == int(y_val[0][i]) and int(y_pred_val[0][i]==1)):
        cm_val[1][1] = cm_val[1][1] + 1
        
accuracy_val = cm_val[1][1] / (cm_val[0][1] + cm_val[1][1] + cm_val[1][0])
accuracy_val = str(accuracy_val*100)+'%'


## Train accuracy
y_pred_train = pred(X_train, y_train, parameters)
y_pred_train[y_pred_train<=0.5] = 0
y_pred_train[y_pred_train>0.5] = 1

n = 0
cm_train = np.zeros((2, 2))
total = y_pred_train.shape[1]
assert (y_pred_train.shape == y_train.shape)

for i in range(len(y_pred_train[0])):
    if (int(y_pred_train[0][i]) == int(y_train[0][i])):
        n = n + 1
    if (int(y_pred_train[0][i]) != int(y_train[0][i]) and int(y_pred_train[0][i]==0)):
        cm_train[0][1] = cm_train[0][1] + 1
    if (int(y_pred_train[0][i]) != int(y_train[0][i]) and int(y_pred_train[0][i]==1)):
        cm_train[1][0] = cm_train[1][0] + 1
    if (int(y_pred_train[0][i]) == int(y_train[0][i]) and int(y_pred_train[0][i]==0)):
        cm_train[0][0] = cm_train[0][0] + 1
    if (int(y_pred_train[0][i]) == int(y_train[0][i]) and int(y_pred_train[0][i]==1)):
        cm_train[1][1] = cm_train[1][1] + 1
 
accuracy_train = cm_train[1][1] / (cm_train[0][1] + cm_train[1][1] + cm_train[1][0])
accuracy_train = str(accuracy_train*100)+'%'

accuracies = {
        'train': accuracy_train,
        'test': accuracy_test,
        'val': accuracy_val
        }

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
'''
Using alpha = 0.4(adam):
    num_iterations = 5000
    layers_dims = [11, 10, 10, 5, 1]
    train accuracy: 79.6
    test accuracy: 79.75
    --> UnderFit (High Bias)
'''
'''
Using alpha = 0.4(adam):
    mini_batch_size = 8000
    num_epochs = 1000
    layers_dims = [11, 10, 10, 10, 1]
    train accuracy: 87.425
    test accuracy: 85.6
    --> UnderFit (High Bias)
'''
'''
Using alpha = 0.01(adam):
    mini_batch_size = 8000
    num_epochs = 1000
    layers_dims = [11, 10, 10, 10, 1]
    train accuracy: 87.65
    test accuracy: 85.8
    --> UnderFit (High Bias)
'''
'''
Using alpha = 0.00024371157127187917(momentum):
    mini_batch_size = 8000
    num_epochs = 2000
    layers_dims = [11, 10, 10, 10, 1]
    train accuracy: 63.6
    test accuracy: 63.05
    --> UnderFit (High Bias)
'''
'''
Using alpha = 0.1(momentum):
    mini_batch_size = 8000
    num_epochs = 2000
    layers_dims = [11, 10, 10, 10, 1]
    train accuracy: 86.25
    test accuracy: 85.95
    --> UnderFit (High Bias)
'''
'''
Using alpha = 0.1(momentum):
    mini_batch_size = 8000
    num_epochs = 2000
    layers_dims = [11, 10, 1]
    train accuracy: 86.8
    test accuracy: 86.15
    --> UnderFit (High Bias)
'''
'''
Using alpha = 0.1(momentum):
    mini_batch_size = 8000
    num_epochs = 2500
    layers_dims = [11, 10, 1]
    train accuracy: 85.95
    test accuracy: 86.5
    --> UnderFit (High Bias)
'''


#layers_dims = [11, 10, 8, 1] --> accuracy = 86.15%
#layers_dims = [11, 10, 8, 1] --> (test) --> accuracy = 86.15%
#layers_dims = [11, 10, 8, 1] --> (train) --> accuracy = 86.225%
#layers_dims = [11, 6, 6, 1] --> accuracy = 86.05%