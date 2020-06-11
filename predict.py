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
from main import adam_model
from data_init import dat_init
from pred_func import accuracy

X_train, y_train, X_test, y_test, X_val, y_val = dat_init()
# layers_dims = [11, 10, 8, 1]
# layers_dims = [11, 20, 10, 10, 1]
# layers_dims = [11, 10, 5, 5, 3, 3, 3, 1]
layers_dims = [11, 8, 4, 1]
parameters, b_par, grads = adam_model(X_train, y_train, layers_dims, optimizer='adam', dropout=True, decay_rate=0.02,
                               keep_prob = 0.7, batch_norm=False,
                               learning_rate=0.001, mini_batch_size=64, num_epochs=1500,
                               beta1=0.9, print_cost=True, decay=False)

# 00024371157127187917

sets = {
        'train': (X_train, y_train),
        'test': (X_test, y_test),
        'val': (X_val, y_val)
    }

results = accuracy(parameters, b_par, sets, train=True, 
                                        val=True, test=False)



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

# layers_dims = [11, 10, 8, 1] --> accuracy = 86.15%
# layers_dims = [11, 10, 8, 1] --> (test) --> accuracy = 86.15%
# layers_dims = [11, 10, 8, 1] --> (train) --> accuracy = 86.225%
# layers_dims = [11, 6, 6, 1] --> accuracy = 86.05%
