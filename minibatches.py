# -*- coding: utf-8 -*-
"""
Created on Mon May 25 21:49:18 2020

@author: Dell
"""
import numpy as np
import math

def create_minibatch(X, Y, mini_batch_size=64):
    '''
    Creates a list of random mini-batches
    '''
    m = X.shape[1]
    mini_batches = []
    
    # Step-1: Shuffle(X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((1, m))
    
    # Step-2: Partition (Shuffled_X, Shuffled_Y). Minus the end case
    num_complete_minibatches = math.floor(m/mini_batch_size)
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k*mini_batch_size: k*mini_batch_size+mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k*mini_batch_size: k*mini_batch_size+mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
        
    # Step-3: Handling the end case
    if m%mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, num_complete_minibatches*mini_batch_size:m]
        mini_batch_Y = shuffled_Y[:, num_complete_minibatches*mini_batch_size:m]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    return mini_batches


    