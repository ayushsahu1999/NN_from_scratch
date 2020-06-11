# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 21:32:53 2020

@author: Dell
"""
import numpy as np

def backward_propagation_with_dropout(X, Y, cache, keep_prob):
    
    m = X.shape[1]
    (Z1, D1, A1, W1, b1, Z2, D2, A2, W2, b2, Z3, A3, W3, b3) = cache
    
    dZ3 = A3 - Y
    dW3 = np.multiply(1/m, np.dot(dZ3, A2.T))
    db3 = (1/m)*np.sum(dZ3, axis=1, keepdims=True)
    dA2 = np.dot(W3.T, dZ3)
    dA2 = np.multiply(dA2, D2)
    dA2 = dA2 / keep_prob
    
    dZ2 = np.multiply(dA2, np.int64(A2 > 0))
    dW2 = np.multiply(1/m, np.dot(dZ2, A1.T))
    db2 = (1/m)*np.sum(dZ2, axis=1, keepdims=True)
    dA1 = np.dot(W2.T, dZ2)
    dA1 = np.multiply(dA1, D1)
    dA1 = dA1 / keep_prob
    
    dZ1 = np.multiply(dA1, np.int64(A1 > 0))
    dW1 = np.multiply(1/m, np.dot(dZ1, X.T))
    db1 = (1/m)*np.sum(dZ1, axis=1, keepdims=True)
    
    gradients = {
                "dZ3": dZ3, "dW3": dW3, "db3": db3, "dA2": dA2, "dZ2": dZ2,
                "dW2": dW2, "db2": db2, "dA1": dA1, "dZ1": dZ1, "dW1": dW1, "db1": db1
            }
    
    return gradients