# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 15:14:07 2020

@author: Dell
"""

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from gradient_checking import grad_check

def update_params(parameters, b_par, grads, batch_norm, learning_rate):
    L = len(parameters) // 2
    '''
    diff = 1e-8
    if check:
        diff = grad_check(parameters, grads, X, Y)
        if (diff>2e-7):
            print (diff)
    '''
    for l in range(L):
        parameters["W"+str(l+1)] = parameters["W"+str(l+1)] - learning_rate * grads["dW"+str(l+1)]
        parameters["b"+str(l+1)] = parameters["b"+str(l+1)] - learning_rate * grads["db"+str(l+1)]
        
        if batch_norm:
            b_par["gamma"+str(l+1)] = b_par["gamma"+str(l+1)] - learning_rate * grads["dgamma"+str(l+1)]
            b_par["beta"+str(l+1)] = b_par["beta"+str(l+1)] - learning_rate * grads["dbeta"+str(l+1)]
    return parameters, b_par




def update_parameters_with_adam(parameters, b_par, grads, v, s, t, batch_norm, learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
    
    L = len(parameters) // 2
    v_corrected = {}
    s_corrected = {}
    
    for l in range(L):
        v["dW"+str(l+1)] = beta1*v["dW"+str(l+1)] + (1-beta1)*grads["dW"+str(l+1)]
        v["db"+str(l+1)] = beta1*v["db"+str(l+1)] + (1-beta1)*grads["db"+str(l+1)]
        
        v_corrected["dW"+str(l+1)] = v["dW"+str(l+1)] / (1 - beta1**t)
        v_corrected["db"+str(l+1)] = v["db"+str(l+1)] / (1 - beta1**t)
        
        s["dW"+str(l+1)] = beta2*s["dW"+str(l+1)] + (1-beta2)*np.square(grads["dW"+str(l+1)])
        s["db"+str(l+1)] = beta2*s["db"+str(l+1)] + (1-beta2)*np.square(grads["db"+str(l+1)])
        
        s_corrected["dW"+str(l+1)] = s["dW"+str(l+1)] / (1 - beta2**t)
        s_corrected["db"+str(l+1)] = s["db"+str(l+1)] / (1 - beta2**t)
        
        parameters["W"+str(l+1)] = parameters["W"+str(l+1)] - learning_rate*(v_corrected["dW"+str(l+1)]/(np.sqrt(s_corrected["dW"+str(l+1)])+epsilon))
        parameters["b"+str(l+1)] = parameters["b"+str(l+1)] - learning_rate*(v_corrected["db"+str(l+1)]/(np.sqrt(s_corrected["db"+str(l+1)])+epsilon))
        
        if batch_norm:
            v["dgamma"+str(l+1)] = beta1*v["dgamma"+str(l+1)] + (1-beta1)*grads["dgamma"+str(l+1)]
            v["dbeta"+str(l+1)] = beta1*v["dbeta"+str(l+1)] + (1-beta1)*grads["dbeta"+str(l+1)]
            
            v_corrected["dgamma"+str(l+1)] = v["dgamma"+str(l+1)] / (1 - beta1**t)
            v_corrected["dbeta"+str(l+1)] = v["dbeta"+str(l+1)] / (1 - beta1**t)
            
            s["dgamma"+str(l+1)] = beta2*s["dgamma"+str(l+1)] + (1-beta2)*np.square(grads["dgamma"+str(l+1)])
            s["dbeta"+str(l+1)] = beta2*s["dbeta"+str(l+1)] + (1-beta2)*np.square(grads["dbeta"+str(l+1)])
            
            s_corrected["dgamma"+str(l+1)] = s["dgamma"+str(l+1)] / (1 - beta2**t)
            s_corrected["dbeta"+str(l+1)] = s["dbeta"+str(l+1)] / (1 - beta2**t)
            
            b_par["gamma"+str(l+1)] = b_par["gamma"+str(l+1)] - learning_rate*(v_corrected["dgamma"+str(l+1)]/(np.sqrt(s_corrected["dgamma"+str(l+1)])+epsilon))
            b_par["beta"+str(l+1)] = b_par["beta"+str(l+1)] - learning_rate*(v_corrected["dbeta"+str(l+1)]/(np.sqrt(s_corrected["dbeta"+str(l+1)])+epsilon))
        
    return parameters, b_par, v, s

def update_parameters_with_momentum(parameters, b_par, grads, v, batch_norm, beta, learning_rate):
    L = len(parameters) // 2
    for l in range(L):
        v["dW"+str(l+1)] = beta*v["dW"+str(l+1)] + (1 - beta)*grads["dW"+str(l+1)]
        v["db"+str(l+1)] = beta*v["db"+str(l+1)] + (1-beta)*grads["db"+str(l+1)]
        
        parameters["W"+str(l+1)] = parameters["W"+str(l+1)] - learning_rate*(v["dW"+str(l+1)])
        parameters["b"+str(l+1)] = parameters["b"+str(l+1)] - learning_rate*(v["db"+str(l+1)])
        
        if batch_norm:
            v["dgamma"+str(l+1)] = beta*v["dgamma"+str(l+1)] + (1-beta)*grads["dgamma"+str(l+1)]
            v["dbeta"+str(l+1)] = beta*v["dbeta"+str(l+1)] + (1-beta)*grads["dbeta"+str(l+1)]
            
            b_par["gamma"+str(l+1)] = b_par["gamma"+str(l+1)] - learning_rate*(v["dgamma"+str(l+1)])
            b_par["beta"+str(l+1)] = b_par["beta"+str(l+1)] - learning_rate*(v["dbeta"+str(l+1)])
        
    return parameters, b_par, v
    
    
    

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