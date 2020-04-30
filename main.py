# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 15:22:09 2020

@author: Dell
"""

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from linear_activation_forward import L_model_forward
from parameter_initialize import para_init
from data_init import dat_init
from cost_func import compute_cost
from backpropagation import L_model_backward
from update_parameters import update_params

def L_layer_model(X, Y, layers_dims, learning_rate=0.0075, num_iterations=3000, print_cost=False):
    """
    Implements L layer neural network
    (Linear->Relu)*(L-1) -> Linear->Sigmoid
    """
    costs = []
    # Parameters Initialization
    parameters = para_init(layers_dims, 0.01)
    
    # Gradient Descent
    for i in range(0, num_iterations):
        
        # Forward propagation
        AL, caches = L_model_forward(X, parameters)
        
        # Compute cost
        cost = compute_cost(AL, Y)
        
        # Backward propagation
        grads = L_model_backward(AL, Y, caches)
        
        # Update parameters
        parameters = update_params(parameters, grads, learning_rate)
        
        # Print the cost after 100 training examples
        if print_cost and i%100==0:
            print ("Cost after iteration %i: %f"%(i, cost))
        if print_cost and i%100==0:
            costs.append(cost)
    # plot the costs
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations(per hundreds)')
    plt.title('Learning rate = '+str(learning_rate))
    plt.show()
    
    return parameters

X_train, y_train, X_test, y_test = dat_init()
layers_dims = [11, 6, 6, 1]
parameters = L_layer_model(X_train, y_train, layers_dims, learning_rate=0.01, num_iterations=2500,print_cost=True)
