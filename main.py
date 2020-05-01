# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 15:22:09 2020

@author: Dell
"""



def L_layer_model(X, Y, layers_dims, learning_rate=0.0075, num_iterations=3000, print_cost=False, train=True):
    # Import libraries
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    from linear_activation_forward import L_model_forward
    from parameter_initialize import para_init
    from cost_func import compute_cost
    from backpropagation import L_model_backward
    from update_parameters import update_params
    import math
    
    """
    Implements L layer neural network
    (Linear->Relu)*(L-1) -> Linear->Sigmoid
    """
    costs = []
    
    s = math.sqrt(2/X.shape[0])
    # Parameters Initialization
    parameters = para_init(layers_dims, s)
    
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
    
    #parameters["costs"] = costs
    #parameters["learning_rate"] = learning_rate
    
    return parameters

"""
from data_init import dat_init
X_train, y_train, X_test, y_test = dat_init()
layers_dims = [11, 10, 8, 1]
parameters = L_layer_model(X_train, y_train, layers_dims, learning_rate=0.1, num_iterations=4000, print_cost=True)
"""


"""
# Different Learning rates
import numpy as np
import matplotlib.pyplot as plt
learning_rates = [0.1, 0.01, 0.001]
models = {}
for i in learning_rates:
    print ('Learning rate is: '+str(i))
    models[str(i)] = L_layer_model(X_train, y_train, layers_dims, learning_rate=i, num_iterations=2500, print_cost=True)

for i in learning_rates:
    plt.plot(np.squeeze(models[str(i)]["costs"]), label= str(models[str(i)]["learning_rate"]))

plt.ylabel('cost')
plt.xlabel('iterations(per hundreds)')
legend = plt.legend(loc='upper center', shadow=True)
frame = legend.get_frame()
frame.set_facecolor('0.90')
plt.title('Different Learning rates')
plt.show()
"""