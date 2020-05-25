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
    from gradient_checking import grad_check
    
    """
    Implements L layer neural network
    (Linear->Relu)*(L-1) -> Linear->Sigmoid
    """
    costs = []
    
    #s = math.sqrt(2/X.shape[0])
    # Parameters Initialization
    parameters = para_init(layers_dims)
    
    # Gradient Descent
    for i in range(0, num_iterations):
        
        # Forward propagation
        AL, caches = L_model_forward(X, parameters)
        
        # Compute cost
        cost = compute_cost(AL, Y)
        
        # Backward propagation
        grads = L_model_backward(AL, Y, caches)
        
        # gradient checking
        '''
        if (i==0 or i==100 or i==500 or i==3000):
            parameters, diff = update_params(X, Y, parameters, grads, learning_rate, check=True)
            if diff>2e-7:
                break
        '''
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


def adam_model(X, Y, layers_dims, optimizer='adam', decay_rate = 0.0005,
               learning_rate=0.0007, mini_batch_size=64, beta=0.9, beta1=0.9,
               beta2=0.999, epsilon=1e-8, num_epochs=10000, print_cost=True, decay=False):
    
    # Import libraries
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    from linear_activation_forward import L_model_forward
    from parameter_initialize import para_init
    from cost_func import compute_cost
    from backpropagation import L_model_backward
    from update_parameters import update_parameters_with_adam
    from adam import initialize_adam
    from minibatches import create_minibatch
    
    L = len(layers_dims)
    costs = []
    t = 0
    m = X.shape[1]
    
    # Initialize parameters
    parameters = para_init(layers_dims)
    
    v, s = initialize_adam(parameters)
    
    # Optimization loop
    for i in range(num_epochs):
        minibatches = create_minibatch(X, Y, mini_batch_size)
        cost_total = 0
        for mini_batch in minibatches:
            
            # select a mini-batch
            (mini_batch_X, mini_batch_Y) = mini_batch
            
            # Forward propagation
            AL, caches = L_model_forward(mini_batch_X, parameters)
            
            # Compute cost
            cost_total = cost_total + compute_cost(AL, mini_batch_Y)
            
            # Backward propagation
            grads = L_model_backward(AL, mini_batch_Y, caches)
            
            # Update Parameters
            t = t + 1
            if decay:
                learning_rate = learning_rate/(1+decay_rate*i)
            parameters, v, s = update_parameters_with_adam(parameters, grads, v, s, t, learning_rate, beta1, beta2, epsilon)
            
        cost_avg = cost_total / m
        
        # Print the cost after 100 training examples
        if print_cost and i%100==0:
            print ("Cost after iteration %i: %f"%(i, cost_avg))
        if print_cost and i%100==0:
            costs.append(cost_avg)
    
    # plot the costs
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations(per hundreds)')
    plt.title('Learning rate = '+str(learning_rate))
    plt.show()
    
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