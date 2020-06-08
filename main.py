# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 15:22:09 2020

@author: Dell
"""



def L_layer_model(X, Y, layers_dims, learning_rate=0.0075, num_iterations=3000, print_cost=False, train=True):
    # Import libraries
    '''
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    from linear_activation_forward import L_model_forward
    
    from parameter_initialize import para_init
    from cost_func import compute_cost
    from backpropagation import L_model_backward
    from update_parameters import update_params
    from batch_norm import forward_prop, batch_norm_init, back_prop
    
    '''
    from gradient_checking import grad_check
    """
    Implements L layer neural network
    (Linear->Relu)*(L-1) -> Linear->Sigmoid
    
    """
    from batch_norm import forward_prop, batch_norm_init, back_prop
    costs = []
    
    #s = math.sqrt(2/X.shape[0])
    # Parameters Initialization
    parameters = para_init(layers_dims)
    b_par = batch_norm_init(layers_dims)
    
    # Gradient Descent
    for i in range(0, num_iterations):
        
        # Forward propagation
        AL, caches = L_model_forward(X, parameters, b_par)
        
        # Compute cost
        cost = compute_cost(AL, Y)
        
        # Backward propagation
        grads = L_model_backward(AL, Y, caches)
        
        # gradient checking
        
        if (i==0 or i==100 or i==500 or i==3000):
            parameters, diff = update_params(X, Y, parameters, grads, learning_rate, check=True)
            if diff>2e-7:
                break
        
        # Update parameters
        parameters, b_par = update_params(parameters, b_par, grads, learning_rate)
        
        
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


def adam_model(X, Y, layers_dims, optimizer='adam', decay_rate = 0.0005, dropout=True, keep_prob=0.5,
               learning_rate=0.0007, mini_batch_size=64, mode='train', beta=0.9, beta1=0.9,
               beta2=0.999, epsilon=1e-8, num_epochs=10000, print_cost=True, decay=False):
    
    # Import libraries
    
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    from linear_activation_forward import L_model_forward
    from linear_activation_forward_with_dropout import L_model_forward as ldr
    from parameter_initialize import para_init
    from cost_func import compute_cost
    from backpropagation import L_model_backward
    from backpropagation_with_dropout import L_model_backward as bdr
    from update_parameters import update_parameters_with_adam, update_parameters_with_momentum
    from adam import initialize_adam
    from batch_norm import forward_prop, batch_norm_init, back_prop
    from minibatches import create_minibatch
    
    L = len(layers_dims)
    costs = []
    t = 0
    m = X.shape[1]
    from batch_norm import forward_prop, batch_norm_init, back_prop
    # Initialize parameters
    parameters = para_init(layers_dims)
    b_par = batch_norm_init(layers_dims)
    b_par['mode'] = mode
    b_par['running_means'] = []
    b_par['running_vars'] = []
    
    v, s = initialize_adam(parameters, b_par)
    alpha = learning_rate # initial learning rate
    
    
    # Optimization loop
    for i in range(num_epochs):
        minibatches = create_minibatch(X, Y, mini_batch_size)
        cost_total = 0
        for mini_batch in minibatches:
            
            # select a mini-batch
            (mini_batch_X, mini_batch_Y) = mini_batch
            
#            print ('Before propagation')
#            print (b_par['gamma1'].shape)
            
            # Forward propagation
            if dropout:
                AL, caches, b_par = ldr(mini_batch_X, parameters, b_par, keep_prob=keep_prob)
            else:
                AL, caches, b_par = L_model_forward(mini_batch_X, parameters, b_par)
            
#            print ('After forward propagation')
#            print (b_par['gamma1'].shape)
            
            # Compute cost
            cost_total = cost_total + compute_cost(AL, mini_batch_Y)
#            print ('Cost: '+str(cost_total))
            
#            print ('After cost')
#            print (b_par['gamma1'].shape)
            
            # Backward propagation
            if dropout:
                grads = bdr(AL, mini_batch_Y, caches, keep_prob)
            else:
                grads = L_model_backward(AL, mini_batch_Y, caches)
            
#            print ('After backpropagation')
#            print (b_par['gamma1'].shape)
            
            # Update Parameters
            
            if decay:
                learning_rate = alpha/(1+decay_rate*i)
            if (optimizer == 'adam'):
                t = t + 1
                parameters, b_par, v, s = update_parameters_with_adam(parameters, b_par, grads, v, s, t, learning_rate, beta1, beta2, epsilon)
            elif (optimizer == 'momentum'):
                parameters, b_par, v = update_parameters_with_momentum(parameters, b_par, grads, v, beta, learning_rate)
            
#            print ('Mini-batch: ' + str(len(mini_batch)))
#            print ('After parameters update')
#            print (b_par['gamma1'].shape)
#            print (grads['dgamma1'].shape)
            
            
        cost_avg = cost_total / m
        
        # Print the cost after 100 training examples
        if print_cost and i%100==0:
            
            print ("Cost after epoch %i: %f"%(i, cost_avg))
        if i%100==0:
            costs.append(cost_avg)
    
    # plot the costs
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations(per hundreds)')
    plt.title('Learning rate = '+str(learning_rate))
    plt.show()
    
    return parameters, b_par


"""
import numpy as np
import matplotlib.pyplot as plt
from linear_activation_forward import L_model_forward
from parameter_initialize import para_init
from cost_func import compute_cost
from backpropagation import L_model_backward
from update_parameters import update_params
from update_parameters import update_parameters_with_adam, update_parameters_with_momentum
from adam import initialize_adam
#from batch_norm import forward_prop, batch_norm_init, back_prop
from minibatches import create_minibatch
from data_init import dat_init
X_train, y_train, X_test, y_test, X_val, y_val = dat_init()
#layers_dims = [11, 20, 10, 10, 1]
parameters, b_par = adam_model(X_train, y_train, layers_dims, optimizer='adam', decay_rate=0.002,
                           learning_rate=0.01, mini_batch_size=64, num_epochs=1500,
                           beta1=0.9, print_cost=True, decay=False)
#parameters, b_par = L_layer_model(X_train, y_train, layers_dims, learning_rate=0.1, num_iterations=4000, print_cost=True)



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