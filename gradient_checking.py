# -*- coding: utf-8 -*-
"""
Created on Mon May 25 16:44:33 2020

@author: Dell
"""
import numpy as np


def dictionary_to_vector(parameters):
    keys = []
    count = 0
    l = list(parameters.keys())
    for key in l:
        # flatten parameters
        new_vector = np.reshape(parameters[key], (-1, 1))
        keys = keys + [key]*new_vector.shape[0]
        if count == 0:
            theta = new_vector
        else:
            theta = np.concatenate((theta, new_vector), axis=0)
        count = count + 1
    return theta, keys

def vector_to_dictionary(theta):
    parameters = {}
    parameters["W1"] = theta[:110].reshape((10, 11))
    parameters["b1"] = theta[110:120].reshape((10, 1))
    parameters["W2"] = theta[120:220].reshape((10, 10))
    parameters["b2"] = theta[220:230].reshape((10, 1))
    parameters["W3"] = theta[230:280].reshape((5, 10))
    parameters["b3"] = theta[280:285].reshape((5, 1))
    parameters["W4"] = theta[285:290].reshape((1, 5))
    parameters["b4"] = theta[290:291].reshape((1, 1))
    
    return parameters

def gradients_to_vector(gradients):
    count = 0
    #g = list(gradients.keys())
    for key in ['dW1', 'db1', 'dW2', 'db2', 'dW3', 'db3', 'dW4', 'db4']:
        new_vector = np.reshape(gradients[key], (-1, 1))
        if count == 0:
            theta = new_vector
        else:
            theta = np.concatenate((theta, new_vector), axis = 0)
        count = count + 1
        
    return theta


def grad_check(parameters, gradients, X, y, epsilon=1e-7):
    from linear_activation_forward import forward_prop_check
    parameters_values, _ = dictionary_to_vector(parameters)
    grad = gradients_to_vector(gradients)
    num_parameters = parameters_values.shape[0]
    J_plus = np.zeros((num_parameters, 1))
    J_minus = np.zeros((num_parameters, 1))
    gradapprox = np.zeros((num_parameters, 1))
    
    for i in range(num_parameters):
        thetaplus = np.copy(parameters_values)
        thetaplus[i][0] = thetaplus[i][0] + epsilon
        
        J_plus[i], _ = forward_prop_check(X, y, vector_to_dictionary(thetaplus))
        
        thetaminus = np.copy(parameters_values)
        thetaminus[i][0] = thetaminus[i][0] - epsilon
        J_minus[i], _ = forward_prop_check(X, y, vector_to_dictionary(thetaminus))
        
        gradapprox[i] = (J_plus[i] - J_minus[i])/(2*epsilon)
    
    #print (grad.shape)

    numerator = np.linalg.norm(grad - gradapprox)
    denominator = np.linalg.norm(grad) + np.linalg.norm(gradapprox)
    difference = numerator / denominator
    if difference > 2e-7:
        print ('There was a mistake in gradient descent')
    else:
        print ('The implementation of gradient descent is fine')
    return difference
    