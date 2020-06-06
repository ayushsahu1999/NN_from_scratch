# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 12:56:31 2020

@author: Dell
"""
import numpy as np
from linear_activation_forward import sigmoid, relu

def forward(Z, gamma, beta, mu, sigma, epsilon=1e-8):
    
    Z_norm = (Z - mu) / np.sqrt(sigma + epsilon)
    Z_telda = np.multiply(gamma, Z_norm) + beta
    
    return Z_telda

def one_step(A_prev, W, b, activation, v_mu, v_sigma, gamma, beta):
    """
    A_prev -> Activation from previous layer
    W -> Weights
    b -> Bias
    activation -> Sigmoid or Relu
    """
    if activation == 'sigmoid':
        Z = np.dot(W, A_prev)+b
        Z_telda = forward(Z, gamma, beta, v_mu, v_sigma)
        A = sigmoid(Z_telda)
        
    elif activation == 'relu':
        Z = np.dot(W, A_prev)+b
        Z_telda = forward(Z, gamma, beta, v_mu, v_sigma)
        A = relu(Z_telda)
        
    assert(A.shape == (W.shape[0], A_prev.shape[1]))
    
    return A, v_mu, v_sigma

def predict(X, parameters, b_par, v_mu, v_sigma):
    A = X
    L = len(parameters) // 2  # number of layers 
    for l in range(1, L):
        A_prev = A
        t_mu = v_mu[l]
        t_sigma = v_sigma[l]
        A, a, b = one_step(A_prev, parameters["W"+str(l)], parameters["b"+str(l)],
                                                      activation = 'relu', v_mu=t_mu, v_sigma=t_sigma,
                                                      gamma=b_par["gamma"+str(l)],
                                                      beta=b_par["beta"+str(l)])
        v_mu[l] = a
        v_sigma[l] = b
        
    t_mu = v_mu[L]
    t_sigma = v_sigma[L]
    AL, v_mu[L], v_sigma[L] = one_step(A, parameters["W"+str(L)], parameters["b"+str(L)],
                                                  activation = 'sigmoid', v_mu=t_mu, v_sigma=t_sigma,
                                                  gamma=b_par["gamma"+str(L)],
                                                  beta=b_par["beta"+str(L)])
    
    
    
    
    #assert (AL.shape == (y_train.shape[0], y_train.shape[1]))
    return AL