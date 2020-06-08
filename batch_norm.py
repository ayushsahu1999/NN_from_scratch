# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 14:27:00 2020

@author: Dell
"""

import numpy as np


def batch_norm_init(layers_dims):
    parameters = {}
    L = len(layers_dims)
    for l in range(1, L):
        parameters["gamma" + str(l)] = np.random.randn(layers_dims[l], 1) * 0.01
        parameters["beta" + str(l)] = np.random.randn(layers_dims[l], 1) * 0.01
    return parameters


# Forward Propagation
def forward_prop(Z, gamma, beta, bn_params):
    mode = bn_params['mode']
    eps = bn_params.get('eps', 1e-5)
    momentum = bn_params.get('momentum', 0.9)

    n = Z.shape[0]
    running_mean = np.zeros(n, dtype=Z.dtype).reshape(-1, 1)
    running_var = np.zeros(n, dtype=Z.dtype).reshape(-1, 1)
    running_means = bn_params['running_means']
    running_vars = bn_params['running_vars']

    if mode == 'train':
        sample_mean = Z.mean(axis=1)
        sample_var = Z.var(axis=1)
        
        sample_mean = sample_mean.reshape(-1, 1)
        sample_var = sample_var.reshape(-1, 1)
        
#        print (Z.shape)
        running_mean = momentum * running_mean + (1 - momentum) * sample_mean
        running_var = momentum * running_var + (1 - momentum) * sample_var

        std = np.sqrt(sample_var + eps)
        x_centered = Z - sample_mean
        x_norm = x_centered / std
        
        out = gamma * x_norm + beta
        running_means.append(running_mean)
        running_vars.append(running_var)
        bn_params['running_means'] = running_means
        bn_params['running_vars'] = running_vars
        
        cache = (x_norm, x_centered, std, gamma)

    elif mode == 'test' or mode == 'val':
        x_norm = (Z - running_mean) / np.sqrt(running_var + eps)
        cache = (0, 0, 0, 0)
        out = gamma * x_norm + beta

    else:
        raise ValueError('Invalid forward batch norm')

    bn_params['running_mean'] = running_mean
    bn_params['running_var'] = running_var

    return out, cache, bn_params


# Backward Propagation
def back_prop(dout, cache):
    m = dout.shape[1]
    x, x_norm, x_centered, std, gamma = cache
    dgamma = (dout * x_norm).sum(axis=1).reshape(-1, 1)
    dbeta = dout.sum(axis=1).reshape(-1, 1)
    dx_norm = dout * gamma
#    print (dx_norm.sum(axis=1))
    dx = 1 / m / std * (m * dx_norm -
                        dx_norm.sum(axis=1).reshape(-1, 1) -
                        x_norm * (dx_norm * x_norm).sum(axis=1).reshape(-1, 1))

    return dx, dgamma, dbeta
