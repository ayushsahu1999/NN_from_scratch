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
        parameters["gamma"+str(l)] = np.random.rand(layers_dims[l], 1) * 0.00001
        parameters["beta"+str(l)] = np.random.rand(layers_dims[l], 1) * 0.00001
    return parameters


# Forward Propagation
def forward_prop(Z, gamma, beta, epsilon=1e-8):
    m = Z.shape[1]
    mu = (1/m)*np.sum(Z, axis=1, keepdims=True)
    q = Z - mu
    q = q**2
    sigma = (1/m)*np.sum(q, axis=1, keepdims=True)
    Z_norm = (Z - mu) / np.sqrt(sigma + epsilon)
    Z_telda = np.multiply(gamma, Z_norm) + beta
    cache = (mu, sigma, Z_norm)
    assert (Z_norm.shape == Z_telda.shape)
    assert (Z_norm.shape == Z.shape)
    assert (Z_telda.shape == Z.shape)
    return Z_telda, cache

# Backward Propagation
def back_prop(Z, dout, mu, sigma, Z_norm, gamma, epsilon=1e-8):
    m = Z.shape[1]
    dZ_norm = dout*gamma
    t = np.multiply(dZ_norm, (Z - mu))
    t = t * (-0.5*np.power((sigma+epsilon), -3/2))
    dsigma = np.sum(t, axis=1, keepdims=True)
    u = dZ_norm*(-1/np.sqrt(sigma+epsilon))
    v = -2*(Z - mu)
    dmu = np.sum(u, axis=1, keepdims=True) + dsigma*(np.sum(v, axis=1, keepdims=True)/m)
    dZ = dZ_norm/(np.sqrt(sigma+epsilon)) + dsigma*2*(Z - mu)/m + dmu/m
    w = np.multiply(dout, Z_norm)
    dgamma = np.sum(w, axis=1, keepdims=True)
    dbeta = np.sum(dout, axis=1, keepdims=True)
    return dZ, dgamma, dbeta
    