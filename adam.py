# -*- coding: utf-8 -*-
"""
Created on Mon May 25 22:02:38 2020

@author: Dell
"""

import numpy as np
def initialize_adam(parameters, b_par):
    L = len(parameters) // 2
    v = {}
    s = {}
    
    for l in range(L):
        v["dW"+str(l+1)] = np.zeros((parameters["W"+str(l+1)].shape[0], parameters["W"+str(l+1)].shape[1]))
        v["db"+str(l+1)] = np.zeros((parameters["b"+str(l+1)].shape[0], parameters["b"+str(l+1)].shape[1]))
        v["dgamma"+str(l+1)] = np.zeros((b_par["gamma"+str(l+1)].shape[0], b_par["gamma"+str(l+1)].shape[1]))
        v["dbeta"+str(l+1)] = np.zeros((b_par["beta"+str(l+1)].shape[0], b_par["beta"+str(l+1)].shape[1]))
        
        s["dW"+str(l+1)] = np.zeros((parameters["W"+str(l+1)].shape[0], parameters["W"+str(l+1)].shape[1]))
        s["db"+str(l+1)] = np.zeros((parameters["b"+str(l+1)].shape[0], parameters["b"+str(l+1)].shape[1]))
        s["dgamma"+str(l+1)] = np.zeros((b_par["gamma"+str(l+1)].shape[0], b_par["gamma"+str(l+1)].shape[1]))
        s["dbeta"+str(l+1)] = np.zeros((b_par["beta"+str(l+1)].shape[0], b_par["beta"+str(l+1)].shape[1]))
        
    return v, s



    