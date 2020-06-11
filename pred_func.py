# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 12:56:31 2020

@author: Dell
"""

import numpy as np
from linear_activation_forward import L_model_forward

def pred(X, Y, parameters, b_par):
    b_par['mode'] = 'test'
    y_pred, c, b = L_model_forward(X, parameters, b_par, batch_norm=False)
    return y_pred


def accuracy(parameters, b_par, sets, train=True, val=True, test=True):
    cm_train = np.zeros((2, 2))
    cm_test = np.zeros((2, 2))
    cm_val = np.zeros((2, 2))
    
    accuracy_train = ''
    accuracy_test = ''
    accuracy_val = ''
    
    X_train, y_train = sets['train']
    X_test, y_test = sets['test']
    X_val, y_val = sets['val']
    
    predictions = {}
    
    if test:
        ## Test Accuracy
        
        y_pred_test = pred(X_test, y_test, parameters, b_par)
        y_pred_test[y_pred_test <= 0.5] = 0
        y_pred_test[y_pred_test > 0.5] = 1
        
        n = 0
        total = y_pred_test.shape[1]
        assert (y_pred_test.shape == y_test.shape)
        
        for i in range(len(y_pred_test[0])):
            if (int(y_pred_test[0][i]) == int(y_test[0][i])):
                n = n + 1
            if (int(y_pred_test[0][i]) != int(y_test[0][i]) and int(y_pred_test[0][i] == 0)):
                cm_test[0][1] = cm_test[0][1] + 1
            if (int(y_pred_test[0][i]) != int(y_test[0][i]) and int(y_pred_test[0][i] == 1)):
                cm_test[1][0] = cm_test[1][0] + 1
            if (int(y_pred_test[0][i]) == int(y_test[0][i]) and int(y_pred_test[0][i] == 0)):
                cm_test[0][0] = cm_test[0][0] + 1
            if (int(y_pred_test[0][i]) == int(y_test[0][i]) and int(y_pred_test[0][i] == 1)):
                cm_test[1][1] = cm_test[1][1] + 1
        
        accuracy_test = cm_test[1][1] / (cm_test[0][1] + cm_test[1][1] + cm_test[1][0])
        accuracy_test = str(accuracy_test * 100) + '%'
        
        predictions['test'] = y_pred_test
        
    if val:
        ## Validation Accuracy
        
        y_pred_val = pred(X_val, y_val, parameters, b_par)
        y_pred_val[y_pred_val <= 0.5] = 0
        y_pred_val[y_pred_val > 0.5] = 1
        
        n = 0
        total = y_pred_val.shape[1]
        assert (y_pred_val.shape == y_test.shape)
        
        for i in range(len(y_pred_val[0])):
            if (int(y_pred_val[0][i]) == int(y_val[0][i])):
                n = n + 1
            if (int(y_pred_val[0][i]) != int(y_val[0][i]) and int(y_pred_val[0][i] == 0)):
                cm_val[0][1] = cm_val[0][1] + 1
            if (int(y_pred_val[0][i]) != int(y_val[0][i]) and int(y_pred_val[0][i] == 1)):
                cm_val[1][0] = cm_val[1][0] + 1
            if (int(y_pred_val[0][i]) == int(y_val[0][i]) and int(y_pred_val[0][i] == 0)):
                cm_val[0][0] = cm_val[0][0] + 1
            if (int(y_pred_val[0][i]) == int(y_val[0][i]) and int(y_pred_val[0][i] == 1)):
                cm_val[1][1] = cm_val[1][1] + 1
        
        accuracy_val = cm_val[1][1] / (cm_val[0][1] + cm_val[1][1] + cm_val[1][0])
        accuracy_val = str(accuracy_val * 100) + '%'
        
        predictions['val'] = y_pred_val
        
    if train:
        ## Train accuracy
        
        y_pred_train = pred(X_train, y_train, parameters, b_par)
        y_pred_train[y_pred_train <= 0.5] = 0
        y_pred_train[y_pred_train > 0.5] = 1
        
        n = 0
        
        total = y_pred_train.shape[1]
        assert (y_pred_train.shape == y_train.shape)
        
        for i in range(len(y_pred_train[0])):
            if (int(y_pred_train[0][i]) == int(y_train[0][i])):
                n = n + 1
            if (int(y_pred_train[0][i]) != int(y_train[0][i]) and int(y_pred_train[0][i] == 0)):
                cm_train[0][1] = cm_train[0][1] + 1
            if (int(y_pred_train[0][i]) != int(y_train[0][i]) and int(y_pred_train[0][i] == 1)):
                cm_train[1][0] = cm_train[1][0] + 1
            if (int(y_pred_train[0][i]) == int(y_train[0][i]) and int(y_pred_train[0][i] == 0)):
                cm_train[0][0] = cm_train[0][0] + 1
            if (int(y_pred_train[0][i]) == int(y_train[0][i]) and int(y_pred_train[0][i] == 1)):
                cm_train[1][1] = cm_train[1][1] + 1
        
        accuracy_train = cm_train[1][1] / (cm_train[0][1] + cm_train[1][1] + cm_train[1][0])
        accuracy_train = str(accuracy_train * 100) + '%'
        
        predictions['train'] = y_pred_train
        
    accuracies = {
        'train': accuracy_train,
        'val': accuracy_val,
        'test': accuracy_test
    }
    
    cms = {
        'train': cm_train,
        'val': cm_val,
        'test': cm_test
    }
    
    results = {
            'accuracies': accuracies,
            'cms': cms,
            'predictions': predictions
        }

    
    return results
    
