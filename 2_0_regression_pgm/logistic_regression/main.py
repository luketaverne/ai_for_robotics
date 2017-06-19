#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 12:13:01 2017

@author: sebastian
"""
import numpy as np
import matplotlib.pyplot as plt

#Useful link:
# http://openclassroom.stanford.edu/MainFolder/DocumentPage.php?course=MachineLearning&doc=exercises/ex5/ex5.html

def logistic_function(w, x):
    # TODO implement the logistic function #DONE

    return 1 / (1 + np.exp(-np.dot(w.transpose(),x.transpose()))) # change this line #DONE

def h(w,x):

    return logistic_function(w, x)

def J_prime(w, x, y, l):
    out  = np.empty((x.shape[1],1));
    m = x.shape[0]
    y_temp = y.reshape((1,len(y)))

    w_temp = w
    w_temp[0] = 0

    for f in range(0, x.shape[1]):
        #For each feature f in an image xi
        out[f] = np.sum((np.dot(h(w, x) - y_temp, x[:,f]) + l * w_temp[f]),axis=0) / m

    return out

def H(w, x, l):
    out = np.zeros((x.shape[1], x.shape[1]));
    m = x.shape[0]

    for i in range(0, m):
        #For each image i
        xi = x[i].reshape((1,len(x[i])))
        dotx = np.dot(xi.transpose(), xi)
        temp3 = h(w, xi) * (1.0 - h(w, xi)) * dotx

        out += temp3

    eye = l * np.eye(out.shape[0]) / m
    eye[0,0] = 0
    out = out/m + eye

    print(out)

    return out

def w_kplus1(w_k, x, y):
    l = 0.5

    H_inv = np.linalg.inv(H(w_k, x, l))
    J_p = J_prime(w_k, x, y, l)
    return w_k - np.dot(H_inv,J_p)

# To make it easier the 24x24 pixels have been reshaped to a vector of 576 pixels. the value corrsponds to the greyscale intensity of the pixel
input_data = np.genfromtxt(open("XtrainIMG.txt"))  # This is an array that has the features (all 576 pixel intensities) in the columns and all the available pictures in the rows
output_data = np.genfromtxt(open("Ytrain.txt"))  # This is a vector that has the classification (1 for open eye 0 for closed eye) in the rows

n_samples = input_data.shape[0]
n_features = input_data.shape[1]

#Pad the input data with a bias term
input_data = np.hstack((np.ones((input_data.shape[0],1)),input_data))

ratio_train_validate = 0.8
idx_switch = int(n_samples * ratio_train_validate)
training_input = input_data[:idx_switch, :]
training_output = output_data[:idx_switch][:,None]
validation_input = input_data[idx_switch:, :]
validation_output = output_data[idx_switch:][:,None]

#TODO initialise w #DONE
w = np.zeros((n_features+1,1)) # change this line

#TODO implement the iterative calculation of w
i = 0;
max_iter = 100;
changed = True;
while (changed and (i < max_iter)):
    print('iter: {}'.format(i))
    w_old = w; #Save last weight vector

    w = w_kplus1(w_old, training_input, training_output)


    if np.allclose(w_old,w):
        #No update, have the minimum. Stop iterating.
        changed = False;
    i+=1 #Count up one

#TODO2: modify the algorithm to account for regularization as well to improve the classifier

#validation
h = logistic_function(w,validation_input)
output = np.round(h).transpose()

error = np.abs(output-validation_output).sum()

print('wrong classification of ',(error/output.shape[0]*100),'% of the cases in the validation set')


# classify test data for evaluation
test_input = np.genfromtxt(open("XtestIMG.txt"))
test_input = np.hstack((np.ones((test_input.shape[0],1)),test_input))
h = logistic_function(w,test_input)
test_output = np.round(h).transpose()
np.savetxt('results.txt', test_output)
