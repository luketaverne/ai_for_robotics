#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 2 10:00 2017
@author: Timo Hinzmann (hitimo@ethz.ch)
"""

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import numpy as np
from matplotlib import pyplot
import pylab
from mpl_toolkits.mplot3d import Axes3D
import copy
from numpy import linalg as LA
from sklearn.neighbors import NearestNeighbors


def getNearestNeighbors(sourceT, targetT):
    neigh = NearestNeighbors(n_neighbors=1).fit(targetT)
    distances, indices = neigh.kneighbors(sourceT, return_distance=True)

    mean_error = np.sum(distances.ravel()) / distances.ravel().size

    return indices.ravel(), mean_error

def computeBestTransformation(source, target):
    source_bar = np.mean(source, axis=0)
    target_bar = np.mean(target, axis=0)

    R_hat = computeBestRotation(source, source_bar, target, target_bar)
    t_hat = computeBestTranslation(source_bar, target_bar, R_hat)
    return getTransformationMatrix(R_hat, t_hat)

def getTransformationMatrix(R, t):
    T = np.eye(4)
    T[0:3,0:3] = R
    T[0:3,3] = t
    return T

def computeBestTranslation(source_bar, target_bar, R):
    t_opt = target_bar.T - np.dot(R,source_bar.T)
    return t_opt

def computeBestRotation(source, source_bar, target, target_bar):
    sum = np.empty((3,3));

    source_zeroed = source - source_bar
    target_zeroed = target - target_bar

    # rotation matrix
    H = np.dot(source_zeroed.T, target_zeroed)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # special reflection case
    if np.linalg.det(R) < 0:
       Vt[2,:] *= -1
       R = np.dot(Vt.T, U.T)
    return R

def make_plot(source, target):
    fig = pylab.figure()
    ax = Axes3D(fig)
    pyplot.cla()
    ax.scatter(source[0,:], source[1,:], source[2,:], color='red')
    ax.scatter(target[0,:], target[1,:], target[2,:], color='green')
    pyplot.draw()
    ax.view_init(azim=69, elev=-97)
    pyplot.show(block=True)

def do_transform(Tr, source2T):
    source2T = np.dot(Tr, source2T)

    return source2T


def main():
    source_original = np.genfromtxt(open("synthetic_source.xyz"))
    target_original = np.genfromtxt(open("synthetic_target.xyz"))
    source = np.ones((4, source_original.shape[0])).astype(np.float32)
    source_orig = np.ones((4, source_original.shape[0])).astype(np.float32)
    target = np.ones((4, target_original.shape[0])).astype(np.float32)
    # print(source_original.shape)
    source[0:3,:] = np.copy(source_original.T)
    source_orig[0:3,:] = np.copy(source_original.T)
    target[0:3,:] = np.copy(target_original.T)

    # Initialize.
    iter_max = 100
    convergence_tolerance = 1.0e-16
    previous_mean_error = 1.0e12

    for iter in range(0, iter_max):
        if iter == iter_max - 1:
            make_plot(source, target)

        # Get correspondences.
        target_indices, current_mean_error = getNearestNeighbors(source[0:3,:].T, target[0:3,:].T)

        # Compute best transformation.
        T = computeBestTransformation(source[0:3,:].T,target[0:3,target_indices].T)

        # Transform the source pointcloud.
        source = do_transform(T, source)

        # Check convergence.
        if abs(previous_mean_error - current_mean_error) < convergence_tolerance:
            print "Converged at iteration: ", iter
            make_plot(source, target)
            break
        else:
            print('Mean error: ', current_mean_error)
            previous_mean_error = current_mean_error


    T_final = computeBestTransformation(source_original, source[0:3,:].T)

    print "Final transformation: ", T_final
    np.savetxt('results_alignment_synthetic.txt', T_final)

if __name__ == "__main__":
    main()
