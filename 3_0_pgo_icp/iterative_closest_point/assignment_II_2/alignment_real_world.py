#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 2 10:00 2017
@author: Timo Hinzmann (hitimo@ethz.ch)
"""

import numpy as np
from matplotlib import pyplot
import pylab
from mpl_toolkits.mplot3d import Axes3D
import copy
from numpy import linalg as LA
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt, numpy as np, numpy.random, scipy
from sklearn.neighbors import NearestNeighbors

def getNearestNeighbors(sourceT, targetT, thresh=1):

    neigh = NearestNeighbors(n_neighbors=1).fit(targetT)
    distances, indices = neigh.kneighbors(sourceT, return_distance=True)

    indices = indices.ravel()
    distances = distances.ravel()
    print(distances.shape)

    source_indices = np.asarray(np.where(distances < thresh)).reshape(-1)
    target_indices = indices[source_indices]

    mean_error = np.sum(distances[source_indices]) / distances[source_indices].size

    return target_indices, source_indices, mean_error

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
    print('source shape: ', source.shape)
    print('target shape: ', target.shape)
    print('H shape:', H.shape)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # special reflection case
    if np.linalg.det(R) < 0:
       Vt[2,:] *= -1
       R = np.dot(Vt.T, U.T)
    return R

def make_plot(source, target):
    # Plotting.
    fig = pylab.figure()
    ax = Axes3D(fig)
    # Visualize only every 10th point. You can change this for quicker visualization.
    n = 10
    source_vis = np.copy(source[:,::n])
    target_vis = np.copy(target[:,::n])
    ax.scatter(source_vis[0,:], source_vis[1,:], source_vis[2,:], color='red', lw = 0, s=1)
    ax.scatter(target_vis[0,:], target_vis[1,:], target_vis[2,:], color='green', lw = 0, s=1)
    # Make sure that the aspect ratio is equal for x/y/z axis.
    X = target_vis[0,:]
    Y = target_vis[1,:]
    Z = target_vis[2,:]
    max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() / 2.0
    mid_x = (X.max()+X.min()) * 0.5
    mid_y = (Y.max()+Y.min()) * 0.5
    mid_z = (Z.max()+Z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    ax.set_aspect('equal')
    pyplot.show(block=True)

def do_transform(Tr, source2T):
    source2T = np.dot(Tr, source2T)

    return source2T

def main():
    source_original = np.genfromtxt(open("vision_source.xyz"))
    target_original = np.genfromtxt(open("laser_target.xyz"))
    source = np.ones((4, source_original.shape[0]))
    target = np.ones((4, target_original.shape[0]))
    source[0:3,:] = np.copy(source_original[:,0:3].T)
    target[0:3,:] = np.copy(target_original[:,0:3].T)

    # Align 'source' to 'target' pointcloud and compute the final transformation.
    # Feel free to reuse code from the previous exercise.
    # Initialize.
    iter_max = 1
    convergence_tolerance = 1.0e-16
    previous_mean_error = 1.0e12
    T_init = np.eye(4)
    T_init[0:3,3] = np.asarray([0,3,-29]).T
    thresh = 0.03

    source = do_transform(T_init, source)

    for iter in range(0, iter_max):
        # if thresh > 0.0005 and (iter+1) % 5 == 0:
        #     thresh = thresh/2.0
        #     print('thresh raised: ', thresh)
        if iter == iter_max - 1:
            make_plot(source, target)

        # Get correspondences.
        target_indices, source_indices, current_mean_error = getNearestNeighbors(source[0:3,:].T, target[0:3,:].T, thresh)

        # Compute best transformation.
        T = computeBestTransformation(source[0:3,source_indices].T,target[0:3,target_indices].T)

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

    # Don't forget to save the final transformation.
    print "Final transformation: ", T_final
    np.savetxt('results_alignment_real_world.txt', T_final)

if __name__ == "__main__":
    main()
