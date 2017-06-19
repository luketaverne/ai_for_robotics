#!/usr/bin/env python2
import numpy as np
import random
from scipy import misc
from scipy import ndimage
from sklearn import preprocessing
import matplotlib.pyplot as plt
from PIL import Image
import os
from sklearn.utils.extmath import row_norms
from sklearn.metrics.pairwise import euclidean_distances

def clustering(X, mu):
    # TODO: Perform optimization of r_nk and fill datapoints into clusters[k] k = 0,...,1-K.
    clusters  = [[] for x in range(len(mu))]
    for point_index in range(X.shape[0]):
        min_index = np.argmin(np.mean(abs(mu - X[point_index]), axis=1))
        clusters[min_index].append(X[point_index])

    return clusters

def reevaluate_mu(mu, clusters):
    # TODO: Perform optimization of mu_k and return optimized mu_k.
    newmu = [[] for x in range(len(clusters))]
    for cluster_index in range(len(clusters)):
        points_in_cluster = clusters[cluster_index]

        newmu[cluster_index] = np.mean(points_in_cluster, axis=0)

    return newmu

def has_converged(mu, oldmu):
    return set([tuple(j) for j in mu]) == set([tuple(j) for j in oldmu])

def k_means_plus_plus_init(X, n_clusters):
    #Adapted from sklearn
    # https://github.com/scikit-learn/scikit-learn/blob/14031f6/sklearn/cluster/k_means_.py#L704

    centers = np.empty((n_clusters,X.shape[1]))
    x_squared_norms = row_norms(X)
    # n_local_trials = 2 + int(np.log(n_clusters))
    n_local_trials = 1000
    random_state = np.random

    # Select first cluster at random
    centers[0] = np.asarray(random.sample(X, 1))

    # Initialize list of closest distances and calculate current potential
    closest_dist_sq = euclidean_distances(
        centers[0, np.newaxis], X, Y_norm_squared=x_squared_norms,
        squared=True)
    current_pot = closest_dist_sq.sum()

    # Pick the remaining n_clusters-1 points
    for c in range(1, n_clusters):
        # Choose center candidates by sampling with probability proportional
        # to the squared distance to the closest existing center
        rand_vals = random_state.random_sample(n_local_trials) * current_pot
        candidate_ids = np.searchsorted(closest_dist_sq.cumsum(), rand_vals)

        # Compute distances to center candidates
        distance_to_candidates = euclidean_distances(
            X[candidate_ids], X, Y_norm_squared=x_squared_norms, squared=True)

        # Decide which candidate is the best
        best_candidate = None
        best_pot = None
        best_dist_sq = None
        for trial in range(n_local_trials):
            # Compute potential when including center candidate
            new_dist_sq = np.minimum(closest_dist_sq,
                                     distance_to_candidates[trial])
            new_pot = new_dist_sq.sum()

            # Store result if it is the best local trial so far
            if (best_candidate is None) or (new_pot < best_pot):
                best_candidate = candidate_ids[trial]
                best_pot = new_pot
                best_dist_sq = new_dist_sq

        # Permanently add best center candidate found in local tries
        centers[c] = X[best_candidate]

        current_pot = best_pot
        closest_dist_sq = best_dist_sq


    return centers

def find_centers(X, K):
    # Initialize to K random centers.
    # TODO: Robustify initialization towards global convergence (Exchange initialization of mu).
    mu = k_means_plus_plus_init(X, K)
    oldmu = np.empty(mu.shape)

    while not has_converged(mu, oldmu):
        oldmu = mu
        # First step of optimization: Assign all datapoints in X to clusters.
        clusters = clustering(X, mu)
        # Second step of optimization: Optimize location of cluster centers mu.
        mu = reevaluate_mu(oldmu, clusters)

    return(mu, clusters)

# Load precomputed (PCA) features from file.
# WARNING: These are different from the results of the first exercise! Use the provided features file!
features = np.genfromtxt(open("features_k_means.txt"))
# Make sure to normalize your data, you may run into numerical issues otherwise.
features = preprocessing.scale(features)
n_samples, n_features = np.shape(features)
# Initialize centers
initial_mu = random.sample(features, 3)

# Perform Lloyd's algorithm.
mu, clusters = find_centers(features, 3)

# Plot results.
for x in range(len(clusters[0])): plt.plot(clusters[0][x][0], clusters[0][x][1], 'o', markersize=7, color='blue', alpha=0.5, label='Cluster 1')
for x in range(len(clusters[1])): plt.plot(clusters[1][x][0], clusters[1][x][1], 'o', markersize=7, color='red', alpha=0.5, label='Cluster 2')
for x in range(len(clusters[2])): plt.plot(clusters[2][x][0], clusters[2][x][1], 'o', markersize=7, color='green', alpha=0.5, label='Cluster 3')
plt.plot([mu[0][0], mu[1][0], mu[2][0]], [mu[0][1], mu[1][1], mu[2][1]], '*', markersize=20, color='red', alpha=1.0, label='Cluster centers')
plt.xlabel('principal component 1')
plt.ylabel('principal component 2')
plt.title('K-means clustering')
plt.show()

# Save results.
np.savetxt('results_k_means.txt', mu)
