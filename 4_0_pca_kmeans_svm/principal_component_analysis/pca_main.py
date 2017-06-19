#!/usr/bin/env python2
# Source: http://sebastianraschka.com/Articles/2014_pca_step_by_step.html
#         https://stats.stackexchange.com/questions/2691/making-sense-of-principal-component-analysis-eigenvectors-eigenvalues
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt

# Load the dataset of orb features from file.
orb_features = np.genfromtxt(open("orb_features.txt"))
orb_features = orb_features.T

orb_size = len(orb_features[:, 0])
cov = np.cov(orb_features)

eig_val_cov, eig_vec_cov = np.linalg.eig(cov)

# Sort eigenvectors and corresponding eigenvalues in descending order.
eig_pairs = [(np.abs(eig_val_cov[i]), eig_vec_cov[:, i]) for i in range(len(eig_val_cov))]
eig_pairs.sort(key=lambda x: x[0], reverse=True)

best_eig_vec = np.hstack(
                            (
                            eig_pairs[0][1].reshape((-1,1)),
                            eig_pairs[1][1].reshape((-1,1)),
                            eig_pairs[2][1].reshape((-1,1)),
                            eig_pairs[3][1].reshape((-1,1)),
                            eig_pairs[4][1].reshape((-1,1))
                            )
                        )

pca_features = orb_features.T.dot(best_eig_vec)

# Normalize pca features.
pca_features = preprocessing.scale(pca_features)

# 2D plot of first 2 principal components.
plt.scatter(pca_features[:, 0], pca_features[:, 1], marker = 'o')
plt.xlabel('principal component 1')
plt.ylabel('principal component 2')
plt.title('PCA result')
plt.show()

# Save results.
np.savetxt('results_pca.txt', pca_features)
