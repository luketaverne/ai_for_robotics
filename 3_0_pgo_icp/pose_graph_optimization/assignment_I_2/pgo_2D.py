#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 2 10:00 2017
@author: Timo Hinzmann (hitimo@ethz.ch)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import linalg as sla
from scipy import array, linalg, dot
from enum import Enum
import copy
import pylab
from scipy.linalg import cho_factor, cho_solve


class PoseGraphOptimization2D():
    def __init__(self, vertices, edges, lc):
        # Position estimates
        self.vertices = vertices
        self.edges = edges
        self.lc = lc
        self.num_nodes = vertices.shape[0]
        self.num_edges = edges.shape[0]
        self.num_lc = lc.shape[0]
        self.x = np.zeros([3*self.num_nodes, 1])

    def optimizePoseGraph(self):

        for i in range(self.num_nodes):
            self.x[3 * i:3 * (i + 1)] = self.vertices[i, 1:4].reshape(3,1)

        num_optimization_iterations = 2
        for iter in range(0, num_optimization_iterations):
            H = np.zeros([3*self.num_nodes, 3*self.num_nodes])
            b = np.zeros([3*self.num_nodes, 1])

            for matrix in [self.edges, self.lc]:
                for vector in matrix:
                    i = int(vector[0])
                    j = int(vector[1])

                    phi_i = float(self.x[3 * i + 2])
                    phi_j = float(self.x[3 * j + 2])
                    phi_ij = float(vector[4])

                    R_i = np.array([[np.cos(phi_i), -np.sin(phi_i)], [np.sin(phi_i), np.cos(phi_i)]])
                    dif_R_i = np.array([[-np.sin(phi_i), -np.cos(phi_i)], [np.cos(phi_i), -np.sin(phi_i)]])
                    R_ij = np.array([[np.cos(phi_ij), -np.sin(phi_ij)], [np.sin(phi_ij), np.cos(phi_ij)]])

                    B_ij_ul = np.dot(R_ij.T, R_i.T)
                    B_ij_ur = np.zeros((2, 1))
                    B_ij_u = np.hstack((B_ij_ul, B_ij_ur))
                    B_ij_l = np.array([0, 0, 1])
                    B_ij = np.vstack((B_ij_u, B_ij_l))

                    t_i = self.x[3 * i:3 * i + 2].reshape(2)
                    t_j = self.x[3 * j:3 * j + 2].reshape(2)
                    t_ij = vector[2:4].reshape(2)

                    A_ij_ul = np.dot(-R_ij.T, R_i.T)
                    A_ij_ur = np.dot(np.dot(R_ij.T, dif_R_i.T), t_j - t_i).reshape((2, 1))
                    A_ij_u = np.hstack((A_ij_ul, A_ij_ur))
                    A_ij_l = np.array([0, 0, -1])
                    A_ij = np.vstack((A_ij_u, A_ij_l))

                    e_ij_u = np.dot(R_ij.T, np.dot(R_i.T, t_j - t_i) - t_ij).reshape((2, 1))
                    e_ij_l = [phi_j - phi_i - phi_ij]
                    e_ij = np.vstack((e_ij_u, e_ij_l))

                    Omega_ij = np.array([vector[5:8], [vector[6], vector[8], vector[9]], [vector[7], vector[9], vector[10]]])


                    b[3 * i:3 * (i + 1)] += np.dot(np.dot(A_ij.T, Omega_ij), e_ij)
                    b[3 * j:3 * (j + 1)] += np.dot(np.dot(B_ij.T, Omega_ij), e_ij)

                    # Hessian
                    H[3 * i:3 * (i + 1), 3 * i:3 * (i + 1)] += np.dot(np.dot(A_ij.T, Omega_ij), A_ij)
                    H[3 * i:3 * (i + 1), 3 * j:3 * (j + 1)] += np.dot(np.dot(A_ij.T, Omega_ij), B_ij)
                    H[3 * j:3 * (j + 1), 3 * i:3 * (i + 1)] += np.dot(np.dot(B_ij.T, Omega_ij), A_ij)
                    H[3 * j:3 * (j + 1), 3 * j:3 * (j + 1)] += np.dot(np.dot(B_ij.T, Omega_ij), B_ij)

            H[0:3, 0:3] += np.eye(3)

            # Sparse Cholesky Factorization
            factor = cho_factor(H)
            self.x += cho_solve(factor,-b)

        self.x = np.reshape(self.x, (self.num_nodes, 3))

def main():

    vertices = np.genfromtxt(open("vertices.dat"))
    edges = np.genfromtxt(open("edges.dat"))
    lc = np.genfromtxt(open("loop_closures.dat"))

    pylab.plot(vertices[:,1], vertices[:,2], 'b')
    plt.pause(5)

    pgo = PoseGraphOptimization2D(vertices, edges, lc)
    pgo.optimizePoseGraph()

    # Save the optimized states in rows: [x_0, y_0, th_0; x_1, y_1, th_1; ...]
    x_opt = pgo.x
    np.savetxt('results_2D.txt', np.transpose(x_opt))

if __name__ == "__main__":
    main()
