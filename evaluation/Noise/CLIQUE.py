# -*- coding: utf-8 -*-
"""
Created on Fri May 31 17:16:14 2019

@author: Eaisy
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cluster_evaluation.data_generator.data_generator import dataset_generator as dg
from sklearn.decomposition import PCA
from pyclustering.cluster.clique import clique, clique_visualizer
from sklearn.metrics import f1_score
import time

np.random.seed(123)

# Input
n_cluster = 5
n = 1000
n_level = 10
intervall = 20
threshold = 20
n_noise = int(n*n_cluster*n_level)
D1 = 3
D2 = 0
D = D1 + D2

# Data generation process
X1 = dg.generator(n_points = n, dimension = D1, min_value = 5, max_value = 15, feature_type = "numeric", distribution = "uniform", transform = False)
X2 = dg.generator(n_points = n, dimension = D1, min_value = 20, max_value = 35, feature_type = "numeric", distribution = "uniform", transform = False)
X3 = dg.generator(n_points = n, dimension = D1, min_value = 40, max_value = 55, feature_type = "numeric", distribution = "uniform", transform = False)
X4 = dg.generator(n_points = n, dimension = D1, min_value = 60, max_value = 75, feature_type = "numeric", distribution = "uniform", transform = False)
X5 = dg.generator(n_points = n, dimension = D1, min_value = 80, max_value = 95, feature_type = "numeric", distribution = "uniform", transform = False)
noise = dg.generator(n_points = n_noise, dimension = (D), min_value = 0, max_value = 100, feature_type = "numeric", distribution = "uniform", transform = False)
X = pd.concat([X1,X2,X3,X4,X5, noise], axis = 0)
del X1, X2, X3, X4, X5

# correct cluster labels in an array
col = []
for i in range(n_cluster):
    for j in range(n):
        col.append(i)
for i in range(n_noise):
    col.append(-1)
col = (col)
del i,j 

# apply DBSCAN multiple times
X_cliq = X.values.tolist()
t0 = time.time()
clique_instance = clique(X_cliq, intervall, threshold)
clique_instance.process()
clusters = clique_instance.get_clusters()
noise_c = clique_instance.get_noise()
t1 = time.time()
clusters.append(noise_c)

## plot the result
# array with all the color per cluster
color = []
help_color = np.array([np.zeros(len(X))]*2).T
counter = 0
for j in range(len(clusters)):
    for index,i in enumerate(clusters[j]):
        help_color[counter,0] = i
        help_color[counter,1] = j
        counter += 1
help_color = help_color[help_color[:,0].argsort()]#.sort(axis = 0)
for j in range(len(X)):
    i = help_color[j,1]
    if (i == 0):
        color.append('b')
    elif (i == 1):
        color.append('g')
    elif (i == 2):
        color.append('r')
    elif (i == 3):
        color.append('c')
    elif (i == 4):
        color.append('pink')
    else:
        color.append('k')
for j in range(len(X)):
    if (help_color[j,1] == 5):
        help_color[j,1] = -1
del i, index, j

# apply PCA to plot in 2D
if (D==2):
    X_2D=X
if (D==3):
    pca = PCA(n_components=2)
    pca.fit(X)
    X_2D = pca.transform(X)
    X_2D = pd.DataFrame(X_2D)

fig = plt.figure()
plt.scatter(X_2D.iloc[5*n:len(X), 0], X_2D.iloc[5*n:len(X), 1], s = 20, marker = 'x', c=color[5*n:len(X)])
plt.scatter(X_2D.iloc[0:n-1, 0], X_2D.iloc[0:n-1, 1], s = 20, marker = 's', c=color[0:n-1])
plt.scatter(X_2D.iloc[n:2*n-1, 0], X_2D.iloc[n:2*n-1, 1], s = 20, marker = '^', c=color[n:2*n-1])
plt.scatter(X_2D.iloc[2*n:3*n-1, 0], X_2D.iloc[2*n:3*n-1, 1], s = 20, marker = 'o', c=color[2*n:3*n-1])
plt.scatter(X_2D.iloc[3*n:4*n-1, 0], X_2D.iloc[3*n:4*n-1, 1], s = 20, marker = 'p', c=color[3*n:4*n-1])
plt.scatter(X_2D.iloc[4*n:5*n-1, 0], X_2D.iloc[4*n:5*n-1, 1], s = 20, marker = '1', c=color[4*n:5*n-1])
plt.show()

#cells = clique_instance.get_cells()
#clique_visualizer.show_grid(cells, X_cliq)    # show grid that has been formed by the algorithm
#clique_visualizer.show_clusters(X_cliq, clusters, noise)

## valuation
vgl = pd.concat([pd.DataFrame(col),pd.DataFrame(help_color[:,1])], axis = 1)

print(f1_score(col, help_color[:,1], average = None))
print(f1_score(col[0:5*n-1], help_color[0:5*n-1,1],average = 'weighted'))
print(t1-t0)
#vgl.to_excel (r'E:/Dropbox/Ausbildung/Master/Sem 4/Masterthesis/eval.xlsx')
