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
from sklearn.cluster import DBSCAN
from sklearn.metrics import f1_score
import time

np.random.seed(123)

# Input
n_cluster = 5
n = 1000
n_level = 50
eps = 3
min_samples = 60
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
t0 = time.time()
dbscan = DBSCAN(eps=eps, min_samples = min_samples)
clusters = dbscan.fit_predict(X)
t1 = time.time()

## plot the result
# array with all the color per cluster
color = []
for index,i in enumerate(clusters):
    if (i == 0):
        color.append('b')
    elif (i == 1):
        color.append('g')
    elif (i == 2):
        color.append('y')
    elif (i == 3):
        color.append('c')
    elif (i == 4):
        color.append('r')
    elif (i == 5):
        color.append('g')
    elif (i == 6):
        color.append('grey')
    elif (i == 12):
        color.append('pink')
    elif (i == -1):
        color.append('k')
    else:
        color.append('y')
del i, index

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

## valuation
vgl = pd.concat([pd.DataFrame(col),pd.DataFrame(clusters)], axis = 1)

print(f1_score(col, clusters, average = None))
print(f1_score(col[0:5*n-1], clusters[0:5*n-1],average = 'weighted'))
#print(t1-t0)
#vgl.to_excel (r'E:/Dropbox/Ausbildung/Master/Sem 4/Masterthesis/eval.xlsx')
