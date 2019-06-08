# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 15:52:04 2019

@author: hubj66
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from synthetic_data.data_generator import dataset_generator as dg
import time
from sklearn.cluster import DBSCAN
from sklearn.metrics import f1_score
from itertools import cycle, islice
from sklearn.decomposition import PCA

np.random.seed(123)

clusters = [[[ 5,15],[20,40],[20,40],[20,40],[60,80]],
            [[20,35],[60,80],[20,40],[20,40],[20,40]],
            [[40,55],[60,80],[60,80],[20,40],[20,40]],
            [[60,75],[60,80],[60,80],[60,80],[60,80]],
            [[80,95],[20,40],[60,80],[60,80],[60,80]]]

#eps and min_sample
parameter = {'1': [15, 14],
             '2': [20, 15],
             '3': [25, 14],
             '5': [39, 8]}

# Input
n_cluster = 5
n = 1000
n_level = 0.1
n_noise = int(n*n_cluster*n_level)
D1 = len(clusters[0])
D2 = 5
D = D1 + D2
eps = parameter[str(D2)][0]
min_samples = parameter[str(D2)][1]

#generate data set
X = pd.DataFrame([])
for i in range(n_cluster):
    min_value = np.zeros(len(clusters[0]))
    max_value = np.zeros(len(clusters[0]))
    for j in range(len(clusters[0])):
        min_value[j] = clusters[i][j][0]
        max_value[j] = clusters[i][j][1]
    min_value = min_value.tolist()
    max_value = max_value.tolist()
    X_temp = dg.generator(n_points = n, dimension = D1, min_value = min_value, max_value = max_value, feature_type = "numeric", distribution = "uniform", transform = False)
    if (D2 != 0):
        X_temp_noise = dg.generator(n_points = n, dimension = D2, min_value = 0, max_value = 100, feature_type = "numeric", distribution = "uniform", transform = False)
        X_temp = pd.concat([X_temp, X_temp_noise], axis = 1)
        del X_temp_noise
    X = pd.concat([X, X_temp], axis = 0)
    del X_temp
noise = dg.generator(n_points = n_noise, dimension = (D), min_value = 0, max_value = 100, feature_type = "numeric", distribution = "uniform", transform = False)
X = pd.concat([X, noise], axis = 0)

# correct cluster labels in an array
label_true = []
for i in range(n_cluster):
    for j in range(n):
        label_true.append(i)
for i in range(n_noise):
    label_true.append(-1)
label_true = (label_true)
del i,j 

# apply DBSCAN multiple times
t0 = time.time()
dbscan = DBSCAN(eps=eps, min_samples = min_samples)
label_pred = dbscan.fit_predict(X)
t1 = time.time()


colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                     '#f781bf', '#a65628', '#984ea3',
                                     '#999999', '#e41a1c', '#dede00']),
                              int(max(label_pred) + 1))))
colors = np.append(colors, ["#000000"])
marker = ['s','^','o','p','1']

# apply PCA to plot in 2D
if (D==2):
    X_2D=X
if (D>2):
    pca = PCA(n_components=2)
    pca.fit(X)
    X_2D = pca.transform(X)
    X_2D = pd.DataFrame(X_2D)

fig = plt.figure()
plt.scatter(X_2D.iloc[n_cluster*n:len(X), 0], X_2D.iloc[n_cluster*n:len(X), 1], s = 20, marker = 'x', c=colors[label_pred[n_cluster*n:len(X)]])
for i in range(n_cluster):
    plt.scatter(X_2D.iloc[i*n:(i+1)*n-1, 0], X_2D.iloc[i*n:(i+1)*n-1, 1], s = 20, marker = marker[i], c=colors[label_pred[i*n:(i+1)*n-1]])
plt.show()

# F-Score
f_clusters = f1_score(label_true, label_pred, average = None)
print('f_clusters:', f_clusters)
f_average = f1_score(label_true[0:n_cluster*n-1], label_pred[0:n_cluster*n-1],average = 'weighted')
print('f_average:', f_average)
