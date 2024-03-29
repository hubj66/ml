# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 12:42:19 2019

@author: hubj66
"""

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

def within(data, label):
    label = np.asarray(label)
    n_clusters = len(set(label))
    average_distance = 0
    max_distance = np.amax(euclidean_distances(data))
    for nc in range(n_clusters):
        data_cluster = data[label == nc]
        n = len(data_cluster)
        distance_matrix = euclidean_distances(data_cluster)
        average_distance += sum(sum(distance_matrix))/2/(n*(n+1))*2
    average_distance = average_distance / n_clusters
    standardisized_distance = 1 - average_distance / max_distance
return (standardisized_distance)
