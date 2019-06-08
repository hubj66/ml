# -*- coding: utf-8 -*-
"""
Created on Fri May 31 16:20:27 2019

@author: hubj66
"""
import numpy as np
import pandas as pd
import random
from sklearn.preprocessing import MinMaxScaler

class dataset_generator:
    
    
    def generator(n_points, dimension = None, mean = 0, sd = 1, min_value = 0, max_value = 1, feature_type = "numeric", distribution = "normal", transform = True):
        
        if (dimension == None):
            dimension = len(mean)
        
        data_combined = pd.DataFrame()
        for d in range(dimension):
            if (type(mean) == list):
                mean_d = mean[d]
            else:
                mean_d = mean
            if (type(sd) == list):
                sd_d = sd[d]
            else:
                sd_d = sd
            if (type(min_value) == list):
                min_value_d = min_value[d]
            else:
                min_value_d = min_value
            if (type(max_value) == list):
                max_value_d = max_value[d]
            else:
                max_value_d = max_value
            if (type(distribution) == list):
                distribution_d = distribution[d]
            else:
                distribution_d = distribution
            if (type(feature_type) == list):
                feature_type_d = feature_type[d]
            else:
                feature_type_d = feature_type
            
            
            if (feature_type_d == "numeric"):
                data = dataset_generator.__numeric_gen(n_points, mean_d, sd_d, min_value_d, max_value_d, distribution_d)
            elif (feature_type_d == "binary"):
                data = dataset_generator.__binary_gen(n_points, mean_d)
            elif (feature_type_d == "categorial"):
                data = dataset_generator.__categorical_gen(n_points)
            if (transform and not(min(data)  == max(data))):
                min_max_scaler = MinMaxScaler()
                data = min_max_scaler.fit_transform(data.reshape(-1, 1))
            data_combined = pd.concat([data_combined,pd.DataFrame(data)], axis = 1)
        return (data_combined)
    
    
    def __numeric_gen(n_points, mean, sd, min_value, max_value, distribution):
        if (distribution == "normal"):
            data = np.random.normal(mean, sd, n_points)
            return (data)
        elif (distribution == "uniform"):
            data = np.random.uniform(min_value, max_value, n_points)
            return (data)
        else:
            return ("wrong distribution input")
    
    
    def __binary_gen(n_points, mean):
        return (pd.DataFrame([mean] * n_points))
    
    
    def __categorical_gen(n_points, seed_nr = 1):
        cat_1 = ["paris", "rom", "london", "amsterdam", "zuerich", "berlin"]
        cat_2 = ["wien", "madrid", "kopenhagen", "riga", "dublin", "prag"]
        cat_3 = ["athen", "zagreb", "helsinki", "warschau", "stockholm", "monaco"]
        cat_all = []
        for i in range(len(cat_1)):
            cat_all.append(cat_1[i])
        for i in range(len(cat_2)):
            cat_all.append(cat_2[i])
        for i in range(len(cat_3)):
            cat_all.append(cat_3[i])
        
        if (seed_nr == 1):
            data = cat_1
        elif (seed_nr == 2):
            data = cat_2
        elif (seed_nr == 3):
            data = cat_3
        else:
            data = cat_all
        
        dataset = []
        for i in range(n_points):
            dataset.append(''.join(random.sample(data,1)))
        
        return (dataset)
        

