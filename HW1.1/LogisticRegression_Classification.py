#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 15:13:44 2021

@author: jialichen
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.optimize import minimize

class LogisticRegression_Classification:
    def __init__(self, datafile = "weight.json", model_type = None):
        self.df = pd.read_json('weight.json')
       
    def normalize(self):
        x_mean = np.mean(self.df['y'])
        y_mean = np.mean(self.df['is_adult'])
        x_std = np.std(self.df['y'])
        y_std = np.std(self.df['is_adult'])    
        x = (self.df['y'] - x_mean)/x_std
        y = (self.df['is_adult'] - y_mean)/y_std  
        return x, y
    # split into train test sets
    def split(self, test_size):
        x = np.array(self.df[['y']])
        y = np.array(self.df['is_adult']) 
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size = test_size, random_state = 123)
        return self.x_train, self.x_test, self.y_train, self.y_test
    # calculate y
    def predict(self, par, x_train): #par will be a list, first element will be a, second element will be b
        A = par[0]
        w = par[1]
        x0 = par[2]
        s = par[3]
        y = [A/(1+np.e**(-(el-x0)/w))+s for el in x_train]
        return y
    
    def loss(self, par):
        result = self.split(0.2)
        y_predict = self.predict(par, result[0])
        mse = 0
        for i in range(len(y_predict)):
            mse += (y_predict[i] - result[2][i])**2
        return mse
    
    
    def optimize(self):
        solution = minimize(self.loss,[0,0,0,0],method='Nelder-Mead')
        return solution.x
    
    def plot(self):
        p = self.optimize() 
        result = self.split(0.2)
        y_predict = p[0]/(1+np.e**(-(result[0]-p[2])/p[1]))+p[3]
        plt.figure(1)
        plt.plot(result[0],result[2],'bo')
        plt.plot(result[0],y_predict, color = 'r')
        plt.xlabel('Weight (lbs)')
        plt.ylabel('Is Adult')
        plt.title("Logistic Regression - Classification")
        plt.legend(['Measured','Predicted'],loc='best')
        plt.show()
        
    def get_x_y(self): 
        result = self.split(0.2)
        x_train=result[0]
        y_train=result[2]
        return x_train, y_train
    
if __name__ == '__main__':
    model_instance = LogisticRegression_Classification()
    model_instance.optimize()
    model_instance.plot()
