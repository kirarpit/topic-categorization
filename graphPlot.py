#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 17:08:33 2018

@author: Arpit
"""
import numpy as np
import matplotlib.pyplot as plt

class GraphPlot:
    
    def __init__(self, name="", xlabel="X", ylabel="Y"):
        self.name = name
        self.xlabel = xlabel
        self.ylabel = ylabel
        
        self.plots = []
        self.labels = []
        
    def addPlot(self, data):
        self.plots.append(data)
    
    def addLabel(self, label):
        self.labels.append(str(label))
            
    def plot(self):
        fig = plt.figure()
        
        for i in range(len(self.plots)):
            plot = np.array(self.plots[i])
            X = list(plot[:, 0])
            Y = list(plot[:, 1])
            plt.plot(X, Y, label=self.labels[i] if len(self.labels) != 0 else i)
            plt.ylabel(self.ylabel)
            plt.xlabel(self.xlabel)

        plt.legend(loc = "best")
        plt.savefig("/Users/Arpit/Desktop/" + self.name + 'chart.png')
        plt.close(fig)