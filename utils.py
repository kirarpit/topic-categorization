#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 30 22:29:27 2018

@author: Arpit
"""
import os, csv
from scipy import sparse
import numpy as np
import matplotlib.pyplot as plt

"""
Returns the sparse version if it exists
otherwise it loads the data, converts it to sparse
and saves it for the future calls
"""
def load_data(filename, header=None):
    sparse_filename = filename + '.npz'
    
    if os.path.exists(sparse_filename):
        matrix = sparse.load_npz(sparse_filename)
    else:
        matrix = get_sparse_matrix(filename)
        sparse.save_npz(sparse_filename, matrix)
        
    return matrix

def get_sparse_matrix(filename):
    lines = []
    csvreader = csv.reader(open(filename))
    for line in csvreader:
        lines.append(list(map(int, line)))

    return sparse.csr_matrix(lines)

"""
Splits the data into training and validation set
according to the training set percentage
"""
def split_data(data, percent):
    t_rows = list(np.random.choice(data.shape[0], int(data.shape[0] * percent), replace=False))
    v_rows = list(set(list(range(data.shape[0])))^set(t_rows))
    return data[t_rows, :], data[v_rows, :]

def plot(Xs, Ys):
    fig = plt.figure()
    plt.plot(Xs, Ys)
    plt.ylabel('% Accuracy')
    plt.xlabel('Log(beta)')
    plt.savefig("accuracy-beta.png")
    plt.close(fig)
    
def save_preds(data, filename):
    f = open(filename,'w')
    f.write("id,class\n")
    
    idx = 12001
    for i in range(len(data)):
        f.write(str(idx) + "," + str(data[i]) + "\n")
        idx += 1
    f.close()