#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 13:57:18 2018

@author: Arpit
"""

from utils import load_data, save_preds
from naiveBayes import NaiveBayes
import math, random
import numpy as np
import operator, copy
from sklearn.feature_selection import VarianceThreshold
from random import choices

def get_gini_index(data):
    total = sum(data.values())
    
    result = 1
    for value in list(data.values()):
        result -= math.pow(value/total, 2)
    
    return result

def split_data(data, percent):
    t_rows = list(np.random.choice(data.shape[0], int(data.shape[0] * percent), replace=False))
    v_rows = list(set(list(range(data.shape[0])))^set(t_rows))
    
    t_columns = list(np.random.choice(data.shape[1], int(data.shape[1] * 1), replace=False))
    if 0 not in t_columns: t_columns.append(0)
    if data.shape[1] - 1 not in t_columns: t_columns.append(data.shape[1]-1)
    t_columns.sort()

    t_data = data[t_rows, :]
    t_data = t_data[:, t_columns]
    
    v_data = data[v_rows, :]
    v_data = v_data[:, t_columns]
    
    return t_rows, v_rows, t_columns, t_data, v_data

def count_votes(rows, preds):
    for i, rowId in enumerate(rows):
        if rowId not in votes:
            votes[rowId] = {}
        
        row_votes = votes[rowId]
        if preds[i] in row_votes:
            row_votes[preds[i]] += 1
        else:
            row_votes[preds[i]] = 1

data = load_data('training.csv')
#sel = VarianceThreshold()
#data = sel.fit_transform(data)

NBs = []
t_rows = {} #contains the rows selected for ith NB
t_columns = {}  #contains the features selected for ith tree
v_rows = {}
v_data = {}

for i in range(100):
    t_rows[i], v_rows[i], t_columns[i], t_data, v_data[i] = split_data(data, 0.85)
    nb = NaiveBayes(t_data, var_red=False)
    lgx = -1 *random.uniform(1.8,2.2)
#    lgx = -2
    print(lgx)
    nb.train(math.pow(10, lgx))
    NBs.append(nb)

votes = {}
for i in range(len(NBs)):
    valid_data = NBs[i].preprocess_testing_data(v_data[i][:, :-1])
    preds = list(NBs[i].predict(valid_data))
    count_votes(v_rows[i], preds)

rows = list(votes.keys())
rows.sort()
labels = list(data[rows, -1].toarray()[:,0])

#check accuracy
cnt = 0
for i, row in enumerate(rows):
    pred_label = max(votes[row].items(), key=operator.itemgetter(1))[0]
    if pred_label == labels[i]:
        cnt += 1

print("Accuracy:", cnt/len(labels))

#testing data
test_data = load_data('testing.csv')
test_data = NBs[0].preprocess_testing_data(test_data)

votes = {}
for i in range(len(NBs)):
    preds = list(NBs[i].predict(test_data))
    row = list(range(test_data.shape[0]))
    print(len(row), len(preds))
    count_votes(row, preds)

preds = []
for r in range(len(row)):
    pred_label = max(votes[r].items(), key=operator.itemgetter(1))[0]
    preds.append(pred_label)
        
save_preds(preds, "prediction.csv")
orig_votes = copy.deepcopy(votes)

c = 0
for i in range(len(votes)):
    if len(votes[i].keys()) <=2 and get_gini_index(votes[i]) > 0.48:
        label = choices(list(votes[i].keys()), list(votes[i].values()))[0]
        print(i, votes[i])
        print(label)
        c+=1
#        votes[i] = {label:1}
print(c)