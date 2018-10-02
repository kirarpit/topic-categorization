#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 30 20:57:32 2018

@author: Arpit
"""

from utils import load_data, preprocess_data, split_data, plot, save_preds
from naiveBayes import NaiveBayes
import math

raw_data = load_data('training.csv')

####################################################
# Trains Naive Bayes on whole training data
# and makes the prediction on testing data for
# a chosen value of Beta.
####################################################

raw_training_data = raw_data
training_data, label_dist = preprocess_data(raw_training_data)

vocab_len = training_data.shape[1]
nb = NaiveBayes(vocab_len)

lgx = -1.3
beta = math.pow(10, lgx)
nb.set_beta(beta)
nb.train(training_data, label_dist)

raw_testing_data = load_data('testing.csv')
rows = raw_testing_data[:, 1:].toarray()
preds = list(nb.predict(rows))
save_preds(preds, "prediction.csv")

####################################################
# Trains Naive Bayes on 85% of the training data.
# Rest of the data is held for validation.
# Trains for different values of Beta and plots
# accuracy vs log(beta).
####################################################

raw_training_data, raw_validation_data = split_data(raw_data, 0.85)

#Training on training_data set
training_data, label_dist = preprocess_data(raw_training_data)
vocab_len = training_data.shape[1]
nb = NaiveBayes(vocab_len)

rows = raw_validation_data[:, 1:-1].toarray()
labels = list(raw_validation_data[:, -1].toarray()[:, 0])

Xs = [] # X coordinates for the graph plot
Ys = [] # Y coordinates for the graph plot
lgx = -5
while lgx <= 0:
    beta = math.pow(10, lgx)
    nb.set_beta(beta)
    nb.train(training_data, label_dist)
    
    #Predicting on validation set to check accuracy
    preds = list(nb.predict(rows))
    
    accuracy = sum([1 if labels[i]==preds[i] else 0 for i in range(len(labels))])/len(labels)
    Xs.append(lgx)
    Ys.append(accuracy)
    print(lgx, accuracy)
    
    lgx += .05
    
plot(Xs, Ys)