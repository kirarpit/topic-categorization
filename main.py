#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 30 20:57:32 2018

@author: Arpit
"""

from utils import load_data, preprocess_data, split_data
from naiveBayes import NaiveBayes

raw_data = load_data('training.csv')
raw_training_data, raw_validation_data = split_data(raw_data, 0.8)

#Training on training_data set
training_data, label_dist = preprocess_data(raw_training_data)
vocab_len = training_data.shape[1]
nb = NaiveBayes(vocab_len)
nb.train(training_data, label_dist)

#Predicting on validation set to check accuracy
rows = raw_validation_data[:, 1:-1].toarray()
labels = list(raw_validation_data[:, -1].toarray()[:, 0])
preds = list(nb.predict(rows))

accuracy = sum([1 if labels[i]==preds[i] else 0 for i in range(len(labels))])/len(labels)
print(accuracy)