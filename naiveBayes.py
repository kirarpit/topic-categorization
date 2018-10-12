#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  1 18:04:51 2018

@author: Arpit
"""
import numpy as np
from scipy import sparse
from collections import Counter

class NaiveBayes:
    def __init__(self, data):
        self.preprocess_data(data)
        self.vocab_len = self.xi_cnt_in_yk.shape[1]
        self.set_beta(1/self.vocab_len)
    
    """
    Computes a matrix for P(Xi/Yk)
    and another matrix for P(Yk)
    """
    def train(self):
        print("Training")
        total_words_in_yk = self.xi_cnt_in_yk.sum(1, keepdims=True) #shape (20, 1)
        denominator = total_words_in_yk + (self.alpha - 1) * self.vocab_len #shape (20, 1)
        
        self.prob_xi_given_yk = (self.xi_cnt_in_yk + (self.alpha - 1)) / denominator #shape (20, vocab_len)
        self.prob_yk = self.label_dist / sum(self.label_dist)

    def predict(self, rows):
        print("Predicting")
        prob_yk_given_document = np.dot(np.log2(self.prob_xi_given_yk), rows.T) + np.log2(self.prob_yk)
        preds = np.argmax(prob_yk_given_document, 0) + 1
        return preds
    
    """
    Calculates the number of times a word has
    appeared in a category and creates a matrix
    of k X V where k is # unique categories and
    V is #vocabulary
    """
    def preprocess_data(self, data):
        print("Pre-processing data")
        labels = data[:, -1].toarray()
        data = data[:, 1:-1]
        
        #separates the data class wise
        data_list = []
        for label in np.unique(labels):
            rows = np.where(labels == label)[0]
            data_list.append(data[rows].sum(0).tolist()[0])
        
        label_dist = Counter(list(labels.T[0]))
        label_dist = np.array(sorted(label_dist.items()))[:, 1]
        label_dist = label_dist.reshape(len(label_dist), 1)
        
        self.xi_cnt_in_yk = sparse.csr_matrix(data_list).toarray()
        self.label_dist = label_dist
        
    def set_beta(self, beta):
        self.beta = beta
        self.alpha = 1 + self.beta