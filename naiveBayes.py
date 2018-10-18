#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  1 18:04:51 2018

@author: Arpit
"""
import numpy as np
from scipy import sparse
from collections import Counter
from sklearn.feature_selection import VarianceThreshold
from utils import split_data

class NaiveBayes:
    def __init__(self, data, split=1.0, var_red=True):
        self.split = split #split percentage on which training and validation sets would be separated
        self.var_red = var_red #variance reduction boolean
        self.sel = VarianceThreshold(threshold=1.8e-7)

        self.preprocess_data(data)
        self.vocab_len = self.xi_cnt_in_yk.shape[1]
        self.set_beta(1/self.vocab_len)
    
    """
    Computes a matrix for P(Xi/Yk)
    and another matrix for P(Yk)
    """
    def train(self, beta=None):
        print("Training")
        if beta is not None: self.set_beta(beta)
        
        total_words_in_yk = self.xi_cnt_in_yk.sum(1, keepdims=True) #shape (20, 1)
        denominator = total_words_in_yk + (self.alpha - 1) * self.vocab_len #shape (20, 1)
        
        self.prob_xi_given_yk = (self.xi_cnt_in_yk + (self.alpha - 1)) / denominator #shape (20, vocab_len)
        self.prob_yk = self.label_dist / sum(self.label_dist)
        
        if self.split < 1:
            self.check_accuracy()

    def predict(self, rows):
        print("Predicting")
        prob_yk_given_document = np.dot(np.log2(self.prob_xi_given_yk), rows.T) + np.log2(self.prob_yk)
        preds = np.argmax(prob_yk_given_document, 0) + 1
        return preds
    
    def check_accuracy(self):
        preds = self.predict(self.X_valid)
        labels = self.Y_valid[:, 0]
        
        accuracy = sum([1 if labels[i]==preds[i] else 0 for i in range(len(labels))])/len(labels)
        self.acc = accuracy
        print("Validation Set Accuracy:", accuracy)
    
    def getConfMatrix(self):
        preds = self.predict(self.X_valid)
        labels = self.Y_valid[:, 0]
        
        matrix = np.zeros((len(self.prob_yk), len(self.prob_yk)))
        for i in range(len(labels)):
            if labels[i] != preds[i]:
                matrix[labels[i]-1][preds[i]-1] += 1
        
        return matrix
    
    def get_accuracy(self):
        return self.acc
    
    """
    Calculates the number of times a word has
    appeared in a category and creates a matrix
    of k X V where k is # unique categories and
    V is #vocabulary
    """
    def preprocess_data(self, data):
        print("Pre-processing data")
        
        #Splits the data into training and validation sets
        X, X_valid = split_data(data, self.split)
        
        Y = X[:, -1].toarray()
        X = X[:, 1:-1]
        self.Y_valid = X_valid[:, -1].toarray()
        X_valid = X_valid[:, 1:-1]
        
        if self.var_red:
            print("Fitting and transforming data of shape:", X.shape)
            X = self.sel.fit_transform(X)
            if X_valid.shape[0] > 0: X_valid = self.sel.transform(X_valid)
            print("Reduced shape:", X.shape)
            
        self.X_valid = X_valid.toarray()
        
        #separates the data class wise
        data_list = []
        for label in np.unique(Y):
            rows = np.where(Y == label)[0]
            data_list.append(X[rows].sum(0).tolist()[0])
        
        label_dist = Counter(list(Y.T[0]))
        label_dist = np.array(sorted(label_dist.items()))[:, 1]
        label_dist = label_dist.reshape(len(label_dist), 1)
        
        self.xi_cnt_in_yk = sparse.csr_matrix(data_list).toarray()
        self.label_dist = label_dist
    
    def preprocess_testing_data(self, data):
        data = data[:, 1:].toarray() #removes 1st column
        
        if self.var_red:
            print("Transforming data of shape:", data.shape)
            data = self.sel.transform(data)
            print("Reduced shape:", data.shape)
            
        return data
        
    def set_beta(self, beta):
        self.beta = beta
        self.alpha = 1 + self.beta