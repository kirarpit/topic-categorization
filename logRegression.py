#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 21:38:14 2018

@author: Arpit
"""
import numpy as np
from scipy import sparse
from sklearn.preprocessing import normalize, Normalizer
from utils import split_data

class LogisticRegression:
    def __init__(self, raw_training_data, raw_testing_data):
        self.preprocess_data(raw_training_data, raw_testing_data)
        self.getDeltaMatrix()
        self.W = sparse.csr_matrix(np.random.rand(self.k, self.n + 1))
        print("W shape", self.W.shape)

    def train(self, eq=None, lr=0.01, iterations=1):
        for step in range(iterations):
            print()
            print("###################")
            print("Step:", step)

            self.P = self.getProbs()
            error = sparse.csr_matrix(self.D - self.P)
    
            if eq is not None: lr = eq.getValue(step)
            print("Learning Rate:", lr)
            
            self.W += lr * error.dot(self.X)
            self.checkAccuracy()
    
    def getProbs(self, X=None):
        if X is None:
            X = self.X
        
        P = self.W.dot(X.T).toarray()
        P = np.exp(P)
        P[-1, :][True] = 1
        P = P/P.sum(0)
        return P

    def predict(self, X=None):
        probs = self.getProbs(X)
        preds = np.argmax(probs, 0) + 1
        return preds
    
    def checkAccuracy(self):
        preds = self.predict(self.X_valid)
        labels = self.Y_valid.toarray()[:, 0]
        accuracy = sum([1 if labels[i]==preds[i] else 0 for i in range(len(labels))])/len(labels)
        print("Validation Set Accuracy:", accuracy)
        
    def preprocess_data(self, raw_training_data, raw_testing_data):
        X, X_valid = split_data(raw_training_data, 0.8)
        
        self.Y = X[:, -1]
        self.Y_valid = X_valid[:, -1]
        X = X[:, 1:-1]
        X_valid = X_valid[:, 1:-1]
        X_test = raw_testing_data[:, 1:]
        
        self.n = X.shape[1]
        self.k = len(np.unique(self.Y.toarray()))

        #Normalize together
        data = sparse.vstack([X, X_valid, X_test])
        data_norm = normalize(data, norm='l1', axis=0)
        X = data_norm[:X.shape[0], :]
        X_valid = data_norm[X.shape[0]:X.shape[0] + X_valid.shape[0], :]
        X_test = data_norm[X_valid.shape[0]:, :]

#        self.X_normalizer = Normalizer(norm='l1').fit(X.T)
#        X = sparse.csr_matrix(self.X_normalizer.transform(X.T).T)

        self.X = self.addBiasColumn(X)
        self.X_valid = self.addBiasColumn(X_valid)
        self.X_test = self.addBiasColumn(X_test)
        
        print("X shape", self.X.shape)
        print("Y shape", self.Y.shape)
        print("X validation shape", self.X_valid.shape)
        print("X testing shape", self.X_test.shape)
        print("# classes ", self.k)
        
#    def preprocess_testing_data(self, raw_data):
#        X_test = raw_data[:, 1:]
#        X_test = sparse.csr_matrix(self.X_normalizer.transform(X_test.T).T)
#        return self.addBiasColumn(X_test)

    def addBiasColumn(self, data):
        bias_column = sparse.csr_matrix(np.ones((data.shape[0], 1)))
        return sparse.csr_matrix(sparse.hstack([bias_column, data]))

    def getDeltaMatrix(self):
        self.D = np.repeat(self.Y.toarray().T, self.k, 0)
        for i in range(1, self.k + 1):
            r = self.D[i-1, :]
            r[r != i] = 0
            r[r == i] = 1

        print("D shape", self.D.shape)        
        
    def load_weights(self):
        self.W = sparse.load_npz("weights.npz")

    def save_weights(self):
        sparse.save_npz('weights.npz', self.W)