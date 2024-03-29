#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 21:38:14 2018

@author: Arpit
"""
import numpy as np
from scipy import sparse
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_selection import VarianceThreshold
from utils import split_data
import time

"""
Multinomial Logistic Regression
"""
class LogisticRegression:
    """
    Takes in the raw_training_data, with additional options
    for data transformation like standardisation,
    normalization or PCA with SVD
    """
    def __init__(self, raw_training_data, var_red=True, std=True, norm=True, svd=True, split=1.0):
        self.var_red = var_red
        self.split = split
        self.std = std
        self.norm = norm
        self.svd = svd
        self.truncSVD = TruncatedSVD(n_components=1000)
        self.stdScaler = StandardScaler(with_mean=False)
        self.sel = VarianceThreshold()

        self.preprocess_data(raw_training_data)
        self.getDeltaMatrix()
        self.W = sparse.csr_matrix(np.random.rand(self.k, self.n))
        
        self.accuracy_data = []
        print("W shape", self.W.shape)

    """
    Trains to get optimal values of the weights
    """
    def train(self, eq=None, lr=0.01, iterations=1, lamda=0.001):
        for step in range(iterations):
            tic = time.time()
            print()
            print("-------------------")
            print("Step:", step)

            self.P = self.getProbs()
            error = sparse.csr_matrix(self.D - self.P)
    
            if eq is not None: lr = eq.getValue(step)
            print("Learning Rate:", lr)
            
            self.W += lr * (error.dot(self.X) - lamda*self.W)
            if self.split < 1 and step % 10 == 0:
                acc = self.checkAccuracy()
                self.accuracy_data.append((step, acc))
            
            print("Time taken:", time.time() - tic)
    
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
        return accuracy
    
    def getAccData(self):
        return self.accuracy_data
    
    """
    Pre processes the data;
    Transforms the data by scaling and adding bias columns
    """
    def preprocess_data(self, raw_training_data):
        X, X_valid = split_data(raw_training_data, self.split)
        
        self.Y = X[:, -1]
        self.Y_valid = X_valid[:, -1]
        X = X[:, 1:-1]
        X_valid = X_valid[:, 1:-1]
        
        self.k = len(np.unique(self.Y.toarray())) #Number of unique classes

        self.X = self.transformData(X, fit=True)
        self.X_valid = self.transformData(X_valid)
        self.n = self.X.shape[1] #Number of features/columns/words

        print("X shape", self.X.shape)
        print("Y shape", self.Y.shape)
        print("X_valid shape", self.X_valid.shape)
        print("# classes ", self.k)
        
    def preprocess_testing_data(self, raw_data):
        X_test = raw_data[:, 1:]
        X_test = self.transformData(X_test)
        return X_test

    def transformData(self, data, fit=False):
        #return if empty
        if data.shape[0] == 0:
            return data
        
        # Feature Selection(Removing features with 0 variance)
        if self.var_red:
            if fit: self.sel.fit(data)
            data = self.sel.transform(data)
        
        #SVD
        if self.svd:
            if fit: self.truncSVD.fit(data)
            data = sparse.csr_matrix(self.truncSVD.transform(data))
        
        #Standardize
        if self.std:
            if fit: self.stdScaler.fit(data)
            data = sparse.csr_matrix(self.stdScaler.transform(data))
        
        #Normalize
        if self.norm:
            if fit: self.X_normalizer = Normalizer(norm='l1').fit(data)
            data = sparse.csr_matrix(self.X_normalizer.transform(data))
        
        #Add bias column
        bias_column = sparse.csr_matrix(np.ones((data.shape[0], 1)))
        data = sparse.csr_matrix(sparse.hstack([bias_column, data]))

        return data
    
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