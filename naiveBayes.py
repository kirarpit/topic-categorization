#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  1 18:04:51 2018

@author: Arpit
"""
import numpy as np

class NaiveBayes:
    def __init__(self, vocab_len):
        self.vocab_len = vocab_len
        self.set_beta(1/self.vocab_len)
    
    """
    Computes a matrix for P(Xi/Yk)
    and another matrix for P(Yk)
    """
    def train(self, xi_cnt_in_yk, label_dist):
        total_words_in_yk = xi_cnt_in_yk.sum(1, keepdims=True) #shape (20, 1)
        denominator = total_words_in_yk + (self.alpha - 1) * self.vocab_len #shape (20, 1)
        
        self.prob_xi_given_yk = (xi_cnt_in_yk + (self.alpha - 1)) / denominator #shape (20, vocab_len)
        self.prob_yk = label_dist / sum(label_dist)

    def predict(self, rows):
        prob_yk_given_document = np.dot(np.log2(self.prob_xi_given_yk), rows.T) + np.log2(self.prob_yk)
        preds = np.argmax(prob_yk_given_document, 0) + 1
        return preds
    
    def set_beta(self, beta):
        self.beta = beta
        self.alpha = 1 + self.beta