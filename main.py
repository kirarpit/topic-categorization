#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 30 20:57:32 2018

@author: Arpit
"""

from utils import load_data, plot, save_preds
from naiveBayes import NaiveBayes
from logRegression import LogisticRegression
from graphPlot import GraphPlot
import math

raw_data = load_data('training.csv')
"""
# Trains Naive Bayes on whole training data
# and makes the prediction on testing data for
# a chosen value of Beta.
"""
NB = NaiveBayes(raw_data, var_red=False, split=0.85)

lgx = -2
beta = math.pow(10, lgx)
NB.train(beta)

preds = list(NB.predict(NB.preprocess_testing_data(load_data('testing.csv'))))

save_preds(preds, "prediction.csv")

"""
# Trains Naive Bayes on 85% of the training data.
# Rest of the data is held for validation.
# Trains for different values of Beta and plots
# accuracy on validation set vs log(beta).
"""
#Training on training_data set
NB = NaiveBayes(raw_data, split=0.85, var_red=True)

Xs = [] # X coordinates for the graph plot
Ys = [] # Y coordinates for the graph plot
lgx = -5
while lgx <= 0:
    beta = math.pow(10, lgx)
    NB.train(beta)
    accuracy = NB.get_accuracy() #Checking accuracy on validation set
    
    Xs.append(lgx)
    Ys.append(accuracy)
    lgx += .05
    
plot(Xs, Ys)

"""
# Trains Multinomial logistic regression
# on 85% of training set for different
# hyper-parameters and then plots them.
"""
raw_training_data = load_data('training.csv')
raw_testing_data = load_data('testing.csv')

# Training on different learning_rates and plotting graph
gp = GraphPlot("learning_rate", xlabel="Iterations", ylabel="% Accuracy")
learning_rate = 0.001
while learning_rate <= 0.01:
    gp.addLabel(learning_rate)
    LR = LogisticRegression(raw_training_data, split=0.85, svd=False)
    LR.train(lr=learning_rate, lamda=0.001, iterations=500)
    gp.addPlot(LR.getAccData())
    learning_rate += 0.002
gp.plot()

# Training on different lambda values and plotting graph
gp = GraphPlot("lambda", xlabel="Iterations", ylabel="% Accuracy")
lamda = 0.001
while lamda <= 0.01:
    gp.addLabel(lamda)
    LR = LogisticRegression(raw_training_data, split=0.85, svd=False)
    LR.train(lr=0.003, lamda=lamda, iterations=2000)
    gp.addPlot(LR.getAccData())
    lamda += 0.002
gp.plot()

preds = list(LR.predict(LR.preprocess_testing_data(raw_testing_data)))
save_preds(preds, "prediction_LR.csv")