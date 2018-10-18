#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 00:06:21 2018

@author: Arpit
"""

from utils import load_data
from naiveBayes import NaiveBayes
import math
import matplotlib.pyplot as plt
import numpy as np
from heatmap import heatmap, annotate_heatmap

raw_data = load_data('training.csv')
"""
# Trains Naive Bayes on 85% of the training set
# and computes confusion matrix on the rest 15%
# 20 times and takes an average.
"""

conf_matrix = None
for i in range(20):
    
    NB = NaiveBayes(raw_data, var_red=False, split=0.85)
    
    lgx = -2
    beta = math.pow(10, lgx)
    NB.train(beta)
    
    matrix = NB.getConfMatrix()
    if conf_matrix is None:
        conf_matrix = matrix
    else:
        conf_matrix += matrix

#conf_matrix /= 20
        
labels = ["alt.atheism","comp.graphics","comp.os.ms-windows.misc",
          "comp.sys.ibm.pc.hardware","comp.sys.mac.hardware","comp.windows.x",
          "misc.forsale","rec.autos","rec.motorcycles","rec.sport.baseball",
          "rec.sport.hockey","sci.crypt","sci.electronics","sci.med","sci.space",
          "soc.religion.christian","talk.politics.guns","talk.politics.mideast",
          "talk.politics.misc","talk.religion.misc"]

fig, ax = plt.subplots()
ax.set_title("Confusion Matrix")
fig.set_size_inches(20, 20)
im, cbar = heatmap(conf_matrix, labels, labels, ax=ax, cmap="YlGn", 
                   cbarlabel="# of mistakes")
texts = annotate_heatmap(im, valfmt="{x}")

#fig.tight_layout()
plt.savefig("/Users/Arpit/Desktop/conf_matrix.png")
plt.show()