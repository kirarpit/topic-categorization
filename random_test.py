#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 15:35:00 2018

@author: Arpit
"""

import csv, random
from utils import load_data, save_preds

lines = []
with open("diff.csv") as file:
    reader = csv.reader(file)
    for row in reader:
        lines.append(row)

preds = {}
i = 0
while i < len(lines) - 1:
    if random.random() < 0.5:
        j = i
    else:
        j = i+1
    print(i, j)
    
    preds[int(lines[i][0])] = lines[j][1]
    i += 2

preds[12769] = 6
preds[12874] = 3

d = {}
with open("best_preds.csv") as file:
    reader = csv.reader(file)
    flag = True
    for row in reader:
        if not flag:
            d[int(row[0])] = row[1]
        flag = False

for key, value in preds.items():
    d[key] = value

preds= []
for key, value in d.items():
    preds.append(int(value))

save_preds(preds, 'random_preds.csv')