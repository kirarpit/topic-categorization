#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 11:41:04 2018

@author: Arpit
"""

class MathEq:
    def __init__(self, eqNo):
        self.eqNo = eqNo
        
    def getValue(self, x):
        value = 0
        if self.eqNo == 1:
            value = 317.6831 + (0.005508051 - 317.6831)/(1 + (x/356108.4) ** 2.261086)
        elif self.eqNo == 2:
            value = 183.0892 + (0.004616234 - 183.0892)/(1 + (x/1245353) ** 1.680354)
        elif self.eqNo == 3:
            value = 199.8125 + (0.004844613 - 199.8125)/(1 + (x/1112497) ** 1.733167)
            
        return value