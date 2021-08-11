# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 10:32:29 2021

@author: 문영주
"""

import numpy as np

test = ['FFFFFFF','FFFFFFF','FFFFFFF','FFFTFFF','FFFFFFF','FTFFFFF','FFFFFFF']

a = []
for i in range(len(test)):
    for j in range(len(test[i])):
        if test[i][j] == 'T':
            print('check here')
            print('i: {}, j: {}'.format(i, j))
            
            a.append((i,j))