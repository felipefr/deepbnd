#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 16:46:37 2020

@author: felipefr
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
from elasticity_utils import Box, Region
 

folder = 'rb_bar_3param_negativePoisson/'


file = open(folder + "nodes.txt") 
nodes = np.loadtxt(file)
file.close()

h = 0.5
L = 5.0
eps  = 0.0001
xa = 2.0

box1 = Box(eps,xa-eps,-h-eps,-h+eps)
box2 = Box(eps,xa-eps,h-eps,h+eps)
box3 = Box(xa-eps,L-eps,-h-eps,-h+eps)
box4 = Box(xa-eps,L-eps,h-eps,h+eps)
box5 = Box(L-eps,L+eps,-h-eps,h+eps)

regionIn = Region([box3,box4])
regionOut = Region([box1,box2, box5])

np.savetxt(folder + "indexesNodesIn.txt",regionIn.getAdmissibleNodes(nodes),fmt = '%i')
np.savetxt(folder + "indexesNodesOut.txt",regionOut.getAdmissibleNodes(nodes),fmt = '%i')