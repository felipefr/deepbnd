#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 11:21:58 2022

@author: felipe
"""

import sys, os
import numpy as np



i = -1

try:
    print(i)
    if i<0:
        raise Exception("Only positive numbers allowed")
except:
    print(i, " The number is negative")