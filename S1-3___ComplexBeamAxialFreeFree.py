# -*- coding: utf-8 -*-
"""
Part II Spring Term
Lecture 1: Axial and Torsional Vibration of Beams

Last modified on Wed Sep 14 01:15:10 2022
@author: Dr Frederic Cegla, Amanda Lee
"""

###############################################################
# Import required functions, modules, packages, and libraries #
###############################################################
# Functions
from numpy import pi as pi         # Import NumPy constants and mathematical functions

# Modules, packages, and libraries
import numpy as np                 # Import the fundamental package for scientific computing with Python
import scipy.linalg as linalg      # Import SciPy linear algebra functions package
import matplotlib.pyplot as plt    # Import the state-based interface for interactive plots and simple cases of programmatic plot generation

##########
# Step 1 #
##########
density = 7800
E = 200e9
L = 0.15
dL = 0.015
R = np.array( [0.05, 0.05, 0.05, 0.05, 0.04, 0.033, 0.028, 0.023, 0.02, 0.02] )
A = pi * (R**2)
kA = (A[:-1] + A[1:]) / 2
k = kA * E / dL
m = dL * A * density
elno = L / dL

M = m * np.identity(10)
K = np.zeros( (10,10) )
for i in range(10):
    print(i)
    if (i == 0):
        K[i, 0] = k[i]
        K[i, 1] = -k[i]
    elif (i == 9):
        K[i, -1] = k[i-1]
        K[i, -2] = -k[i-1]
    else:
        K[i, i-1] = -k[i-1]
        K[i, i] = k[i] + k[i-1]
        K[i, i+1] = -k[i]

val, vec = linalg.eigh(K, M)
w = np.sqrt(val)
f = w / (2*pi)

##########
# Step 2 #
##########
n = 3
l = np.linspace(0,1,10)
plt.figure()
for i in range(n):
    plt.plot(l, vec[:,i])
plt.xlim(0, 1)
plt.show()