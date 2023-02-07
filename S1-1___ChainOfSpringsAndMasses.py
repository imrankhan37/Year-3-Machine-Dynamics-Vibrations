# -*- coding: utf-8 -*-
"""
Part II Spring Term
Lecture 1: Axial and Torsional Vibration of Beams

Last modified on Wed Sep 14 01:04:45 2022
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
L = 1
W = 0.01
H = 0.01

elno = 10
k = (W*H*E) / (L/elno)
m = (L*W*H*density) / elno

M = m * np.identity(10)
K = np.zeros( (10,10) )
for i in range(10):
    if (i == 0):
        K[i, 0] = 2*k
        K[i, 1] = -k
    elif (i == 9):
        K[i, -1] = k
        K[i, -2] = -k
    else:
        K[i, i-1] = -k
        K[i, i] = 2*k
        K[i, i+1] = -k

val, vec = linalg.eigh(K, M)
w = np.sqrt(val)
f = w / (2*pi)

##########
# Step 2 #
##########
n = 3
l = np.linspace(0,1,11)
plt.figure()
for i in range(n):
    modeshape = np.append([0],vec[:,i]) / (2*pi*f[i])
    if (modeshape[1] < 0):
        modeshape = -modeshape
    plt.plot(l, modeshape)
plt.xlim(0, 1)
plt.show()

##########
# Step 3 #
##########
F = np.zeros(n)
for i in range(n):
    F[i] = ((2*i)+1) * (pi/2) / (2*pi) * np.sqrt(E/density) / L