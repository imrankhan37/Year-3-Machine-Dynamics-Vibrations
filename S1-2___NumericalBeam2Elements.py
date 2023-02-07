# -*- coding: utf-8 -*-
"""
Part II Spring Term
Lecture 1: Axial and Torsional Vibration of Beams

Last modified on Wed Sep 14 01:10:33 2022
@author: Dr Frederic Cegla, Amanda Lee
"""

###############################################################
# Import required functions, modules, packages, and libraries #
###############################################################
# Functions
from numpy import pi as pi, sin as sin, cos as cos, sinh as sinh, cosh as cosh # Import NumPy constants and mathematical functions

# Modules, packages, and libraries
import numpy as np                 # Import the fundamental package for scientific computing with Python
import scipy.linalg as linalg      # Import SciPy linear algebra functions package
import matplotlib.pyplot as plt    # Import the state-based interface for interactive plots and simple cases of programmatic plot generation

##########
# Step 1 #
##########
density = 7800
E = 200e9
B = 0.01
D = 0.01
LB = 1
L = LB/2
A = B*D
I = B * (D**3) / 12

M = np.array( [[312,   0,         54,    -13*L    ],\
               [0,     8*(L**2),  13*L,  -3*(L**2)],\
               [54,    13*L,      156,   -22*L    ],\
               [-13*L, -3*(L**2), -22*L, 4*(L**2) ]] )
M = M*density*A*L/420
K = np.array( [[24,  0,        -12,  6*L     ],\
               [0,   8*(L**2), -6*L, 2*(L**2)],\
               [-12, -6*L,     12,   -6*L    ],\
               [6*L, 2*(L**2), -6*L, 4*(L**2)]] )
K = K*E*I/(L**3)

val, vec = linalg.eigh(K, M)
w = np.sqrt(val)
f = w / (2*pi)

##########
# Step 2 #
##########
mode1 = vec[:,0]
mode2 = vec[:,1]
num_mode1 = -np.array( [0, mode1[0], (mode1[1]*L)+mode1[2]] )
num_mode2 = -np.array( [0, mode2[0], (mode2[1]*L)+mode2[2]] )

xvec = np.linspace(0, 1, 21)
lambda1 = np.sqrt(3.52 / (LB**2))
C1 = 1
C2 = -(cosh(lambda1*LB)+cos(lambda1*LB)) / (sinh(lambda1*LB)+sin(lambda1*LB))
C3 = -C1
C4 = -C2
U1 = C1*cosh(lambda1*xvec) + C2*sinh(lambda1*xvec) + C3*cos(lambda1*xvec) + C4*sin(lambda1*xvec)
lambda2 = np.sqrt(22.4 / (LB**2))
C1 = 1
C2 = -(cosh(lambda2*LB)+cos(lambda2*LB)) / (sinh(lambda2*LB)+sin(lambda2*LB))
C3 = -C1
C4 = -C2
U2 = C1*cosh(lambda2*xvec) + C2*sinh(lambda2*xvec) + C3*cos(lambda2*xvec) + C4*sin(lambda2*xvec)
ana_mode1 = U1 / max(U1) * max(num_mode1)
ana_mode2 = U2 / max(U2) * max(num_mode2)

exact1 = 3.52 * np.sqrt(E*I/A/density) / (LB**2) / (2*pi)
exact2 = 22.4 * np.sqrt(E*I/A/density) / (LB**2) / (2*pi)

plt.figure()
plt.plot([0, L, LB], num_mode1, "b+--")
plt.plot(xvec, ana_mode1, "b-")
plt.plot([0, L, LB], num_mode2, "r+--")
plt.plot(xvec, ana_mode2, "r-")
plt.xlim(0, 1)
plt.show()