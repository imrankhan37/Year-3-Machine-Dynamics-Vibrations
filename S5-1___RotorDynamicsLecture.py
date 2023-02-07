# -*- coding: utf-8 -*-
"""
Part II Spring Term
Lecture 5: Modal analysis of conceptually-simple rotor models

Last modified on Tue Sep 13 23:55:18 2022
@author: Dr Ludovic Renson, Amanda Lee
"""

###############################################################
# Import required functions, modules, packages, and libraries #
###############################################################
# Functions
from numpy import pi as pi, exp as exp # Import NumPy constants and mathematical functions

# Modules, packages, and libraries
import numpy as np                     # Import the fundamental package for scientific computing with Python
import scipy.linalg as linalg          # Import SciPy linear algebra functions package
import matplotlib.pyplot as plt        # Import the state-based interface for interactive plots and simple cases of programmatic plot generation

##########
# Step 1 #
##########
m  = 122.68
Ip = 0.6134
Id = 2.8625
kT = 2300e3
kR = 143.75e3
kC = 75e3

M = np.array( [[m, 0], [0, Id]] )
K = np.array( [[kT, kC], [kC, kR]] )
K2 = np.array( [[kT, -kC], [-kC, kR]] )

w, X = linalg.eigh(K, M)
w2, X2 = linalg.eigh(K2, M)

##########
# Step 2 #
##########
m  = 122.68
Ip = 0.6134
Id = 2.8625
kT = 2300e3
kR = 143.75e3
kC = 0

M = np.array( [[Id, 0], [0, Id]] )
K = np.array( [[kR, 0], [0, kR]] )
G = lambda omega: np.array( [[0, omega*Ip], [-omega*Ip, 0]] )

##########
# Step 3 #
##########
RotationSpeed = np.linspace(0, 4000*((2*pi)/60), 100)
w = np.zeros( (len(RotationSpeed),2) )

fig = plt.figure()
fig.suptitle("Simple rotor model", fontweight="bold")
for i in range( len(RotationSpeed) ):
    A1 = np.hstack( [M, G(RotationSpeed[i])] )
    A2 = np.hstack( [np.zeros((2,2)), M] )
    A = np.append(A1, A2, axis=0)
    
    B1 = np.hstack( [np.zeros((2,2)), K] )
    B2 = np.hstack( [-M, np.zeros((2,2))] )
    B = np.append(B1, B2, axis=0)
    
    lambda1, y = linalg.eig(B, -A)
    wtemp = np.imag( lambda1 )
    w[i,:] = wtemp[wtemp>=0]

plt.title("Evolution of the natural frequencies as a function of the rotation speed")
plt.plot(np.array([RotationSpeed]*2).T, w, "k+-")
plt.xlim(RotationSpeed[0], RotationSpeed[-1])
plt.xlabel("Rotation speed [rad/s]")
plt.ylabel("Natural frequencies [rad/s]")

##########
# Step 4 #
##########
t = np.linspace(0, 2*pi/w[-1,0], 100)
Mode11 = (y[0,0]*exp(lambda1[0]*t)) + (y[0,1]*exp(lambda1[1]*t))
Mode12 = (y[1,0]*exp(lambda1[0]*t)) + (y[1,1]*exp(lambda1[1]*t))

fig = plt.figure()
fig.suptitle("Forward mode", fontweight="bold")
ax = plt.subplot(1,1,1)
plt.title(f"Mode 1: $\lambda$ = {int(np.imag(lambda1[0]))} rad/s")
plt.plot(Mode11, Mode12, "+-")
plt.axhline(y=0, color="k", zorder=1); plt.axvline(x=0, color="k", zorder=1)
plt.xlabel("$\\theta(t)$ [rad]")
plt.ylabel("$\phi(t)$ [rad]")
ax.set_box_aspect(1)

t = np.linspace(0, 2*pi/w[-1,1], 100)
Mode21 = (y[0,2]*exp(lambda1[2]*t)) + (y[0,3]*exp(lambda1[3]*t))
Mode22 = (y[1,2]*exp(lambda1[2]*t)) + (y[1,3]*exp(lambda1[3]*t))

fig = plt.figure()
fig.suptitle("Backward mode", fontweight="bold")
ax = plt.subplot(1,1,1)
plt.title(f"Mode 2: $\lambda$ = {int(np.imag(lambda1[2]))} rad/s")
plt.plot(Mode21, Mode22, "+-")
plt.axhline(y=0, color="k", zorder=1); plt.axvline(x=0, color="k", zorder=1)
plt.xlabel("$\\theta(t)$ [rad]")
plt.ylabel("$\phi(t)$ [rad]")
ax.set_box_aspect(1)
plt.show()