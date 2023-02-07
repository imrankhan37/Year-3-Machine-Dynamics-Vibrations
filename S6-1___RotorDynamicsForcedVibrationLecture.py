# -*- coding: utf-8 -*-
"""
Part II Spring Term
Lecture 6: Rotor dynamics - response to excitation

Last modified on Wed Sep 14 00:32:39 2022
@author: Dr Ludovic Renson, Amanda Lee
"""

###############################################################
# Import required functions, modules, packages, and libraries #
###############################################################
# Functions
from numpy import pi as pi         # Import NumPy constants and mathematical functions

# Modules, packages, and libraries
import numpy as np                 # Import the fundamental package for scientific computing with Python
import scipy.linalg as linalg      # Import SciPy linear algebra functions package
import matplotlib.pyplot as plt    # Import a state-based interface for interactive plots and simple cases of programmatic plot generation

##########
# Step 1 #
##########
epsilon = 0.1
omega = np.linspace(0, 10, 1001)
zeta = np.linspace(0.01, 0.5, 4)
wn = 1

plt.figure()
for zetai in zeta:
    r0 = epsilon*(omega**2) / ((wn**2-omega**2)+(2j*omega*zetai*wn))
    plt.loglog(omega/wn, np.abs(r0))
plt.xlim(0.1, 10)

plt.figure()
for zetai in zeta:
    r0 = epsilon*(omega**2) / ((wn**2-omega**2)+(2j*omega*zetai*wn))
    plt.semilogx(omega/wn, -np.angle(r0, deg=True))
plt.xlim(0.1, 10)
plt.ylim(0, 180)

##########
# Step 2 #
##########
epsilon = 1e-4
m       = 122.68
Ip      = 0.6134
Id      = 2.8625
kT      = 2300e3
kR      = 143.75e3
kC      = 0
cT      = 23
cR      = 1.4375
cC      = 0

M = np.diag( [m, m, Id, Id] )
K = np.array( [[kT, 0, 0, kC],
                [0, kT, -kC, 0],
                [0, -kC, kR, 0],
                [kC, 0, 0, kR]] )
C = np.array( [[cT, 0, 0, cC],
                [0, cT, -cC, 0],
                [0, -cC, cR, 0],
                [cC, 0, 0, cR]] )
G = lambda omega: np.array( [[0, 0,         0,        0],
                              [0, 0,         0,        0],
                              [0, 0,         0, omega*Ip],
                              [0, 0, -omega*Ip,        0]] )

RotationSpeed = np.linspace(0, 3500*((2*pi)/60), 10000)
L = np.zeros( (4, len(RotationSpeed)), dtype="complex_" )
for i in range( len(RotationSpeed) ):
    A1 = np.hstack( [M, G(RotationSpeed[i])+C] )
    A2 = np.hstack( [np.zeros((4,4)), M] )
    A = np.append(A1, A2, axis=0)
    
    B1 = np.hstack( [np.zeros((4,4)), K] )
    B2 = np.hstack( [-M, np.zeros((4,4))] )
    B = np.append(B1, B2, axis=0)
    
    lambda1, y = linalg.eig(B, -A)
    L[:,i] = lambda1[np.imag(lambda1)>0]
    e = np.roots( [Id, cR-(1j*Ip*RotationSpeed[i]), kR] )

plt.figure()
plt.plot(np.tile(RotationSpeed*(60/(2*pi)), (4,1)), np.real(L), "k.")
plt.xlim(0, 3500)

plt.figure()
plt.plot(np.tile(RotationSpeed*(60/(2*pi)), (4,1)), np.imag(L)/(2*pi), "k.")
plt.plot(RotationSpeed*(60/(2*pi)), RotationSpeed/(2*pi))
plt.xlim(0, 3500)
plt.ylim(bottom=0)

##########
# Step 3 #
##########
Mcomplex = np.diag( [m, Id] )
Kcomplex = np.array( [[kT, kC],
                      [kC, kR]] )
Ccomplex = np.array( [[cT, cC],
                      [cC, cR]] )
Gcomplex = lambda omega: np.array( [[           0, 0],
                                    [-1j*omega*Ip, 0]] )

x = np.zeros( (2, len(RotationSpeed)), dtype="complex_" )

for i in range( len(RotationSpeed) ):
    omegai = RotationSpeed[i]
    
    H = -(omegai**2)*Mcomplex + 1j*omegai*(Ccomplex+Gcomplex(omegai)) + Kcomplex
    F = np.array( [m*epsilon*(omegai**2), 0] )
    x[:,i] = linalg.lstsq(H, F)[0]

plt.figure()
plt.semilogy(RotationSpeed*(60/(2*pi)), np.abs(x[0,:]))
plt.xlim(0, 3500)

plt.figure()
plt.plot(RotationSpeed*(60/(2*pi)), -np.angle(x[0,:]))
plt.xlim(0, 3500)
plt.ylim(0, pi)

##########
# Step 4 #
##########
epsilon = 1e-3
m = 122.68
Ip = 0.6134
Id = 2.8625
kT = 2300e3
kR = 143.75e3
kC = 75e3
cT = 23
cR = 1.4375
cC = 0.75

M = np.diag( [m, m, Id, Id] )
K = np.array( [[kT, 0, 0, kC],
                [0, kT, -kC, 0],
                [0, -kC, kR, 0],
                [kC, 0, 0, kR]] )
C = np.array( [[cT, 0, 0, cC],
                [0, cT, -cC, 0],
                [0, -cC, cR, 0],
                [cC, 0, 0, cR]] )
G = lambda omega: np.array( [[0, 0,         0,        0],
                              [0, 0,         0,        0],
                              [0, 0,         0, omega*Ip],
                              [0, 0, -omega*Ip,        0]] )

RotationSpeed = np.linspace(0, 3500*((2*pi)/60), 10000)
L = np.zeros( (4, len(RotationSpeed)), dtype="complex_" )
for i in range( len(RotationSpeed) ):
    A1 = np.hstack( [M, G(RotationSpeed[i])+C] )
    A2 = np.hstack( [np.zeros((4,4)), M] )
    A = np.append(A1, A2, axis=0)
    
    B1 = np.hstack( [np.zeros((4,4)), K] )
    B2 = np.hstack( [-M, np.zeros((4,4))] )
    B = np.append(B1, B2, axis=0)
    
    lambda1, y = linalg.eig(B, -A)
    L[:,i] = lambda1[np.imag(lambda1)>0]
    e = np.roots( [Id, cR-(1j*Ip*RotationSpeed[i]), kR] )

plt.figure()
plt.plot(np.tile(RotationSpeed*(60/(2*pi)), (4,1)), np.real(L), "k.")
plt.xlim(0, 3500)

plt.figure()
plt.plot(np.tile(RotationSpeed*(60/(2*pi)), (4,1)), np.imag(L)/(2*pi), "k.")
plt.plot(RotationSpeed*(60/(2*pi)), RotationSpeed/(2*pi))
plt.xlim(0, 3500)
plt.ylim(bottom=0)

##########
# Step 5 #
##########
Mcomplex = np.diag( [m, Id] )
Kcomplex = np.array( [[kT, kC],
                      [kC, kR]] )
GCcomplex = lambda omega: np.array( [[cT,               cC],
                                     [cC, cR-(1j*omega*Ip)]] )

x = np.zeros( (2, len(RotationSpeed)), dtype="complex_" )
r0 = np.zeros(len(RotationSpeed), dtype="complex_" )

for i in range( len(RotationSpeed) ):
    omegai = RotationSpeed[i]
    
    H = -(omegai**2)*Mcomplex + 1j*omegai*(GCcomplex(omegai)) + Kcomplex
    F = np.array( [m*epsilon*(omegai**2), 0] )
    x[:,i] = linalg.lstsq(H, F)[0]
    
    D = (-(Id-Ip)*(omegai**2) + (1j*cR*omegai) + kR)*(-m*(omegai**2) + (1j*cT*omegai) + kT) - (1j*cC*omegai + kC)**2
    r0[i] = (m*epsilon*(omegai**2)) * (-(Id-Ip)*(omegai**2) + (1j*cR*omegai) + kR) / D

plt.figure()
plt.semilogy(RotationSpeed*(60/(2*pi)), np.abs(x[0,:]))
plt.semilogy(RotationSpeed*(60/(2*pi)), np.abs(r0), "--")
plt.xlim(0, 3500)

plt.figure()
plt.plot(RotationSpeed*(60/(2*pi)), -np.angle(x[0,:]))
plt.plot(RotationSpeed*(60/(2*pi)), -np.angle(r0), "--")
plt.xlim(0, 3500)
plt.ylim(0, pi)

plt.figure()
plt.plot(RotationSpeed*(60/(2*pi)), -np.angle(x[0,:]))
plt.xlim(0, 3500)
plt.ylim(0, pi)

plt.figure()
plt.plot(np.tile(RotationSpeed*(60/(2*pi)), (4,1)), np.imag(L)/(2*pi), "k.")
plt.plot(RotationSpeed*(60/(2*pi)), RotationSpeed/(2*pi))
plt.plot(RotationSpeed*(60/(2*pi)), 2*RotationSpeed/(2*pi))
plt.plot(RotationSpeed*(60/(2*pi)), 3*RotationSpeed/(2*pi))
plt.xlim(0, 3500)
plt.ylim(0, 50)
plt.show()