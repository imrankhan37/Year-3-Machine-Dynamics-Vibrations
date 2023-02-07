# -*- coding: utf-8 -*-
"""
Part II Spring Term
Tutorial Questions Lecture 5: Modal analysis of conceptually-simple rotor models

Last modified on Wed Sep 14 00:07:57 2022
@author: Dr Ludovic Renson, Amanda Lee
"""

################################################
# Select question(s) to run and display output #
################################################
# Questions
run_Q1 = 1  #---> Edit "1" (True) or "0" (False)
run_Q2 = 0  #---> Edit "1" (True) or "0" (False)
run_Q3 = 0  #---> Edit "1" (True) or "0" (False)
run_Q4 = 0  #---> Edit "1" (True) or "0" (False)

###############################################################
# Import required functions, modules, packages, and libraries #
###############################################################
# Functions
from numpy import pi as pi, exp as exp # Import NumPy constants and mathematical functions

# Modules, packages, and libraries
import numpy as np                     # Import the fundamental package for scientific computing with Python
import scipy.linalg as linalg          # Import SciPy linear algebra functions package
import matplotlib.pyplot as plt        # Import the state-based interface for interactive plots and simple cases of programmatic plot generation

################################################
# Question 1: Rigid rotor on flexible supports #
################################################
if run_Q1:
    m  = 123
    Id = 2.8625
    Ip = 0.6134
    kT = 2300e3
    kC = 75e3
    kR = 143.75e3
    
    M = np.diag( [m, m, Id, Id] )
    K = np.array( [[kT,   0,   0,  kC],
                   [ 0,  kT, -kC,   0],
                   [ 0, -kC,  kR,   0],
                   [kC,   0,   0,  kR]] )
    G = lambda w: np.array( [[0,     0,     0,     0],
                             [0,     0,     0,     0],
                             [0,     0,     0,  w*Ip],
                             [0,     0, -w*Ip,     0]] )
    
    w = np.linspace(0, 4000*(2*pi/60), 100)
    
    fig = plt.figure()
    fig.suptitle("Rigid rotor on flexible supports", fontweight="bold")
    plt.title("Evolution of the natural frequencies as a function of the rotation speed")
    for wi in w:
        A1 = np.hstack( [              M,           G(wi)] )
        A2 = np.hstack( [np.zeros((4,4)),               M] )
        A = np.append(A1, A2, axis=0)
        B1 = np.hstack( [np.zeros((4,4)),               K] )
        B2 = np.hstack( [             -M, np.zeros((4,4))] )
        B = np.append(B1, B2, axis=0)
        
        lambda1, v = linalg.eig(B, -A)
        wn = np.zeros(lambda1.shape, dtype="complex_")
        wn = np.imag(lambda1)
        wn = wn[wn>=0]
        
        plt.plot([wi]*len(wn), wn, "k+")
    plt.xlim(w[0], w[-1])
    plt.xlabel("Rotation speed [rad/s]")
    plt.ylabel("Natural frequencies [rad/s]")
    
    t = np.linspace(0, 2*pi/wn[0], 100)
    Mode41 = (v[0,0]*exp(lambda1[0]*t)) + (v[0,1]*exp(lambda1[1]*t))
    Mode42 = (v[1,0]*exp(lambda1[0]*t)) + (v[1,1]*exp(lambda1[1]*t))
    fig = plt.figure()
    fig.suptitle("Forward mode", fontweight="bold")
    ax = plt.subplot(1,1,1)
    plt.title(f"Mode 1: $\lambda$ = {int(np.imag(lambda1[0]))} rad/s")
    plt.plot(Mode41, Mode42, "+-")
    plt.xlabel("$\\theta(t)$ [rad]")
    plt.ylabel("$\phi(t)$ [rad]")
    plt.axhline(y=0, color="k", zorder=1); plt.axvline(x=0, color="k", zorder=1)
    ax.set_box_aspect(1)
    
    t = np.linspace(0, 2*pi/wn[1], 100)
    Mode31 = (v[0,2]*exp(lambda1[2]*t)) + (v[0,3]*exp(lambda1[3]*t))
    Mode32 = (v[1,2]*exp(lambda1[2]*t)) + (v[1,3]*exp(lambda1[3]*t))
    fig = plt.figure()
    fig.suptitle("Backward mode", fontweight="bold")
    ax = plt.subplot(1,1,1)
    plt.title(f"Mode 2: $\lambda$ = {int(np.imag(lambda1[2]))} rad/s")
    plt.plot(Mode31, Mode32, "+-")
    plt.xlabel("$\\theta(t)$ [rad]")
    plt.ylabel("$\phi(t)$ [rad]")
    plt.axhline(y=0, color="k", zorder=1); plt.axvline(x=0, color="k", zorder=1)
    ax.set_box_aspect(1)
    
    t = np.linspace(0, 2*pi/wn[2], 100)
    Mode21 = (v[0,4]*exp(lambda1[4]*t)) + (v[0,5]*exp(lambda1[5]*t))
    Mode22 = (v[1,4]*exp(lambda1[4]*t)) + (v[1,5]*exp(lambda1[5]*t))
    fig = plt.figure()
    fig.suptitle("Forward mode", fontweight="bold")
    ax = plt.subplot(1,1,1)
    plt.title(f"Mode 3: $\lambda$ = {int(np.imag(lambda1[4]))} rad/s")
    plt.plot(Mode21, Mode22, "+-")
    plt.xlabel("$\\theta(t)$ [rad]")
    plt.ylabel("$\phi(t)$ [rad]")
    plt.axhline(y=0, color="k", zorder=1); plt.axvline(x=0, color="k", zorder=1)
    ax.set_box_aspect(1)
    
    t = np.linspace(0, 2*pi/wn[3], 100)
    Mode11 = (v[0,6]*exp(lambda1[6]*t)) + (v[0,7]*exp(lambda1[7]*t))
    Mode12 = (v[1,6]*exp(lambda1[6]*t)) + (v[1,7]*exp(lambda1[7]*t))
    fig = plt.figure()
    fig.suptitle("Backward mode", fontweight="bold")
    ax = plt.subplot(1,1,1)
    plt.title(f"Mode 4: $\lambda$ = {int(np.imag(lambda1[6]))} rad/s")
    plt.plot(Mode11, Mode12, "+-")
    plt.xlabel("$\\theta(t)$ [rad]")
    plt.ylabel("$\phi(t)$ [rad]")
    plt.axhline(y=0, color="k", zorder=1); plt.axvline(x=0, color="k", zorder=1)
    ax.set_box_aspect(1)
    plt.show()

###################################################
# Question 2: Rigid rotor on anisotropic supports #
###################################################
if run_Q2:
    m   = 123
    Ip  = 0.6134
    Id  = 2.8625
    kTx = 2300e3
    kTy = 2500e3
    kRx = 143.75e3
    kRy = 156.25e3
    kCx = 75e3
    kCy = 75e3
    
    M = np.diag( [m, m, Id, Id] )
    K = np.array( [[kTx,    0,    0, kCx],
                   [  0,  kTy, -kCy,   0],
                   [  0, -kCy,  kRy,   0],
                   [kCx,   0,     0, kRx]] )
    G = lambda omega: np.array( [[0, 0,         0,        0],
                                 [0, 0,         0,        0],
                                 [0, 0,         0, omega*Ip],
                                 [0, 0, -omega*Ip,        0]] )
    
    RotationSpeed = np.linspace(0, 8000*((2*pi)/60), 3)
    
    fig = plt.figure()
    fig.suptitle("Rigid rotor on anisotropic supports", fontweight="bold")
    plt.title("Evolution of the natural frequencies as a function of the rotation speed")
    for i in range( len(RotationSpeed) ):
        A1 = np.hstack( [M, G(RotationSpeed[i])] )
        A2 = np.hstack( [np.zeros((4,4)), M] )
        A = np.append(A1, A2, axis=0)
        B1 = np.hstack( [np.zeros((4,4)), K] )
        B2 = np.hstack( [-M, np.zeros((4,4))] )
        B = np.append(B1, B2, axis=0)
        
        lambda1, y = linalg.eig(B, -A)
        w = np.imag( lambda1 )
        w = w[w>0]
        
        plt.plot([RotationSpeed[i]]*len(w), w, "k+")
    plt.xlim(RotationSpeed[0], RotationSpeed[-1])
    plt.xlabel("Rotation speed [rad/s]")
    plt.ylabel("Natural frequencies [rad/s]")
    
    t = np.linspace(0, 2*pi/w[0], 100)
    Mode41 = (y[0,0]*exp(lambda1[0]*t)) + (y[0,1]*exp(lambda1[1]*t))
    Mode42 = (y[1,0]*exp(lambda1[0]*t)) + (y[1,1]*exp(lambda1[1]*t))
    fig = plt.figure()
    fig.suptitle("Forward mode", fontweight="bold")
    ax = plt.subplot(1,1,1)
    plt.title(f"Mode 1: $\lambda$ = {int(np.imag(lambda1[0]))} rad/s")
    plt.plot(Mode41, Mode42, "+-")
    plt.xlabel("$\\theta(t)$ [rad]")
    plt.ylabel("$\phi(t)$ [rad]")
    plt.axhline(y=0, color="k", zorder=1); plt.axvline(x=0, color="k", zorder=1)
    ax.set_box_aspect(1)
    
    t = np.linspace(0, 2*pi/w[1], 100)
    Mode31 = (y[0,2]*exp(lambda1[2]*t)) + (y[0,3]*exp(lambda1[3]*t))
    Mode32 = (y[1,2]*exp(lambda1[2]*t)) + (y[1,3]*exp(lambda1[3]*t))
    fig = plt.figure()
    fig.suptitle("Backward mode", fontweight="bold")
    ax = plt.subplot(1,1,1)
    plt.title(f"Mode 1: $\lambda$ = {int(np.imag(lambda1[2]))} rad/s")
    plt.plot(Mode31, Mode32, "+-")
    plt.xlabel("$\\theta(t)$ [rad]")
    plt.ylabel("$\phi(t)$ [rad]")
    plt.axhline(y=0, color="k", zorder=1); plt.axvline(x=0, color="k", zorder=1)
    ax.set_box_aspect(1)
    
    t = np.linspace(0, 2*pi/w[2], 100)
    Mode21 = (y[0,4]*exp(lambda1[4]*t)) + (y[0,5]*exp(lambda1[5]*t))
    Mode22 = (y[1,4]*exp(lambda1[4]*t)) + (y[1,5]*exp(lambda1[5]*t))
    fig = plt.figure()
    fig.suptitle("Forward mode", fontweight="bold")
    ax = plt.subplot(1,1,1)
    plt.title(f"Mode 1: $\lambda$ = {int(np.imag(lambda1[4]))} rad/s")
    plt.plot(Mode21, Mode22, "+-")
    plt.xlabel("$\\theta(t)$ [rad]")
    plt.ylabel("$\phi(t)$ [rad]")
    plt.axhline(y=0, color="k", zorder=1); plt.axvline(x=0, color="k", zorder=1)
    ax.set_box_aspect(1)
    
    t = np.linspace(0, 2*pi/w[3], 100)
    Mode11 = (y[0,6]*exp(lambda1[6]*t)) + (y[0,7]*exp(lambda1[7]*t))
    Mode12 = (y[1,6]*exp(lambda1[6]*t)) + (y[1,7]*exp(lambda1[7]*t))
    fig = plt.figure()
    fig.suptitle("Backward mode", fontweight="bold")
    ax = plt.subplot(1,1,1)
    plt.title(f"Mode 1: $\lambda$ = {int(np.imag(lambda1[6]))} rad/s")
    plt.plot(Mode11, Mode12, "+-")
    plt.xlabel("$\\theta(t)$ [rad]")
    plt.ylabel("$\phi(t)$ [rad]")
    plt.axhline(y=0, color="k", zorder=1); plt.axvline(x=0, color="k", zorder=1)
    ax.set_box_aspect(1)
    plt.show()

############################
# Question 3: Pinned rotor #
############################
if run_Q3:
    Ip = 0.6
    Id = 10
    k  = 1e6
    L  = 0.5
    
    lambda1 = lambda omega:\
              1j*np.array( [[ (Ip*omega)/(2*Id) + np.sqrt((Ip*omega)/(2*Id) + (k*(L**2))/Id)],
                            [ (Ip*omega)/(2*Id) - np.sqrt((Ip*omega)/(2*Id) + (k*(L**2))/Id)],
                            [-(Ip*omega)/(2*Id) + np.sqrt((Ip*omega)/(2*Id) + (k*(L**2))/Id)],
                            [-(Ip*omega)/(2*Id) - np.sqrt((Ip*omega)/(2*Id) + (k*(L**2))/Id)]] )
    
    a = lambda1(0) / (2*pi)
    b = lambda1(3000*((2*pi)/60)) / (2*pi)
    c = lambda1(10000*((2*pi)/60)) / (2*pi)

###########################################################
# Question 4: Mass elastically coupled to a rotating ring #
###########################################################
if run_Q4:
    m = 1
    k1 = 50
    k2 = 98
    c1 = 0
    
    M = np.diag( [m, m] )
    G = lambda omega: np.array( [[        0, -2*m*omega],
                                 [2*m*omega,          0]] )
    K = lambda omega: np.array( [[k1-((omega**2)*m),                 0],
                                 [                0, k2-((omega**2)*m)]] )
    C = np.array( [[c1, 0],
                   [ 0, 0]] )
    
    RotationSpeed = np.linspace(0, 25, 100)
    lambda1 = np.zeros((4, len(RotationSpeed)), dtype="complex_")
    
    fig = plt.figure() # Create a new figure
    fig.suptitle("Mass elastically coupled to a rotating ring", fontweight="bold")
    for i in range( len(RotationSpeed) ):
        Gi = G(RotationSpeed[i])
        Ki = K(RotationSpeed[i])
        
        A1 = np.hstack( [M, np.zeros((2,2))] )
        A2 = np.hstack( [np.zeros((2,2)), Ki] )
        A = np.append(A1, A2, axis=0)
        B1 = np.hstack( [Gi+C, Ki] )
        B2 = np.hstack( [-Ki, np.zeros((2,2))] )
        B = np.append(B1, B2, axis=0)
        
        lambda1[:,i], y = linalg.eig(B, -A)
        w = np.imag( lambda1 )
        w = w[w>0]
    plt.subplot(2,1,1)
    plt.title("Evolution of the natural frequencies as a function of the rotation speed")
    plt.plot(np.tile(RotationSpeed,(4,1)).T, np.real(lambda1).T)
    plt.xlim(RotationSpeed[0], RotationSpeed[-1])
    plt.xlabel("Rotation speed [rad/s]")
    plt.ylabel("$\mathbb{R}$ $(\lambda)$  [arb. unit]")
    plt.legend(["Mode 1", "Mode 2", "Mode 3", "Mode 4"], loc="upper right")
    plt.subplot(2,1,2)
    plt.plot(np.tile(RotationSpeed,(4,1)).T, np.imag(lambda1).T)
    plt.xlim(RotationSpeed[0], RotationSpeed[-1])
    plt.xlabel("Rotation speed [rad/s]")
    plt.ylabel("$\mathbb{I}$ $(\lambda)$\nOR\nNatural frequencies [rad/s]")
    plt.legend(["Mode 1", "Mode 2", "Mode 3", "Mode 4"], loc="upper right")
    plt.tight_layout()
    plt.show()