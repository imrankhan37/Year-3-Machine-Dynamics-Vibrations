# -*- coding: utf-8 -*-
"""
Part II Spring Term
Tutorial Questions Lecture 6: Rotor dynamics - response to excitation

Last modified on Wed Sep 14 00:17:42 2022
@author: Dr Ludovic Renson, Amanda Lee
"""

################################################
# Select question(s) to run and display output #
################################################
# Questions
run_Q1 = 1  #---> Edit "1" (True) or "0" (False)
run_Q2 = 0  #---> Edit "1" (True) or "0" (False)

###############################################################
# Import required functions, modules, packages, and libraries #
###############################################################
# Functions
from numpy import pi as pi, exp as exp # Import NumPy constants and mathematical functions

# Modules, packages, and libraries
import numpy as np                     # Import the fundamental package for scientific computing with Python
import numpy.linalg as LA              # Import NumPy linear algebra functions package
import matplotlib.pyplot as plt        # Import the state-based interface for interactive plots and simple cases of programmatic plot generation

#################################
# Question 1: Coupled equations #
#################################
if run_Q1:
    epsilon = 1e-3
    m  = 122.68
    Ip = 0.6134
    Id = 2.8625
    kT = 2300e3
    kR = 143.75e3
    kC = 75e3
    cT = 23
    cR = 1.4375
    cC = 0.75
    
    Mcomplex = np.array( [[m, 0], [0, Id]] )
    Kcomplex = np.array( [[kT, kC], [kC, kR]] )
    Gcomplex = lambda omega: np.array( [[cT, cC], [cC, cR-(1j*omega*Ip)]] )
    
    RotationSpeed = np.linspace(0, 3500*((2*pi)/60), 10000)
    x = np.zeros( (2, len(RotationSpeed)), dtype="complex_" )
    r0 = np.zeros( len(RotationSpeed), dtype="complex_" )
    phi0 = np.zeros( len(RotationSpeed), dtype="complex_" )
    
    for i in range( len(RotationSpeed) ):
        omegai = RotationSpeed[i]
        H = -(omegai**2)*Mcomplex + 1j*omegai*Gcomplex(omegai) + Kcomplex
        F = np.array( [m*epsilon*(omegai**2), 0] )
        x[:,i] = LA.lstsq(H, F, rcond=None)[0]
        D = (-(Id-Ip)*(omegai**2) + 1j*cR*omegai + kR)*(-m*(omegai**2) + 1j*cT*omegai + kT)\
            - (1j*cC*omegai + kC)**2
        r0[i] = (m*epsilon*(omegai**2)) * (-(Id-Ip)*(omegai**2) + 1j*cR*omegai + kR) / D
        phi0[i] = (m*epsilon*(omegai**2)) * (-kC - 1j*omegai*cC) / D
    
    fig = plt.figure()
    fig.suptitle("Coupled equations - Bode diagram", fontweight="bold")
    plt.subplot(2,2,1)
    plt.title("Mode 1")
    plt.semilogy(RotationSpeed*(60/(2*pi)), np.abs(x[0,:]), label="First-order formulation")
    plt.semilogy(RotationSpeed*(60/(2*pi)), np.abs(r0), "--", label="Analytical solution")
    plt.xlim(RotationSpeed[0]*(60/(2*pi)), RotationSpeed[-1]*(60/(2*pi)))
    plt.xlabel("Rotation speed [rpm]")
    plt.ylabel("Response amplitude\n[arb. unit]")
    plt.grid(which="major")
    plt.grid(which="minor", color="#BBBBBB", linestyle="--", linewidth=0.5); plt.minorticks_on()
    plt.legend(loc="upper right")
    
    plt.subplot(2,2,3)
    plt.plot(RotationSpeed*(60/(2*pi)), -np.angle(x[0,:]), label="First-order formulation")
    plt.plot(RotationSpeed*(60/(2*pi)), -np.angle(r0), "--", label="Analytical solution")
    plt.xlim(RotationSpeed[0]*(60/(2*pi)), RotationSpeed[-1]*(60/(2*pi)))
    plt.ylim(0, pi)
    plt.xlabel("Rotation speed [rpm]")
    plt.ylabel("Response phase\n[rad]")
    plt.grid(which="major")
    plt.grid(which="minor", color="#BBBBBB", linestyle="--", linewidth=0.5); plt.minorticks_on()
    plt.legend(loc="upper right")
    
    plt.subplot(2,2,2)
    plt.title("Mode 2")
    plt.semilogy(RotationSpeed*(60/(2*pi)), np.abs(x[1,:]), label="First-order formulation")
    plt.semilogy(RotationSpeed*(60/(2*pi)), np.abs(phi0), "--", label="Analytical solution")
    plt.xlim(RotationSpeed[0]*(60/(2*pi)), RotationSpeed[-1]*(60/(2*pi)))
    plt.xlabel("Rotation speed [rpm]")
    plt.ylabel("Response amplitude\n[arb. unit]")
    plt.grid(which="major")
    plt.grid(which="minor", color="#BBBBBB", linestyle="--", linewidth=0.5); plt.minorticks_on()
    plt.legend(loc="upper right")
    
    plt.subplot(2,2,4)
    plt.plot(RotationSpeed*(60/(2*pi)), -np.angle(x[1,:]), label="First-order formulation")
    plt.plot(RotationSpeed*(60/(2*pi)), -np.angle(phi0), "--", label="Analytical solution")
    plt.xlim(RotationSpeed[0]*(60/(2*pi)), RotationSpeed[-1]*(60/(2*pi)))
    plt.ylim(-pi, pi)
    plt.xlabel("Rotation speed [rpm]")
    plt.ylabel("Response phase\n[rad]")
    plt.axhline(y=0, color="k", zorder=1); plt.axvline(x=0, color="k", zorder=1)
    plt.grid(which="major")
    plt.grid(which="minor", color="#BBBBBB", linestyle="--", linewidth=0.5); plt.minorticks_on()
    plt.legend(loc="upper right")
    plt.tight_layout()
    fig.canvas.manager.window.showMaximized()
    plt.show()

###################################
# Question 2: Anisotropic support #
###################################
if run_Q2:
    m   = 122.68
    Ip  = 0.6134
    Id  = 2.8625
    kx1 = 1e6
    kx2 = 1.3e6
    ky1 = 1.5e6
    ky2 = 1.8e6
    cx1 = 20
    cx2 = 26
    cy1 = 30
    cy2 = 36
    a   = 0.25
    b   = 0.25
    
    epsilon = 1e-3
    beta    = 1e-2
    delta   = 0
    gamma   = pi/5
    
    M = np.diag( [m, m, Id, Id] )
    K = np.array( [[        kx1+kx2,               0,                         0,           (b*kx2)-(a*kx1)],
                   [              0,         ky1+ky2,           (a*ky1)-(b*ky2),                         0],
                   [              0, (a*ky1)-(b*ky2), ((a**2)*ky1)+((b**2)*ky2),                         0],
                   [(b*kx2)-(a*kx1),               0,                         0, ((a**2)*kx1)+((b**2)*kx2)]] )
    C = np.array( [[        cx1+cx2,               0,                         0,           (b*cx2)-(a*cx1)],
                   [              0,         cy1+cy2,           (a*cy1)-(b*cy2),                         0],
                   [              0, (a*cy1)-(b*cy2), ((a**2)*cy1)+((b**2)*cy2),                         0],
                   [(b*cx2)-(a*cx1),               0,                         0, ((a**2)*cx1)+((b**2)*cx2)]] )
    G = lambda omega: np.array( [[0, 0,         0,        0],
                                 [0, 0,         0,        0],
                                 [0, 0,         0, omega*Ip],
                                 [0, 0, -omega*Ip,        0]] )
    
    RotationSpeed = np.linspace(0, 3500*((2*pi)/60), 10000)
    x = np.zeros( (4, len(RotationSpeed)), dtype="complex_" )
    
    for i in range( len(RotationSpeed) ):
        omegai = RotationSpeed[i]
        H = -(omegai**2)*M + 1j*omegai*(C+G(omegai)) + K
        F = (omegai**2) * np.array( [           m  * epsilon * exp(1j*delta),\
                                     -1j *      m  * epsilon * exp(1j*delta),\
                                      1j * (Id-Ip) *    beta * exp(1j*gamma),\
                                           (Id-Ip) *    beta * exp(1j*gamma)] )
        x[:,i] = LA.lstsq(H, F, rcond=None)[0]
    
    fig = plt.figure()
    fig.suptitle("Anisotropic supports - Bode diagram", fontweight="bold")
    plt.subplot(2,4,1)
    plt.title("Mode 1")
    plt.semilogy(RotationSpeed*(60/(2*pi)), np.abs(x[0,:]))
    plt.xlim(RotationSpeed[0]*(60/(2*pi)), RotationSpeed[-1]*(60/(2*pi)))
    plt.xlabel("Rotation speed [rpm]")
    plt.ylabel("Response amplitude\n[arb. unit]")
    plt.subplot(2,4,2)
    plt.title("Mode 2")
    plt.semilogy(RotationSpeed*(60/(2*pi)), np.abs(x[1,:]))
    plt.xlim(RotationSpeed[0]*(60/(2*pi)), RotationSpeed[-1]*(60/(2*pi)))
    plt.xlabel("Rotation speed [rpm]")
    plt.ylabel("Response amplitude\n[arb. unit]")
    plt.subplot(2,4,3)
    plt.title("Mode 3")
    plt.semilogy(RotationSpeed*(60/(2*pi)), np.abs(x[2,:]))
    plt.xlim(RotationSpeed[0]*(60/(2*pi)), RotationSpeed[-1]*(60/(2*pi)))
    plt.xlabel("Rotation speed [rpm]")
    plt.ylabel("Response amplitude\n[arb. unit]")
    plt.subplot(2,4,4)
    plt.title("Mode 4")
    plt.semilogy(RotationSpeed*(60/(2*pi)), np.abs(x[3,:]))
    plt.xlim(RotationSpeed[0]*(60/(2*pi)), RotationSpeed[-1]*(60/(2*pi)))
    plt.xlabel("Rotation speed [rpm]")
    plt.ylabel("Response amplitude\n[arb. unit]")
    
    plt.subplot(2,4,5)
    plt.plot(RotationSpeed*(60/(2*pi)), np.angle(x[0,:]))
    plt.xlim(RotationSpeed[0]*(60/(2*pi)), RotationSpeed[-1]*(60/(2*pi)))
    plt.ylim(-pi, pi)
    plt.xlabel("Rotation speed [rpm]")
    plt.ylabel("Response phase\n[rad]")
    plt.axhline(y=0, color="k", zorder=1); plt.axvline(x=0, color="k", zorder=1)
    plt.subplot(2,4,6)
    plt.plot(RotationSpeed*(60/(2*pi)), np.angle(x[1,:]))
    plt.xlim(RotationSpeed[0]*(60/(2*pi)), RotationSpeed[-1]*(60/(2*pi)))
    plt.ylim(-pi, pi)
    plt.xlabel("Rotation speed [rpm]")
    plt.ylabel("Response phase\n[rad]")
    plt.axhline(y=0, color="k", zorder=1); plt.axvline(x=0, color="k", zorder=1)
    plt.subplot(2,4,7)
    plt.plot(RotationSpeed*(60/(2*pi)), np.angle(x[2,:]))
    plt.xlim(RotationSpeed[0]*(60/(2*pi)), RotationSpeed[-1]*(60/(2*pi)))
    plt.ylim(-pi, pi)
    plt.xlabel("Rotation speed [rpm]")
    plt.ylabel("Response phase\n[rad]")
    plt.axhline(y=0, color="k", zorder=1); plt.axvline(x=0, color="k", zorder=1)
    plt.subplot(2,4,8)
    plt.plot(RotationSpeed*(60/(2*pi)), np.angle(x[3,:]))
    plt.xlim(RotationSpeed[0]*(60/(2*pi)), RotationSpeed[-1]*(60/(2*pi)))
    plt.ylim(-pi, pi)
    plt.xlabel("Rotation speed [rpm]")
    plt.ylabel("Response phase\n[rad]")
    plt.axhline(y=0, color="k", zorder=1); plt.axvline(x=0, color="k", zorder=1)
    plt.tight_layout()
    fig.canvas.manager.window.showMaximized()
    plt.show()