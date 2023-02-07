# -*- coding: utf-8 -*-
"""
Part I Autumn Term
Lecture 4: Single-degree-of-freedom Systems

This script demonstrates the behaviour of single-degree-of-freedom systems
under free response (Demo 1), forced response (Demo 2), excitation due to
unbalance (Demo 3), and vibration isolation (Demo 4).

Last modified on Mon Sep 12 22:04:54 2022
@author: Dr Ludovic Renson, Amanda Lee
"""

############################################
# Select demo(s) to run and display output #
############################################
# Demos
run_D1 = 1 #---> Edit "1" (True) or "0" (False)
run_D2 = 0 #---> Edit "1" (True) or "0" (False)
run_D3 = 0 #---> Edit "1" (True) or "0" (False)
run_D4 = 0 #---> Edit "1" (True) or "0" (False)

###############################################################
# Import required functions, modules, packages, and libraries #
###############################################################
# Functions
from numpy import pi as pi            # Import NumPy constants and mathematical functions
from scipy.integrate import solve_ivp # Import SciPy function to solve an initial value problem for a system of ODEs

# Modules, packages, and libraries
import numpy as np                    # Import the fundamental package for scientific computing with Python
import matplotlib.pyplot as plt       # Import the state-based interface to matplotlib for interactive plots and simple cases of programmatic plot generation

#########################
# Demo 1: Free response #
#########################
if run_D1:
    # System parameters
    m     = 1                                                       #---> Edit mass in kg
    k     = 1                                                       #---> Edit stiffness in N/m
    c1    = 0.1                                                     #---> Edit under-damped damping ratio
    c2    = 2                                                       #---> Edit critically-damped damping ratio
    c3    = 8                                                       #---> Edit over-damped damping ratio
    t_lim = [0, round(5*(2*pi), 3)]                                 # Define bounds for time vector in s
    t     = np.linspace(t_lim[0], t_lim[1], int(t_lim[1] * 1000)+1) # Create time vector in s
    x0    = [3, 3]                                                  # Initial displacements in m
    
    # Solve initial value problem
    def fode(t, x):
        return [x[1], -(k*x[0] + c*x[1])/m]
    def ode45(c_eval):
        global c
        c = c_eval
        sol = solve_ivp(fode, t_lim, x0, t_eval=t)
        return sol.y
    x1 = ode45(c1); x2 = ode45(c2); x3 = ode45(c3)
    
    # Plot time-domain signal
    fig1 = plt.figure(); fig1.suptitle("Free response", fontweight="bold")
    ax1 = fig1.add_subplot(1, 1, 1); ax1.set_title("Time domain signal")
    ax1.plot(t, x1[0,:], label="c = 0.1 (Under-damped)")
    ax1.plot(t, x2[0,:], label="c = 2 (Critically-damped)")
    ax1.plot(t, x3[0,:], label="c = 8 (Over-damped)")
    ax1.set_xlim(t_lim)
    ax1.set_xlabel("Time [s]"); ax1.set_ylabel("Amplitude [arb. unit]")
    ax1.axhline(y=0, color="k", zorder=1); ax1.axvline(x=0, color="k", zorder=1)
    ax1.legend(loc="upper right")
    
    # Plot phase plane diagram
    fig2 = plt.figure(); fig2.suptitle("Free response", fontweight="bold")
    ax2 = fig2.add_subplot(1, 1, 1); ax2.set_title("Phase plane diagram")
    ax2.plot(x1[0,:], x1[1,:], label="c = 0.1 (Under-damped)") # Plot y vs t
    ax2.plot(x2[0,:], x2[1,:], label="c = 2 (Critically-damped)")
    ax2.plot(x3[0,:], x3[1,:], label="c = 8 (Over-damped)")
    ax2.set_xlabel("x [arb. unit]"); ax2.set_ylabel("y [arb. unit]")
    ax2.axhline(y=0, color="k", zorder=1); ax2.axvline(x=0, color="k", zorder=1)
    ax2.legend(loc="upper right")
    ax2.axis("equal")
    plt.show()

###########################
# Demo 2: Forced response #
###########################
if run_D2:
    # System parameters
    m       = 1                       #---> Edit mass in kg
    k       = 1                       #---> Edit stiffness in N/m
    epsilon = [0, 0.05, 0.15, 0.25]   #---> Edit out-of-balance in mm
    w       = np.linspace(0, 4, 1000) # Create frequency vector in Hz
    w0      = np.sqrt(k/m)            # Find resonance frequency in Hz
    ww0     = w/w0                    # Create normalised frequency vector
    
    # Plot frequency response function
    fig3 = plt.figure(); fig3.suptitle("Forced response", fontweight="bold")
    ax3 = fig3.add_subplot(2, 2, 1); ax3.set_title("Bode diagram (magnitude)")
    ax4 = fig3.add_subplot(2, 2, 3); ax4.set_title("Bode diagram (phase)")
    ax5 = fig3.add_subplot(2, 2, 2); ax5.set_title("Freqeuncy response function (real)")
    ax6 = fig3.add_subplot(2, 2, 4); ax6.set_title("Freqeuncy response function (imaginary)")
    
    for ei in epsilon:
        H = (1/k) * (1/(1-(ww0**2)+(2j*ei*ww0)))
        mag = np.abs(H)
        phase = np.angle(H)
        real = np.real(H)
        imag = np.imag(H)
        
        ax3.semilogy(ww0, mag, label="$\epsilon$ = "+str(ei))
        ax4.plot(ww0, phase, label="$\epsilon$ = "+str(ei))
        ax5.plot(ww0, real, label="$\epsilon$ = "+str(ei))
        ax6.plot(ww0, imag, label="$\epsilon$ = "+str(ei))
    
    ax3.set_xlim(ww0[0], ww0[-1])
    ax3.set_xlabel("$\dfrac{\omega}{\omega_{0}}$ [no unit]"); ax3.set_ylabel("$|H(\omega)|$  [arb. unit]")
    ax3.grid(which="major")
    ax3.legend(loc="upper right")
    
    ax4.set_xlim(ww0[0], ww0[-1])
    ax4.set_xlabel("$\dfrac{\omega}{\omega_{0}}$ [no unit]"); ax4.set_ylabel("arg $(H(\omega))$  [rad]")
    ax4.grid(which="major")
    ax4.legend(loc="upper right")
    
    ax5.set_xlim(ww0[0], ww0[-1])
    ax5.set_xlabel("$\dfrac{\omega}{\omega_{0}}$ [no unit]"); ax5.set_ylabel("$\mathbb{R}$ $(H(\omega))$  [arb. unit]")
    ax5.grid(which="major")
    ax5.legend(loc="upper right")
    
    ax6.set_xlim(ww0[0], ww0[-1])
    ax6.set_xlabel("$\dfrac{\omega}{\omega_{0}}$ [no unit]"); ax6.set_ylabel("$\mathbb{I}$ $(H(\omega))$  [arb. unit]")
    ax6.grid(which="major")
    ax6.legend(loc="upper right")
    fig3.tight_layout()
    
    # System parameters (redefined)
    m       = 1                              #---> Edit mass in kg
    k       = 1                              #---> Edit stiffness in N/m
    epsilon = [0.15, 0.25, 0.5, 1]           # Edit out-of-balance in mm
    w       = np.linspace(0, 10000, 1000000) # Create frequency vector in Hz
    w0      = np.sqrt(k/m)                   # Find resonance frequency in Hz
    ww0     = w/w0                           # Create normalised frequency vector
    
    # Plot Nyquist diagram
    fig4 = plt.figure(); fig4.suptitle("Forced response", fontweight="bold")
    ax7 = fig4.add_subplot(1,1,1); ax7.set_title("Nyquist plot")
    
    for ei in epsilon:
        H = 1 / (1-(ww0**2)+(2j*ei*ww0))
        real = np.real(H)
        imag = np.imag(H)
        
        ax7.plot(real, imag, label="$\epsilon$ = "+str(ei))
    
    ax7.set_ylim(top=0)
    ax7.set_xlabel("$\mathbb{R}$ $(H(\omega))$  [arb. unit]"); ax7.set_ylabel("$\mathbb{I}$ $(H(\omega))$  [arb. unit]")
    ax7.axhline(y=0, color="k", zorder=1); ax7.axvline(x=0, color="k", zorder=1)
    ax7.grid(which="major")
    ax7.legend(loc="upper right")
    plt.show()

#######################################
# Demo 3: Excitation due to unbalance #
#######################################
if run_D3:
    # System parameters
    M       = 1                         #---> Edit mass in kg
    k       = 1                         #---> Edit stiffness in N/m
    epsilon = [0.1, 0.2, 0.375, 0.5, 1] #---> Edit out-of-balance in mm
    w       = np.linspace(0, 5, 1000)   # Create frequency vector in Hz
    wn      = np.sqrt(k/M)              # Find natural frequency in Hz
    wwn     = w/wn                      # Create normalised frequency vector
    
    # Plot frequency response function
    fig5 = plt.figure(); fig5.suptitle("Excitation due to unbalance", fontweight="bold")
    ax8 = fig5.add_subplot(1,1,1); ax8.set_title("Frequency response function")
    
    for ei in epsilon:
        MXme = wwn**2 / np.sqrt( (1-wwn**2)**2+(2*ei*wwn)**2 )
        
        ax8.plot(wwn, MXme, label="$\epsilon$ = "+str(ei))
    
    ax8.set_xlim(wwn[0], wwn[-1]); ax8.set_ylim(bottom=0)
    ax8.set_xlabel("$\dfrac{\omega}{\omega_{n}}$ [no unit]"); ax8.set_ylabel("$\dfrac{MX}{me}$ [arb. unit]")
    ax8.grid(which="major")
    ax8.grid(which="minor", color="#BBBBBB", linestyle="--", linewidth=0.5); ax8.minorticks_on()
    ax8.legend(loc="upper right")
    plt.show()

###############################
# Demo 4: Vibration isolation #
###############################
if run_D4:
    # System parameters
    m       = 0.05                        #---> Edit mass in kg
    k       = 1                           #---> Edit stiffness in N/m
    epsilon = [0.05, 0.375, 0.5, 1]       #---> Edit out-of-balance in mm
    w       = np.linspace(0, 50, 1000000) # Create frequency vector in Hz
    w0      = np.sqrt(k/m)                # Find resonance frequency in Hz
    ww0     = w/w0                        # Create normalised frequency vector
    
    # Plot transmissibility diagram
    fig6 = plt.figure(); fig6.suptitle("Vibration isolation", fontweight="bold")
    ax9 = fig6.add_subplot(1,1,1); ax9.set_title("Transmissibility plot")
    
    for ei in epsilon:
        T = np.sqrt( (1+(2*ei*ww0)**2) / ((1-ww0**2)**2+(2*ei*ww0)**2) )
        
        ax9.semilogy(ww0, T, label="$\epsilon$ = "+str(ei))
    
    ax9.axhline(y=1, color="k", linestyle="--"); ax9.axvline(x=np.sqrt(2), color="k", linestyle="--")
    ax9.set_xlim(ww0[0], ww0[-1])
    ax9.set_xlabel("$\dfrac{\omega}{\omega_{0}}$ [no unit]"); ax9.set_ylabel("$T(\omega)$  [arb. unit]")
    ax9.grid(which="major")
    ax9.grid(which="minor", color="#BBBBBB", linestyle="--", linewidth=0.5); ax9.minorticks_on()
    ax9.legend(loc="upper right")
    plt.show()