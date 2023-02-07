# -*- coding: utf-8 -*-
"""
Part I Autumn Term
Lecture 5: Vibration of multi-degree-of-freedom systems

This script compares the frequency response function of a two-degree-of-freedom
system, where excitation and response measurement are co-located / not co-located
(Demo 1), and subsequently demonstrates the application of a dynamic vibration
absorber in a two-degree-of-freedom system (Demo 2).

Last modified on Tue Sep 13 20:43:59 2022
@author: Dr Ludovic Renson, Amanda Lee
"""

############################################
# Select demo(s) to run and display output #
############################################
# Demos
run_D1 = 1 #---> Edit "1" (True) or "0" (False)
run_D2 = 0 #---> Edit "1" (True) or "0" (False)

###############################################################
# Import required functions, modules, packages, and libraries #
###############################################################
from numpy import pi as pi         # Import NumPy constants and mathematical functions

# Modules, packages, and libraries
import numpy as np                 # Import the fundamental package for scientific computing with Python
import numpy.linalg as LA          # Import NumPy linear algebra functions package
import scipy.linalg as linalg      # Import SciPy linear algebra functions package
import matplotlib.pyplot as plt    # Import the state-based interface for interactive plots and simple cases of programmatic plot generation

#######################################
# Demo 1: Frequency response function #
#######################################
if run_D1:
    # System definition
    M     = np.identity(2)
    K     = np.matrix([[2, -1], [-1, 2]])
    w2, X = linalg.eigh(K, M)
    
    # Orthogonality properties
    ortho1 = LA.multi_dot([X.T, M, X])
    ortho2 = LA.multi_dot([X.T, K, X])
    
    # Proportional damping
    a    = 0.01
    b    = 0
    C    = (a*K) + (b*M)
    zeta = 0.5 * np.diag(LA.multi_dot([X.T, C, X])) / np.sqrt(w2)
    
    # Frequency response function (physical)
    w        = np.linspace(0, 2*np.sqrt(w2[1]), 10000)
    Hw       = np.zeros((2,2,len(w)), dtype="complex_")
    Hw_modal = np.zeros((2,2,len(w)), dtype="complex_")
    for i in range(len(w)):
        for j in range(2):
            F = np.zeros(2); F[j] = 1
            d = K + (1j*w[i]*C) - ((w[i]**2)*M)
            Hw[:, j, i]       = LA.lstsq(d, F, rcond=None)[0]
            Hw_modal[:, j, i] = LA.lstsq( LA.multi_dot([X.T, d, X]), LA.multi_dot([X.T, F]), rcond=None )[0]
    x_f1 = np.multiply( (np.matrix(X[0,:]).T @ np.matrix(np.ones(len(w)))), np.squeeze(Hw_modal[:,0,:]) ).T
    x_f2 = np.multiply( (np.matrix(X[1,:]).T @ np.matrix(np.ones(len(w)))), np.squeeze(Hw_modal[:,0,:]) ).T
    
    # Plot Bode diagram and Nyquist plot
    fig1 = plt.figure(); fig1.suptitle("Bode diagram and Nyquist plot\n of a 2DOF system", fontweight="bold")
    ax1 = fig1.add_subplot(3,2,1); ax1.set_title("Excitation and response measurement\nare co-located")
    ax1.semilogy(w, np.squeeze(np.abs(Hw[0,0,:])), "k")
    ax1.set_xlim(w[0], w[-1])
    ax1.set_xlabel("Frequency [Hz]"); ax1.set_ylabel("Amplitude\n[arb. unit]")
    ax1.grid(which="major")
    
    ax2 = fig1.add_subplot(3,2,2); ax2.set_title("Excitation and response measurement\nare not co-located")
    ax2.semilogy(w, np.squeeze(np.abs(Hw[1,0,:])), "k")
    ax2.set_xlim(w[0], w[-1])
    ax2.set_xlabel("Frequency [Hz]"); ax2.set_ylabel("Amplitude\n[arb. unit]")
    ax2.grid(which="major")
    
    ax3 = fig1.add_subplot(3,2,3)
    ax3.plot(w, np.squeeze(np.angle(Hw[0,0,:])), "k")
    ax3.set_xlim(w[0], w[-1]); ax3.set_ylim(-pi, 0)
    ax3.set_xlabel("Frequency [Hz]"); ax3.set_ylabel("Phase\n[rad]")
    ax3.axhline(y=0, color="k", zorder=1); ax3.axvline(x=0, color="k", zorder=1)
    ax3.grid(which="major")
    
    ax4 = fig1.add_subplot(3,2,4)
    ax4.plot(w, np.squeeze(np.angle(Hw[1,0,:])), "k")
    ax4.set_xlim(w[0], w[-1]); ax4.set_ylim(-pi, pi)
    ax4.set_xlabel("Frequency [Hz]"); ax4.set_ylabel("Phase\n[rad]")
    ax4.axhline(y=0, color="k", zorder=1); ax4.axvline(x=0, color="k", zorder=1)
    ax4.grid(which="major")    

    ax5 = fig1.add_subplot(3,2,5)
    ax5.plot(np.squeeze(np.real(Hw[0,0,:])), np.squeeze(np.imag(Hw[0,0,:])), "k")
    ax5.set_ylim(top=0)
    ax5.set_xlabel("Real axis [arb. unit]"); ax5.set_ylabel("Imaginary axis\n[arb. unit]")
    ax5.axhline(y=0, color="k", zorder=1); ax5.axvline(x=0, color="k", zorder=1)
    ax5.grid(which="major")
    
    ax6 = fig1.add_subplot(3,2,6)
    ax6.plot(np.squeeze(np.real(Hw[1,0,:])), np.squeeze(np.imag(Hw[1,0,:])), "k")
    ax6.set_xlabel("Real axis [arb. unit]"); ax6.set_ylabel("Imaginary axis\n[arb. unit]")
    ax6.axhline(y=0, color="k", zorder=1); ax6.axvline(x=0, color="k", zorder=1)
    plt.grid(which="major")
    fig1.tight_layout(); fig1.canvas.manager.window.showMaximized()
    
    # Plot frequency response function
    fig2 = plt.figure(); fig2.suptitle("Frequency response function\n of a 2DOF system", fontweight="bold")
    ax7 = fig2.add_subplot(3,2,1); ax7.set_title("Excitation and response measurement\nare co-located")
    ax7.semilogy(w, np.squeeze(np.abs(Hw[0,0,:])), "k", linewidth=5, label="Mode 1 + Mode 2")
    ax7.semilogy(w, np.abs(x_f1[:,0]), "c--", linewidth=2, label="Mode 1")
    ax7.semilogy(w, np.abs(x_f1[:,1]), "m--", linewidth=2, label="Mode 2")
    ax7.set_xlim(w[0], w[-1])
    ax7.set_xlabel("Frequency [Hz]"); ax7.set_ylabel("Amplitude\n[arb. unit]")
    ax7.grid(which="major")
    ax7.legend(loc="upper right")
    
    ax8 = fig2.add_subplot(3,2,2); ax8.set_title("Excitation and response measurement\nare not co-located")
    ax8.semilogy(w, np.squeeze(np.abs(Hw[1,0,:])), "k", linewidth=5, label="Mode 1 + Mode 2")
    ax8.semilogy(w, np.abs(x_f2[:,0]), "c--", linewidth=2, label="Mode 1")
    ax8.semilogy(w, np.abs(x_f2[:,1]), "m--", linewidth=2, label="Mode 2")
    ax8.set_xlim(w[0], w[-1])
    ax8.set_xlabel("Frequency [Hz]"); ax8.set_ylabel("Amplitude\n[arb. unit]")
    ax8.grid(which="major")
    ax8.legend(loc="upper right")
    
    ax9 = fig2.add_subplot(3,2,3)
    ax9.plot(w, np.squeeze(np.real(Hw[0,0,:])), "k", linewidth=5, label="Mode 1 + Mode 2")
    ax9.plot(w, np.real(x_f1[:,0]), "c--", linewidth=2, label="Mode 1")
    ax9.plot(w, np.real(x_f1[:,1]), "m--", linewidth=2, label="Mode 2")
    ax9.set_xlim(w[0], w[-1])
    ax9.set_xlabel("Frequency [Hz]"); ax9.set_ylabel("Real amplitude\n[arb. unit]")
    ax9.axhline(y=0, color="k", zorder=1); ax9.axvline(x=0, color="k", zorder=1)
    ax9.grid(which="major")
    ax9.legend(loc="upper right")
    
    ax10 = fig2.add_subplot(3,2,4)
    ax10.plot(w, np.squeeze(np.real(Hw[1,0,:])), "k", linewidth=5, label="Mode 1 + Mode 2")
    ax10.plot(w, np.real(x_f2[:,0]), "c--", linewidth=2, label="Mode 1")
    ax10.plot(w, np.real(x_f2[:,1]), "m--", linewidth=2, label="Mode 2")
    ax10.set_xlim(w[0], w[-1])
    ax10.set_xlabel("Frequency [Hz]"); ax10.set_ylabel("Real amplitude\n[arb. unit]")
    ax10.axhline(y=0, color="k", zorder=1); ax10.axvline(x=0, color="k", zorder=1)
    ax10.grid(which="major")
    ax10.legend(loc="upper right")
    
    ax11 = fig2.add_subplot(3,2,5)
    ax11.plot(w, np.squeeze(np.imag(Hw[0,0,:])), "k", linewidth=5, label="Mode 1 + Mode 2")
    ax11.plot(w, np.imag(x_f1[:,0]), "c--", linewidth=2, label="Mode 1")
    ax11.plot(w, np.imag(x_f1[:,1]), "m--", linewidth=2, label="Mode 2")
    ax11.set_xlim(w[0], w[-1]); ax11.set_ylim(top=0)
    ax11.set_xlabel("Frequency [Hz]"); ax11.set_ylabel("Imaginary amplitude\n[arb. unit]")
    ax11.axhline(y=0, color="k", zorder=1); ax11.axvline(x=0, color="k", zorder=1)
    ax11.grid(which="major")
    ax11.legend(loc="upper right")
    
    ax12 = fig2.add_subplot(3,2,6)
    ax12.plot(w, np.squeeze(np.imag(Hw[1,0,:])), "k", linewidth=5, label="Mode 1 + Mode 2")
    ax12.plot(w, np.imag(x_f2[:,0]), "c--", linewidth=2, label="Mode 1")
    ax12.plot(w, np.imag(x_f2[:,1]), "m--", linewidth=2, label="Mode 2")
    ax12.set_xlim(w[0], w[-1])
    ax12.set_xlabel("Frequency [Hz]"); ax12.set_ylabel("Imaginary amplitude\n[arb. unit]")
    ax12.axhline(y=0, color="k", zorder=1); ax12.axvline(x=0, color="k", zorder=1)
    ax12.grid(which="major")
    ax12.legend(loc="upper right")
    fig2.tight_layout(); fig2.canvas.manager.window.showMaximized()
    
    # Plot frequency response function (to prove symmetry)
    fig3 = plt.figure(); fig3.suptitle("Frequency response function\n of a 2DOF system", fontweight="bold")
    ax13 = fig3.add_subplot(1,1,1); ax13.set_title("Symmetry of the $|H(\omega)|$ matrix")
    ax13.semilogy(w, np.squeeze(np.abs(Hw[0,1,:])), "k", linewidth=5, label="$|H(\omega)| [0,1]$")
    ax13.semilogy(w, np.squeeze(np.abs(Hw[1,0,:])), "r--", linewidth=2, label="$|H(\omega)| [1,0]$")
    ax13.set_xlim(w[0], w[-1])
    ax13.set_xlabel("Frequency [Hz]"); ax13.set_ylabel("Amplitude [arb. unit]")
    ax13.grid(which="major")
    ax13.legend(loc="upper right")
    fig3.canvas.manager.window.showMaximized(); plt.show()

#################################################################
# Demo 2: Dynamic vibration absorber (a.k.a. tuned mass damper) #
#################################################################
if run_D2:
    # System definition
    m1    = 1
    m2    = 0.05*m1
    k1    = 1
    k2    = k1*m2/m1
    c1    = 0.002
    c2    = 0.0
    M     = np.array( [[m1, 0], [0, m2]] )
    K     = np.array( [[k1+k2, -k2], [-k2, k2]] )
    C     = np.array( [[c1+c2, -c2], [-c2, c2]] )
    w2, X = linalg.eigh(K, M)
    
    # Proportional damping
    zeta = 0.5 * np.diag(LA.multi_dot([X.T, C, X])) / np.sqrt(w2)
    
    # Mass ratio and natural frequencies
    mu = np.linspace(0, 5, 100001)
    wp = 1 + 0.5*mu + np.sqrt(mu+(0.25*(mu**2)))
    wm = 1 + 0.5*mu - np.sqrt(mu+(0.25*(mu**2)))
    
    # Frequency response function (physical)
    w     = np.linspace(0, 2*np.sqrt(w2[1]), 10000)
    Htva  = np.zeros((2, len(w)), dtype="complex_")
    Hsdof = np.zeros((1, len(w)), dtype="complex_")
    F = np.zeros(2); F[0] = 1
    for i in range(len(w)):
        Hsdof[:,i] = F[0] / (k1+(1j*w[i]*C[0,0])-((w[i]**2)*m1))
        Htva[:,i]  = LA.lstsq( K+(1j*w[i]*C)-((w[i]**2)*M), F, rcond=None )[0]
    
    # Equal-peak method
    k2   = 0.0454
    c2   = 0.0128
    K    = np.array( [[k1+k2, -k2], [-k2, k2]] )
    C    = np.array( [[c1+c2, -c2], [-c2, c2]] )
    Heqp = np.zeros( (2, len(w)), dtype="complex_" )
    for i in range(len(w)):
        Heqp[:,i] = LA.lstsq( K+(1j*w[i]*C)-((w[i]**2)*M), F, rcond=None )[0]
        
    # Plot figure 4
    fig4 = plt.figure(); fig4.suptitle("Effect of a dynamic vibration absorber on\nthe frequency response function of a 2DOF system", fontweight="bold")
    ax14 = fig4.add_subplot(2,1,1)
    ax14.semilogy(w, np.abs(Hsdof[0,:]), "--", label="Primary system")
    ax14.semilogy(w, np.abs(Htva[0,:]), label="Primary system +\nDynamic vibration absorber")
    ax14.set_xlim(w[0], w[-1])
    ax14.set_xlabel("$\dfrac{\omega}{\omega_{n}}$ [no unit]"); ax14.set_ylabel("$|x_{1}|$  [arb. unit]")
    ax14.legend(loc="upper right")
    
    ax15 = fig4.add_subplot(2,1,2)
    ax15.semilogy(w, np.abs(Htva[1,:]), "k", label="Dynamic vibration absorber")
    ax15.set_xlim(w[0], w[-1])
    ax15.set_xlabel("$\dfrac{\omega}{\omega_{n}}$ [no unit]"); ax15.set_ylabel("$|x_{2}|$  [arb. unit]")
    ax15.legend(loc="upper right")
    fig4.tight_layout()
    
    # Plot figure 5
    fig5 = plt.figure(); fig5.suptitle("Effect of the mass ratio on\nthe natural frequencies of the combined system", fontweight="bold")
    ax16 = fig5.add_subplot(1,1,1)
    ax16.plot(mu, wp, "k"); ax16.plot(mu, wm, "k")
    ax16.set_xlim(mu[0], mu[-1]); ax16.set_ylim(bottom=0)
    ax16.set_xlabel("$\dfrac{m_{2}}{m_{1}}$ [no unit]"); ax16.set_ylabel("$\dfrac{\omega_{1,2}^{2}}{\omega_{n}^{2}}$ [no unit]")
    
    # Plot figure 6
    fig6 = plt.figure(); fig6.suptitle("Effect of a dynamic vibration absorber on\nthe frequency response function of a 2DOF system", fontweight="bold")
    ax17 = fig6.add_subplot(2,1,1)
    ax17.semilogy(w, np.abs(Hsdof[0,:]), "--", label="Primary system")
    ax17.semilogy(w, np.abs(Htva[0,:]), label="Primary system + Dynamic vibration absorber\n(designed such that $\omega_{a}=\omega_{n}$)")
    ax17.semilogy(w, np.abs(Heqp[0,:]), label="Primary system + Dynamic vibration absorber\n(using the equal-peak method)")
    ax17.set_xlim(w[0], w[-1])
    ax17.set_xlabel("$\dfrac{\omega}{\omega_{n}}$ [no unit]"); ax17.set_ylabel("$|x_{1}|$  [arb. unit]")
    ax17.legend(loc="upper right", prop={"size":6})
    
    ax18 = fig6.add_subplot(2,1,2)
    ax18.semilogy(w, np.abs(Htva[1,:]), "k", label="Dynamic vibration absorber\n(designed such that $\omega_{a}=\omega_{n}$)")
    ax18.semilogy(w, np.abs(Heqp[1,:]), "darkgrey", label="Dynamic vibration absorber\n(using the equal-peak method)")
    ax18.set_xlim(w[0], w[-1])
    ax18.set_xlabel("$\dfrac{\omega}{\omega_{n}}$ [no unit]"); ax18.set_ylabel("$|x_{2}|$  [arb. unit]")
    ax18.legend(loc="upper right", prop={"size":6})
    fig6.tight_layout(); plt.show()