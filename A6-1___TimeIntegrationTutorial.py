# -*- coding: utf-8 -*-
"""
Part I Autumn Term
Tutorial Questions Lecture 6: Direct time integration methods

This script demonstrates the Euler forward, Euler backward, and trapezoidal
rule methods (Question 1), and subsequently applies these methods to vibration
problems arising from external excitations (Question 2), single-degree-of-freedom
quarter-car models (Question 3), and dynamic vibration absorbers (Question 4).

Last modified on Tue Sep 13 23:39:45 2022
@author: Dr Ludovic Renson, Amanda Lee
"""

################################################
# Select question(s) to run and display output #
################################################
# Questions
run_Q1 = 1  #---> Edit "1" (True) or "0" (False)
run_Q2 = 0  #---> Edit "1" (Periodic excitation), "2" (Half-sine impulse), "3" (Sine sweep), or "0" (False)
run_Q3 = 0  #---> Edit "1" (True) or "0" (False)
run_Q4 = 0  #---> Edit "1" (True) or "0" (False)

###############################################################
# Import required functions, modules, packages, and libraries #
###############################################################
# Functions
from numpy import exp as exp, pi as pi, sin as sin, cos as cos # Import NumPy constants and mathematical functions
from numpy.fft import fft as fft, ifft as ifft                 # Import NumPy discrete Fourier transform functions
from scipy.integrate import solve_ivp as solve_ivp             # Import SciPy function to solve an initial value problem for a system of ODEs
def nextpow2(N):                                               # Create user-defined function to calculate the next higher power of 2
    n = 1
    while n < N: n *= 2
    return n

# Modules, packages, and libraries
import numpy as np                # Import the fundamental package for scientific computing with Python
import numpy.linalg as LA         # Import NumPy linear algebra functions package
import matplotlib.pyplot as plt   # Import a state-based interface to matplotlib for interactive plots and simple cases of programmatic plot generation

###########################################################################
# Question 1: Euler forward, Euler backward, and trapezoidal rule methods #
###########################################################################
if run_Q1:
    w = 2*pi
    A = np.array( [[0, 1], [-w**2, 0]] )
    u0 = [[1], [0]]
    t_lim = [0, 5]
    sol = solve_ivp(lambda t,x: np.dot(A, x), t_lim, np.squeeze(u0))
    t_ref, u_ref = sol.t, sol.y
    
    # Euler forward
    h = [0.005, 0.01]
    theta = np.linspace(0, 1.999*pi, 1000)
    
    fig1 = plt.figure(); fig1.suptitle("Direct time integration methods", fontweight="bold")
    ax1 = fig1.add_subplot(3,3,1); ax1.set_title("Euler forward", fontweight="bold")
    for ih in h:
        time = np.linspace(0, 5, int(5/ih)+1)
        u = np.zeros((2, len(time)))
        u[:,0] = np.squeeze(u0)
        for it in range(len(time)-1):
            u[:,it+1] = u[:,it] + ih*np.dot(A, u[:,it])
        ax1.plot(time, u[0,:], label=f"h = {ih}")
    ax1.plot(time, cos(w*time), "k--", label="Exact solution")
    ax1.set_xlim(time[0], time[-1]); ax1.set_ylim(min(u[0,:]), max(u[0,:]))
    ax1.set_xlabel("Time [s]"); ax1.set_ylabel("Displacement [arb. unit]")
    ax1.axhline(y=0, color="k", zorder=1); ax1.axvline(x=0, color="k", zorder=1)
    ax1.grid(which="major")
    ax1.legend(loc="upper right", prop={"size":8})
    ax1.annotate("Solution", xy=(0, 0.5), xytext=(-ax1.yaxis.labelpad-5, 0),
                xycoords=ax1.yaxis.label, textcoords='offset points',
                size='large', ha='right', va='center', rotation=90, fontweight="bold")
    
    ax2 = fig1.add_subplot(3,3,4)
    ax2.plot(cos(theta), sin(theta), "k")
    for ih in h:
        eigvals = LA.eigvals(np.identity(2)+(ih*A))
        ax2.plot(np.real(eigvals), np.imag(eigvals), "x", markersize=10, label=f"h = {ih}")
    ax2.set_xlabel("$\mathbb{R}$ $(\lambda_{A})$  [arb. unit]"); ax2.set_ylabel("$\mathbb{I}$ $(\lambda_{A})$  [arb. unit]")
    ax2.axhline(y=0, color="k", zorder=1); plt.axvline(x=0, color="k", zorder=1)
    ax2.grid(which="major")
    ax2.legend(loc="upper right", prop={"size":8})
    ax2.annotate("Eigenvalue", xy=(0, 0.5), xytext=(-ax2.yaxis.labelpad-5, 0),
                xycoords=ax2.yaxis.label, textcoords='offset points',
                size='large', ha='right', va='center', rotation=90, fontweight="bold")
    
    ax3 = fig1.add_subplot(3,3,7)
    ax3.fill(cos(theta)-1, sin(theta), edgecolor="k", linewidth=2, zorder=2, label="Stable region")
    ax3.set_xlim(-3, 3); ax3.set_ylim(-3, 3)
    ax3.set_xlabel("$\mathbb{R}$ $(h\lambda_{A})$  [arb. unit]"); ax3.set_ylabel("$\mathbb{I}$ $(h\lambda_{A})$  [arb. unit]")
    ax3.axhline(y=0, color="k", zorder=1); ax3.axvline(x=0, color="k", zorder=1)
    ax3.legend(loc="upper right", prop={"size":8})
    ax3.annotate("Stability diagram", xy=(0, 0.5), xytext=(-ax3.yaxis.labelpad-5, 0),
                xycoords=ax3.yaxis.label, textcoords='offset points',
                size='large', ha='right', va='center', rotation=90, fontweight="bold")
    
    # Euler backward
    h = [0.005, 0.01]
    theta = np.linspace(0, 1.999*pi, 1000)
    
    ax4 = fig1.add_subplot(3,3,2); ax4.set_title("Euler backward", fontweight="bold")
    for ih in h:
        time = np.linspace(0, 5, int(5/ih)+1)
        u = np.zeros((2, len(time)))
        u[:,0] = np.squeeze(u0)
        for it in range(len(time)-1):
            u[:,it+1] = LA.lstsq(np.identity(2)-(ih*A), u[:,it], rcond=None)[0]
        ax4.plot(time, u[0,:], label=f"h = {ih}")
    ax4.plot(time, cos(w*time), "k--", label="Exact solution")
    ax4.set_xlim(time[0], time[-1]); ax4.set_ylim(-1, 1)
    ax4.set_xlabel("Time [s]"); ax4.set_ylabel("Displacement [arb. unit]")
    ax4.axhline(y=0, color="k", zorder=1); ax4.axvline(x=0, color="k", zorder=1)
    ax4.grid(which="major")
    ax4.legend(loc="upper right", prop={"size":8})
    
    ax5 = fig1.add_subplot(3,3,5)
    ax5.plot(cos(theta), sin(theta), "k")
    for ih in h:
        eigvals = LA.eigvals( LA.lstsq(np.identity(2)-(ih*A), np.identity(2), rcond=None)[0] )
        ax5.plot(np.real(eigvals), np.imag(eigvals), "x", markersize=10, label=f"h = {ih}")
    ax5.set_xlabel("$\mathbb{R}$ $(\lambda_{A})$  [arb. unit]"); ax5.set_ylabel("$\mathbb{I}$ $(\lambda_{A})$  [arb. unit]")
    ax5.axhline(y=0, color="k", zorder=1); ax5.axvline(x=0, color="k", zorder=1)
    ax5.grid(which="major")
    ax5.legend(loc="upper right", prop={"size":8})
    
    ax6 = fig1.add_subplot(3,3,8)
    ax6.fill([-3,3,3,-3], [-3,-3,3,3], zorder=1, label="Stable region")
    ax6.fill(1-cos(theta), -sin(theta), facecolor="w", edgecolor="k", linewidth=2, zorder=3)
    ax6.set_xlim(-3, 3); ax6.set_ylim(-3, 3)
    ax6.set_xlabel("$\mathbb{R}$ $(h\lambda_{A})$  [arb. unit]"); ax6.set_ylabel("$\mathbb{I}$ $(h\lambda_{A})$  [arb. unit]")
    ax6.axhline(y=0, color="k", zorder=2); ax6.axvline(x=0, color="k", zorder=2)
    ax6.legend(loc="upper right", prop={"size":8})
    
    # Trapezoidal rule
    h = [0.005, 0.01]
    theta = np.linspace(0, 1.999*pi, 1000)
    
    ax7 = fig1.add_subplot(3,3,3); ax7.set_title("Trapezoidal rule", fontweight="bold")
    for ih in h:
        time = np.linspace(0, 5, int(5/ih)+1)
        u = np.zeros((2, len(time)))
        u[:,0] = np.squeeze(u0)
        for it in range(len(time)-1):
            u[:,it+1] = LA.lstsq(np.identity(2)-(0.5*ih*A), np.dot(np.identity(2)+(0.5*ih*A), u[:,it]), rcond=None)[0]
        ax7.plot(time, u[0,:], label=f"h = {ih}")
    ax7.plot(time, cos(w*time), "k--", label="Exact solution")
    ax7.set_xlim(time[0], time[-1]); ax7.set_ylim(min(u[0,:]), max(u[0,:]))
    ax7.set_xlabel("Time [s]"); ax7.set_ylabel("Displacement [arb. unit]")
    ax7.axhline(y=0, color="k", zorder=1); ax7.axvline(x=0, color="k", zorder=1)
    ax7.grid(which="major")
    ax7.legend(loc="upper right", prop={"size":8})
    
    ax8 = fig1.add_subplot(3,3,6)
    ax8.plot(cos(theta), sin(theta), "k")
    for ih in h:
        eigvals = LA.eigvals( LA.lstsq(np.identity(2)-(0.5*ih*A), np.identity(2)+(0.5*ih*A), rcond=None)[0] )
        ax8.plot(np.real(eigvals), np.imag(eigvals), "x", markersize=10, label=f"h = {ih}")
    ax8.set_xlabel("$\mathbb{R}$ $(\lambda_{A})$  [arb. unit]"); ax8.set_ylabel("$\mathbb{I}$ $(\lambda_{A})$  [arb. unit]")
    ax8.axhline(y=0, color="k", zorder=1); ax8.axvline(x=0, color="k", zorder=1)
    ax8.grid(which="major")
    ax8.legend(loc="upper right", prop={"size":8})
    
    ax9 = fig1.add_subplot(3,3,9)
    ax9.fill([-3,0,0,-3], [-3,-3,3,3], zorder=1, label="Stable region")
    ax9.set_xlim(-3, 3); ax9.set_ylim(-3, 3)
    ax9.set_xlabel("$\mathbb{R}$ $(h\lambda_{A})$  [arb. unit]"); ax9.set_ylabel("$\mathbb{I}$ $(h\lambda_{A})$  [arb. unit]")
    ax9.axhline(y=0, color="k", zorder=2); ax9.axvline(x=0, color="k", zorder=2)
    ax9.legend(loc="upper right", prop={"size":8})
    fig1.subplots_adjust(hspace=0.4, wspace=0.4); fig1.canvas.manager.window.showMaximized(); plt.show()
    
###################################
# Question 2: External excitation #
###################################
# Demo 1: Periodic excitation
if run_Q2==1:
    m = 1
    k = 1
    c = 0.05
    w2 = k/m
    zeta = 0.5 * c / np.sqrt(w2)
    
    A = np.array( [[0, 1], [-k/m, -c/m]] )
    B = np.array( [[0], [1/m]] )
    
    h = 0.01
    a = 1
    w = np.linspace(0, 5, 1001)
    Xtime = np.zeros(len(w))
    Xfreq = a*1/(k - ((w**2)*m) + (1j*w*c))
    u0 = np.array( [[np.abs(Xfreq[0])], [0]] )
    
    for iw in range(1,len(w)):
        T = 2*pi/w[iw]
        time = np.linspace(0, round(5*T,2), int(5*T/h)+1)
        t_lim = np.array( [time[0], time[-1]] )
        nsamples_period = int( np.ceil(T/h) )
        u = np.zeros( (2, len(time)) )
        u[:,0] = np.squeeze(u0)
        F = a * cos(w[iw]*time)
        for it in range(len(time)-1):
            arg1 = np.identity(2)-(0.5*h*A)
            arg2 = np.dot(np.identity(2)+(0.5*h*A),u[:,it]).T + np.squeeze( 0.5*h*B*(F[it]+F[it+1]) )
            u[:,it+1] = LA.solve(arg1, arg2)
        Xtime[iw] = max( np.abs(u[0, -nsamples_period:]) )
        u0 = u[:,-1]

    fig2 = plt.figure(); fig2.suptitle("Periodic excitation", fontweight="bold")
    ax10 = fig2.add_subplot(1,1,1); ax10.set_title("Frequency-domain signal")
    ax10.semilogy(w, np.abs(Xfreq), label="Frequency-domain method")
    ax10.semilogy(w, Xtime, ":", label="Time-domain method")
    ax10.set_xlim(w[0], w[-1])
    ax10.set_xlabel("Excitation frequency [Hz]"); ax10.set_ylabel("Response amplitude [arb. unit]")
    ax10.legend(loc="upper right")
    plt.show()
    
# Demo 2: Half-sine impulse
elif run_Q2==2:
    m = 1
    k = 1
    c = 0.05
    w2 = k/m
    zeta = 0.5 * c / np.sqrt(w2)

    A = np.array( [[0, 1], [-k/m, -c/m]] )
    B = np.array( [[0], [1/m]] )
    u0 = np.array( [[0], [0]] )
    
    h = 0.01
    a = 1
    tend = 10000
    nsamples = nextpow2((tend-h)/h)
    time = h * np.linspace(0, nsamples-1, nsamples)
    fs = 1/h
    freq = 0.5 * fs * np.linspace(0, 1, (nsamples//2)+1)
    w = 2*pi*freq
    T0 = 1
    W0 = 2*pi/T0
    F = a * sin(W0*time)
    F[int(0.5*T0/h):] = 0
    
    Fw = fft(F) / nsamples
    Fwabs = np.abs( Fw[:(nsamples//2)+1] )
    Fwabs[1:-1] = 2*Fwabs[1:-1]
    Ft_rec = ifft(Fw) * nsamples
    
    wfft = np.append(w, np.flip(w[1:-1]))
    Xw = Fw / (k+(1j*wfft*c)-((wfft**2)*m))
    Xw = np.append(Xw[:(nsamples//2)+1], np.conj(np.flip(Xw[1:nsamples//2])))
    Xt_rec = np.real(ifft(Xw)) * nsamples
    
    u = np.zeros((2,len(time)))
    u[:,0] = np.squeeze(u0)
    for it in range(len(time)-1):
        arg1 = np.identity(2)-(0.5*h*A)
        arg2 = np.dot(np.identity(2)+(0.5*h*A),u[:,it]).T + np.squeeze( 0.5*h*B*(F[it]+F[it+1]) )
        u[:,it+1] = LA.solve(arg1, arg2)
    Xwt = fft(u[0,:]) / nsamples
    Xwabs = np.abs(Xwt[:(nsamples//2)+1])
    Xwabs[1:-1] = 2*Xwabs[1:-1]
    Fwa = (-W0*(exp(-pi*wfft*1j/W0)+1)) / ((wfft+W0)*(wfft-W0))
    Fwa = np.append(Fwa[:(nsamples//2)+1], np.conj(np.flip(Fwa[1:nsamples//2])))
    Fta_rec = (1/h) * np.real(ifft(Fwa))

    Xwta = Fwa / (k+(1j*wfft*c)-((wfft**2)*m)) / nsamples / h
    Xwta = np.append(Xwta[:(nsamples//2)+1], np.conj(np.flip(Xwta[1:nsamples//2])))
    Xta_rec = np.real(ifft(Xw)) * nsamples
    
    fig3 = plt.figure(); fig3.suptitle("Half-sine impulse", fontweight="bold")
    ax11 = fig3.add_subplot(1,2,1); ax11.set_title("Time-domain signal")
    ax11.plot(time, F)
    ax11.set_xlim(0, 2); ax11.set_ylim(0, 1)
    ax11.set_xlabel("Time [s]"); ax11.set_ylabel("f(t)  [arb. unit]")
    
    ax12 = fig3.add_subplot(1,2,2); ax12.set_title("Frequency spectrum")
    ax12.semilogy(w, Fwabs)
    ax12.set_xlim(0, 50)
    ax12.set_xlabel("Frequency [Hz]"); ax12.set_ylabel("$|F(\omega)|$  [arb. unit]")
    fig3.tight_layout()
    
    fig4 = plt.figure(); fig4.suptitle("Response to the half-sine impulse", fontweight="bold")
    ax13 = fig4.add_subplot(2,2,1); ax13.set_title("Time-domain signal")
    ax13.plot(time, u[0,:], label="Time-domain method")
    ax13.plot(time, Xt_rec, ":", label="Frequency-domain method")
    ax13.set_xlim(0, 100)
    ax13.set_xlabel("Time [s]"); ax13.set_ylabel("x(t)  [arb. unit]")
    ax13.axhline(y=0, color="k", zorder=1); ax13.axvline(x=0, color="k", zorder=1)
    ax13.legend(loc="upper right")
    
    ax14 = fig4.add_subplot(2,2,2); ax14.set_title("Frequency-domain signal")
    ax14.semilogy(wfft, np.abs(Xw), label="Frequency-domain method")
    ax14.semilogy(wfft, np.abs(Xwt), ":", label="Time-domain method")
    ax14.set_xlim(w[0], w[-1])
    ax14.set_xlabel("Frequency [Hz]"); ax14.set_ylabel("$|X(\omega)|$  [arb. unit]")
    ax14.legend(loc="upper right")
    
    ax15 = fig4.add_subplot(2,2,3)
    ax15.plot(time, u[0,:], label="Time-domain method")
    ax15.plot(time, Xta_rec, ":", label="Frequency-domain method (analytical)")
    ax15.set_xlim(0, 100)
    ax15.set_xlabel("Time [s]"); ax15.set_ylabel("x(t)  [arb. unit]")
    ax15.axhline(y=0, color="k", zorder=1); ax15.axvline(x=0, color="k", zorder=1)
    ax15.legend(loc="upper right")
    
    ax16 = fig4.add_subplot(2,2,4)
    ax16.semilogy(wfft, np.abs(Xw), label="Frequency-domain method")
    ax16.semilogy(wfft, np.abs(Xwta), ":", label="Frequency-domain method (analytical)")
    ax16.set_xlim(w[0], w[-1])
    ax16.set_xlabel("Frequency [Hz]"); ax16.set_ylabel("$|X(\omega)|$  [arb. unit]")
    ax16.legend(loc="upper right")
    fig4.tight_layout(); fig4.canvas.manager.window.showMaximized()
    
    fig5 = plt.figure(); fig5.suptitle("Reconstruct of the half-sine impulse", fontweight="bold")
    ax17 = fig5.add_subplot(2,1,1); ax17.set_title("Time-domain signal")
    ax17.plot(time, F, label="Original signal")
    ax17.plot(time, Ft_rec, ":", label="Reconstructed signal\n(Frequency-domain method)")
    ax17.set_xlim(0, 2); ax17.set_ylim(0, 1)
    ax17.set_xlabel("Time [s]"); ax17.set_ylabel("f(t)  [arb. unit]")
    ax17.legend(loc="upper right")
    
    ax18 = fig5.add_subplot(2,1,2)
    ax18.plot(time, F, label="Original signal")
    ax18.plot(time, Fta_rec, ":", label="Reconstructed signal\n(Frequency-domain method analytical)")
    ax18.set_xlim(0, 2); ax18.set_ylim(0, 1)
    ax18.set_xlabel("Time [s]"); ax18.set_ylabel("f(t)  [arb. unit]")
    ax18.legend(loc="upper right")
    fig5.tight_layout(); plt.show()

# Demo 3: Sine sweep
elif run_Q2==3:
    m = 1
    k = 1
    c = 0.05
    w2 = k/m
    zeta = 0.5 * c / np.sqrt(w2)

    A = np.array( [[0, 1], [-k/m, -c/m]] )
    B = np.array( [[0], [1/m]] )
    u0 = np.array( [[0], [0]] )
    
    h = 0.01
    a = 1
    f0 = 0.1 / (2*pi)
    fend = 5 / (2*pi)
    w = np.linspace(0, 5, 1001)
    Xfreq = a*1/(k - ((w**2)*m) + (1j*w*c))
    rates = 0.1 * np.array([0.1, 1, 3, 5]) / (2*pi)
    
    fig6 = plt.figure(); fig6.suptitle("Sine sweep response", fontweight="bold")
    ax19 = fig6.add_subplot(1,1,1); ax19.set_title("Frequency-domain signal")
    for ir in rates:
        tend = (fend-f0) * (60/ir)
        k = np.sign(ir) * np.abs(fend-f0) / tend
        time = np.linspace(0, tend, int(tend/h)+1)
        finst = (k*time) + f0
        psi = (2*pi) * ((f0*time) + (0.5*k*(time**2)))
        u = np.zeros( (2, len(time)) )
        u[:,0] = np.squeeze(u0)
        F = a * sin(psi)
        for it in range(len(time)-1):
            arg1 = np.identity(2)-(0.5*h*A)
            arg2 = np.dot(np.identity(2)+(0.5*h*A),u[:,it]).T + np.squeeze( 0.5*h*B*(F[it]+F[it+1]) )
            u[:,it+1] = LA.solve(arg1, arg2)
        ax19.plot((2*pi*finst), u[0,:], label=f"Time-domain method\n(r = {round(ir, -int(np.floor(np.log10(abs(ir)))))} Hz/min)")
    ax19.plot(w, np.abs(Xfreq), "k", label="Frequency-domain method")
    ax19.set_xlim((2*pi*f0), 3)
    ax19.set_xlabel("Instantaneous frequency [Hz]"); ax19.set_ylabel("Response amplitude [arb. unit]")
    ax19.axhline(y=0, color="k", zorder=1); ax19.axvline(x=0, color="k", zorder=1)
    ax19.legend(loc="upper right")
    
    fig7 = plt.figure()
    fig7.suptitle("Sine sweep excitation", fontweight="bold")
    ax20 = fig7.add_subplot(1,1,1); ax20.set_title("Time-domain signal")
    ax20.plot(time, F)
    ax20.set_xlim(0, 150); ax20.set_ylim(-1, 1)
    ax20.set_xlabel("Time [s]"); ax20.set_ylabel("Excitation amplitude [arb. unit]")
    ax20.axhline(y=0, color="k", zorder=1); ax20.axvline(x=0, color="k", zorder=1)
    plt.show()

######################################
# Question 3: SDOF quarter-car model #
######################################
if run_Q3:
    m = 1
    k = 1
    c = 0.05
    w2 = k/m
    zeta = 0.5 * c / np.sqrt(w2)

    A = np.array( [[0, 1], [-k/m, -c/m]] )
    B = np.array( [[0], [1/m]] )
    
    h = 0.01
    a = 0.1
    w = np.linspace(0, 5, 1000)
    Xtime = np.zeros(len(w))
    Xfreq = a * (k+(1j*w*c)) / (k-((w**2)*m)+(1j*w*c))
    u0 = np.array( [[np.abs(Xfreq[0])], [0]] )

    for iw in range(1,len(w)):
        T = 2*pi/w[iw]
        time = np.linspace(0, round(5*T,2), int(5*T/h)+1)
        nsamples_period = int( np.ceil(T/h) )
        u = np.zeros( (2, len(time)) )
        u[:,0] = np.squeeze(u0)
        F = ((k*a*cos(w[iw]*time)) - (c*w[iw]*a*sin(w[iw]*time))) / m
        for it in range(len(time)-1):
            arg1 = np.identity(2)-(0.5*h*A)
            arg2 = np.dot(np.identity(2)+(0.5*h*A),u[:,it]).T + np.squeeze( 0.5*h*B*(F[it]+F[it+1]) )
            u[:,it+1] = LA.lstsq(arg1, arg2, rcond=None)[0]
        Xtime[iw] = max( np.abs(u[0, -nsamples_period:]) )
        u0 = u[:,-1]

    fig8 = plt.figure(); fig8.suptitle("Harmonic road profile", fontweight="bold")
    ax21 = fig8.add_subplot(1,1,1); ax21.set_title("Frequency-domain response")
    ax21.semilogy(w, np.abs(Xfreq), label="Frequency-domain method")
    ax21.semilogy(w, Xtime, ":", label="Time-domain method")
    ax21.set_xlim(w[0], w[-1])
    ax21.set_xlabel("Excitation frequency [Hz]"); ax21.set_ylabel("Response amplitude [arb. unit]")
    ax21.legend(loc="upper right")
    
    w = w[750]
    time = np.linspace(0, (200*2*pi/w), int((200*2*pi/w)/h)+1)
    Fsin = ((k*a*cos(w*time)) - (c*w*a*sin(w*time))) / m
    T0 = 1
    W0 = 2*pi/T0
    Fimpulse = a*sin(W0*time)
    nsamples_halfperiod = int(0.5*T0/h)
    Fimpulse[nsamples_halfperiod:] = 0
    Ftot = Fsin
    Ftot[(len(Ftot)//2) : (len(Ftot)//2)+nsamples_halfperiod] = Fimpulse[:nsamples_halfperiod]

    u = np.zeros( (2, len(time)) )
    u[:,0] = np.squeeze(u0)
    for it in range(len(time)-1):
        arg1 = np.identity(2)-(0.5*h*A)
        arg2 = np.dot(np.identity(2)+(0.5*h*A),u[:,it]).T + np.squeeze( 0.5*h*B*(Ftot[it]+Ftot[it+1]) )
        u[:,it+1] = LA.solve(arg1, arg2)

    fig9 = plt.figure(); fig9.suptitle("Harmonic road profile + Half-sine impulse", fontweight="bold")
    ax22 = fig9.add_subplot(2,1,1); ax22.set_title("Time-domain excitation")
    ax22.plot(time, Ftot)
    ax22.set_xlim(time[0], time[-1])
    ax22.set_xlabel("Time [s]"); ax22.set_ylabel("Excitation amplitude\n[arb. unit]")
    ax22.axhline(y=0, color="k", zorder=1); ax22.axvline(x=0, color="k", zorder=1)

    ax23 = fig9.add_subplot(2,1,2); ax23.set_title("Time-domain response")
    ax23.plot(time, u[0,:], label="Mode 1")
    ax23.set_xlim(time[0], time[-1])
    ax23.set_xlabel("Time [s]"); ax23.set_ylabel("Response amplitude\n[arb. unit]")
    ax23.axhline(y=0, color="k", zorder=1); ax23.axvline(x=0, color="k", zorder=1)
    ax23.legend(loc="upper right")
    fig9.tight_layout(); plt.show()

##########################################################
# Question 4: Transient response of the dynamic absorber #
##########################################################
if run_Q4:
    m1 = 1
    m2 = 0.05*m1
    k1 = 1
    k2 = k1*m2/m1
    c1 = 0.002
    c2 = 0.0
    M = np.array( [[m1, 0], [0, m2]] )
    C = np.array( [[c1+c2, -c2], [-c2, c2]] )
    K = np.array( [[k1+k2, -k2], [-k2, k2]] )
    
    A11 = np.zeros((2,2))
    A12 = np.identity(2)
    A21 = LA.lstsq(-M, K, rcond=None)[0]
    A22 = LA.lstsq(-M, C, rcond=None)[0]
    A1 = np.append(A11, A12, axis=1)
    A2 = np.append(A21, A22, axis=1)
    A = np.append(A1, A2, axis=0)
    B21 = np.zeros((2,1))
    B22 = LA.lstsq(M, np.ones((2,1)), rcond=None)[0]
    B1 = np.array([[0], [0], [1], [0]])
    B2 = np.append(B21, B22, axis=0)
    B =  np.multiply(B1, B2)
    
    h = 0.01
    a = 1
    T0 = 1
    W0 = 2*pi/T0
    tend = 10000
    nsamples = nextpow2( (tend-h)/h )
    time = h * np.linspace(0, nsamples-1, nsamples)
    F = a*sin(W0*time)
    F[int(0.5*T0/h):] = 0
    
    u = np.zeros( (4, len(time)) )
    for it in range(len(time)-1):
        arg1 = np.identity(4)-(0.5*h*A)
        arg2 = np.dot(np.identity(4)+(0.5*h*A),u[:,it]).T + np.squeeze( 0.5*h*B*(F[it]+F[it+1]) )
        u[:,it+1] = LA.solve(arg1, arg2)
    
    fig10 = plt.figure(); fig10.suptitle("Half-sine impulse", fontweight="bold")
    ax24 = fig10.add_subplot(2,1,1); ax24.set_title("Time-domain excitation")
    ax24.plot(time, F)
    ax24.set_xlim(time[0], time[-1]); ax24.set_ylim(0,1)
    ax24.set_xlabel("Time [s]"); ax24.set_ylabel("Excitation amplitude\n[arb. unit]")
    ax24.axhline(y=0, color="k", zorder=1); ax24.axvline(x=0, color="k", zorder=1)
       
    ax25 = fig10.add_subplot(2,1,2); ax25.set_title("Time-domain response")
    ax25.plot(time, u[2,:], zorder=3, label="Mode 3")
    ax25.plot(time, u[3,:], zorder=2, label="Mode 4")
    ax25.set_xlim(time[0], time[-1])
    ax25.set_xlabel("Time [s]"); ax25.set_ylabel("Response amplitude\n[arb. unit]")
    ax25.axhline(y=0, color="k", zorder=1); ax25.axvline(x=0, color="k", zorder=1)
    ax25.legend(loc="upper right")
    fig10.tight_layout(); plt.show()