# -*- coding: utf-8 -*-
"""
Part I Autumn Term
Tutorial Questions Lecture 4: Single-degree-of-freedom Systems

This script generates a single-degree-of-freedom model (Question 1).
Note that there is no code for Questions 2.

Last modified on Mon Sep 12 13:08:40 2022
@author: Dr Frederic Cegla, Amanda Lee
"""

###############################################################
# Import required functions, modules, packages, and libraries #
###############################################################
# Functions
from numpy import pi as pi, cos as cos  # Import NumPy constants and mathematical functions

# Modules, packages, and libraries
import numpy as np                      # Import the fundamental package for scientific computing with Python
import matplotlib.pyplot as plt         # Import the state-based interface to matplotlib for interactive plots and simple cases of programmatic plot generation
import tkinter as tk                    # Import the standard Python interface to the Tcl/Tk GUI toolkit
import tkinter.messagebox as messagebox # Import Tkinter message prompts module

##############################################
# Question 1: Single-degree-of-freedom model #
##############################################
# System parameters
f      = np.arange(1, 501, 1)  # Create frequency vector in Hz
m1     = 1                     #---> Edit mass in kg
f1     = 100                   #---> Edit resonance frequency in Hz
c1     = 0.05                  #---> Edit damping ratio
w      = 2*pi*f                # Convert frequency vector to rad/s
w1     = 2*pi*f1               # Find resonance frequency in rad/s
k1     = m1*(w1**2)            # Find stiffness in N/m
x      = w/w1                  # Create normalised frequency vector
a      = 1-(x**2)              # Create real denominator vector for frequency response function
b      = 2*c1*x                # Create imaginary denominator vector for frequency response function
denom  = a + 1j*b              # Create complex denominator vector for frequency response function
h      = 1/k1/denom            # Find frequency response function
rrecap = np.real(h)            # Find real part of frequency response function
irecap = np.imag(h)            # Find imaginary part of frequency response function
mag    = np.abs(h)             # Find magnitude of frequency response function
phase  = np.angle(h, deg=True) # Find phase of frequency response function in deg

# Plot frequency response function
fig1 = plt.figure(); fig1.suptitle("Single-degree-of-freedom model", fontweight="bold")

ax1 = fig1.add_subplot(2, 2, 1); ax1.set_title("Bode diagram (magnitude)")
line1, = ax1.semilogy(f, mag, "r", picker=True, pickradius=2)
line2, = ax1.plot([0,0], [min(mag),max(mag)], "k--"); line3, = ax1.plot([], [], "ko")
ax1.set_xlim(f[0], f[-1]); ax1.set_ylim(min(mag), max(mag))
ax1.set_xlabel("Frequency [Hz]"); ax1.set_ylabel("Log receptance [no unit]")
ax1.grid(which="major")
ax1.grid(which="minor", color="#BBBBBB", linestyle="--", linewidth=0.5); ax1.minorticks_on()

ax2 = fig1.add_subplot(2, 2, 3); ax2.set_title("Bode diagram (phase)")
line4, = ax2.plot(f, phase)
line5, = ax2.plot([0,0], [-180, 0], "k--"); line6, = ax2.plot([], [], "ko")
ax2.set_xlim(f[0], f[-1]); ax2.set_ylim(min(phase), max(phase))
ax2.set_xlabel("Frequency [Hz]"); ax2.set_ylabel("Phase [deg]")
ax2.grid(which="major")
ax2.grid(which="minor", color="#BBBBBB", linestyle="--", linewidth=0.5); ax2.minorticks_on()

ax3 = fig1.add_subplot(2, 2, 2); ax3.set_title("Nyquist plot")
line7, = ax3.plot(rrecap, irecap, "+-") # Plot y vs t
line8, = ax3.plot([0, rrecap[0]], [0, irecap[0]], "ko--")
ax3.set_ylim(top=0)
ax3.set_xlabel("Real axis [arb. unit]"); ax3.set_ylabel("Imaginary axis [arb. unit]")
ax3.axhline(y=0, color="k", zorder=1); ax3.axvline(x=0, color="k", zorder=1)
ax3.grid(which="major")
ax3.grid(which="minor", color="#BBBBBB", linestyle="--", linewidth=0.5); ax3.minorticks_on()

ax4 = fig1.add_subplot(2, 2, 4); ax5 = ax4.twinx(); ax4.set_title("Time history")
line9, = ax4.plot([], [], "b"); line10, = ax5.plot([], [], "g")
ax4.set_ylim(-1, 1)
ax4.set_xlabel("Time [s]"); ax4.set_ylabel("Amplitude [N]", color="b"); ax5.set_ylabel("Amplitude [mm]", color="g")
ax4.tick_params(axis="y", colors="b"); ax5.tick_params(axis="y", colors="g")
ax4.axhline(y=0, color="k", zorder=1); ax4.axvline(x=0, color="k", zorder=1)
ax4.grid(which="major")
ax4.grid(which="minor", color="#BBBBBB", linestyle="--", linewidth=0.5); ax4.minorticks_on()

# Create interactive plot tool
def onpick(event):
    myLine = event.artist
    xData = myLine.get_xdata()
    yData = myLine.get_ydata()
    idx = event.ind
    points = tuple(zip(xData[idx], yData[idx]))
    myIdx = idx[0]
    myXPoint = points[0][0]
    myYPoint = points[0][1]
    
    t = np.linspace(0, 3/myXPoint, 61)
    F = cos(2*pi*myXPoint*t)
    x1 = 1000 * mag[myIdx] * cos(2*pi*myXPoint*t + phase[myIdx]*(pi/180))
    
    line2.set_xdata( [myXPoint, myXPoint] )
    line3.set_xdata( [myXPoint] )
    line3.set_ydata( [myYPoint] )
    line5.set_xdata( [myXPoint, myXPoint] )
    line6.set_xdata( [myXPoint] )
    line6.set_ydata( [phase[myIdx]] )
    line8.set_xdata( [0, rrecap[myIdx]] )
    line8.set_ydata( [0, irecap[myIdx]] )
    line9.set_xdata(t)
    line9.set_ydata(F)
    line10.set_xdata(t)
    line10.set_ydata(x1)
    ax4.set_xlim(t[0], t[-1])
    
    fig1.canvas.draw_idle()

fig1.canvas.mpl_connect("pick_event", onpick)
fig1.tight_layout(); fig1.canvas.manager.window.showMaximized(); plt.show()

root = tk.Tk(); root.lift()
messagebox.showinfo(title="Help dialog",\
                    message=("Click any point in the top left plot to select a"
                             " frequency at which the time history is plotted"
                             " in the bottom right plot.")); root.destroy()

#######################################################################
# Question 2: Quarter-car model with rigid tyres (SDOF approximation) #
#######################################################################
# No code for Question 2