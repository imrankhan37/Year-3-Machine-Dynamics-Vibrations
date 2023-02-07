# -*- coding: utf-8 -*-
"""
Part II Spring Term
Lecture 3: Plate vibration, disc vibration, and FRFs

Last modified on Wed Sep 14 01:29:50 2022
@author: Dr Frederic Cegla, Amanda Lee
"""

###############################################################
# Import required functions, modules, packages, and libraries #
###############################################################
# Functions
from numpy.fft import fft as fft   # Import NumPy discrete Fourier transform functions

# Modules, packages, and libraries
import numpy as np                 # Import the fundamental package for scientific computing with Python
import matplotlib.pyplot as plt    # Import the state-based interface for interactive plots and simple cases of programmatic plot generation

#######################
# Step 1: Time domain #
#######################
t = np.linspace(0, 1-(1/512), 512) # Create time vector
y1 = np.zeros(len(t)) # Initialise signal y1
y2 = np.zeros(len(t)) # Initialise signal y2
y3 = np.zeros(len(t)) # Initialise signal y3
p1 = t[:6] # Initialise signal p1
p2 = t[:11] # Initialise signal p2
p3 = t[:21] # Initialise signal p3
y1[:6] = np.multiply( np.ones(len(p1)), np.hanning(len(p1)) ) # Create signal y1 (apply Hanning window to p1 signal)
y2[:11] = np.multiply( np.ones(len(p2)), np.hanning(len(p2)) ) # Create signal y2 (apply Hanning window to p2 signal)
y3[:21] = np.multiply( np.ones(len(p3)), np.hanning(len(p3)) ) # Create signal y3 (apply Hanning window to p3 signal)

fig = plt.figure("Figure 1: SDOF impact signals") # Create a new figure
fig.suptitle("SDOF impact signals", fontweight="bold") # Add a centered suptitle to the figure 

plt.subplot(2, 1, 1) # Add an axes to the current figure (top plot)
plt.title("Time domain") # Set a title for the axes
plt.plot(t, y1, linewidth=2, label=r"$\dfrac{5}{512}$ s impact") # Plot y1 vs t
plt.plot(t, y2, linewidth=2, label=r"$\dfrac{10}{512}$ s impact") # Plot y2 vs t
plt.plot(t, y3, linewidth=2, label=r"$\dfrac{20}{512}$ s impact") # Plot y3 vs t
plt.xlim(0, 1) # Set the x limits of the current axes
plt.ylim(bottom=0) # Set the y limits of the current axes
plt.xlabel("Time / s") # Set the label for the x-axis
plt.ylabel("Amplitude / arb. unit") # Set the label for the y-axis
plt.legend(loc="upper right") # Place a legend on the axes

############################
# Step 2: Frequency domain #
############################
T = t[-1] # Define time period
Fs = (len(t)-1)/T # Define sampling frequency
FNy = Fs/2 # Define Nyquist frequency
F = np.linspace(0, Fs, len(t)) # Create frequency vector
Fy1 = fft(y1) # Take FFT of signal y1
Fy2 = fft(y2) # Take FFT of signal y2
Fy3 = fft(y3) # Take FFT of signal y3

plt.subplot(2, 1, 2) # Add an axes to the current figure (bottom plot)
plt.title("Frequency domain") # Set a title for the axes
plt.plot(F, np.abs(Fy1), linewidth=2, label=r"$\dfrac{5}{512}$ s impact") # Plot Fy1 vs F (magnitude)
plt.plot(F, np.abs(Fy2), linewidth=2, label=r"$\dfrac{10}{512}$ s impact") # Plot Fy2 vs F (magnitude)
plt.plot(F, np.abs(Fy3), linewidth=2, label=r"$\dfrac{20}{512}$ s impact") # Plot Fy3 vs F (magnitude)
plt.xlim(0, FNy) # Display up to Nyquist frequency
plt.ylim(bottom=0) # Set the y limits of the current axes
plt.xlabel("Frequency / Hz") # Set the label for the x-axis
plt.ylabel("Amplitude / arb. unit") # Set the label for the y-axis
plt.legend(loc="upper right") # Place a legend on the axes

plt.subplots_adjust(hspace=0.5) # Adjust the subplot layout parameters
plt.show() # Display all open figures