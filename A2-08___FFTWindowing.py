# -*- coding: utf-8 -*-
"""
Part I Autumn Term
Lecture 2: Frequency domain analysis

Fast Fourier Transform - windowing.

Last modified on Thu Sep 15 04:56:17 2022
@author: Dr Frederic Cegla, Amanda Lee
"""

###############################################################
# Import required functions, modules, packages, and libraries #
###############################################################
# Functions
from numpy import pi as pi, sin as sin # Import NumPy constants and mathematical functions
from numpy.fft import fft as fft       # Import NumPy discrete Fourier transform functions

# Modules, packages, and libraries
import numpy as np                     # Import the fundamental package for scientific computing with Python
import matplotlib.pyplot as plt        # Import the state-based interface for interactive plots and simple cases of programmatic plot generation

##########################################################
# Step 1: Signal Processing (time and frequency domains) #
##########################################################
t = np.linspace(0, 0.511, 512) # Create time vector
y = sin(2*pi*32.25*t) # Create signal t
yw = np.multiply( y, np.hanning(len(y)) ) # Create signal yw (apply Hanning window to signal y)

plt.figure("Figure 1: Time domain signals") # Create a new figure
plt.title("Time domain signals") # Set a title for the axes
plt.plot(t, y) # Plot y vs t
plt.plot(t, yw) # Plot yw vs t
plt.xlim(0, 0.5) # Set the x limits of the current axes
plt.ylim(-1, 1) # Set the y limits of the current axes
plt.xlabel("Time / s") # Set the label for the x-axis
plt.ylabel("Amplitude / arb. unit") # Set the label for the y-axis

############################################
# Step 2: Display frequency domain signals #
############################################
T = t[-1] # Define time period
Fs = (len(t)-1)/T # Define sampling frequency
FNy = Fs/2 # Define Nyquist frequency
F = np.linspace(0, Fs, len(t)) # Create frequency vector
Fy = np.zeros(len(F), dtype="complex_")
Fy = fft(y) # Take FFT of signal y
Fyw = fft(yw) # Take FFT of signal yw

plt.figure("Figure 2: Frequency domain signals") # Create a new figure
plt.title("Frequency domain signals") # Add a centered suptitle to the figure
plt.plot(F, np.abs(Fy)) # Plot Fy vs F (magnitude)
plt.plot(F, np.abs(Fyw)) # Plot Fyw vs F (magnitude)
plt.xlim(0, FNy) # Set the x limits of the current axes
plt.ylim(bottom=0) # Set the y limits of the current axes
plt.xlabel("Frequency / Hz") # Set the label for the x-axis
plt.ylabel("Amplitude / arb. unit") # Set the label for the y-axis
plt.grid(which="major", color="#AAAAAA", linestyle="-", linewidth=0.8) # Configure the major grid lines
plt.grid(which="minor", color="#BBBBBB", linestyle="--", linewidth=0.5) # Configure the minor grid lines
plt.minorticks_on() # Display minor ticks on the axes
plt.show() # Display all open figures