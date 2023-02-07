# -*- coding: utf-8 -*-
"""
Part I Autumn Term
Lecture 2: Frequency domain analysis

Fast Fourier Transform - aliasing.

Last modified on Thu Sep 15 04:48:11 2022
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

#######################################
# Step 1: Display time domain signals #
#######################################
t = np.linspace(0, 1-(1/512), 512) # Create time vector

# Maximum frequency in the spectrum is Fmax = 1/dt = 512 Hz
# However, maximum frequency that can be resolved is the Nyquist frequency FNy = Fmax/2 = 256 Hz
# Thus, define one signal that is resolved, e.g. 50 Hz and one signal that is not resolved, e.g. 400 Hz
# that are periodic in the time window
y1 = sin(2 * pi * 50 * t) # Create signal y1 (50 Hz)
y2 = sin(2 * pi * 400 * t) # Create signal y2 (400 Hz)

plt.figure("Figure 1: Time domain signals") # Create a new figure
plt.title("Time domain signals") # Set a title for the axes
plt.plot(t, y1, label="50 Hz") # Plot y1 vs t
plt.plot(t, y2, label="400 Hz") # Plot y2 vs t
plt.xlim(0, 1) # Set the x limits of the current axes
plt.ylim(-1, 1) # Set the y limits of the current axes
plt.xlabel("Time / s") # Set the label for the x-axis
plt.ylabel("Amplitude / arb. unit") # Set the label for the y-axis
plt.legend() # Place a legend on the axes

############################################
# Step 2: Display frequency domain signals #
############################################
T = t[-1] # Define time period
Fs = (len(t)-1)/T # Define sampling frequency
FNy = Fs/2 # Define Nyquist frequency
F = np.linspace(0, Fs, len(t)) # Create frequency vector
Fy1 = fft(y1) # Take FFT of signal y1
Fy2 = fft(y2) # Take FFT of signal y2

fig = plt.figure("Figure 2: Frequency domain signals") # Create a new figure
fig.suptitle("Frequency domain signals") # Add a centered suptitle to the figure

plt.subplot(2, 1, 1) # Add an axes to the current figure (top plot)
plt.plot(F, np.abs(Fy1), label="50 Hz") # Plot Fy1 vs F (magnitude)
plt.plot(F, np.abs(Fy2), label="400 Hz") # Plot Fy2 vs F (magnitude)
plt.axvline(x=FNy, color="k", linestyle="--") # Display axis of symmetry at Nyquist frequency
plt.xlim(F[0], F[-1]) # Set the x limits of the current axes
plt.ylim(bottom=0) # Set the y limits of the current axes
plt.xlabel("Frequency / Hz") # Set the label for the x-axis
plt.ylabel("Amplitude / arb. unit") # Set the label for the y-axis
plt.legend(loc="upper right") # Place a legend on the axes
plt.grid(which="major", color="#AAAAAA", linestyle="-", linewidth=0.8) # Configure the major grid lines
plt.grid(which="minor", color="#BBBBBB", linestyle="--", linewidth=0.5) # Configure the minor grid lines
plt.minorticks_on() # Display minor ticks on the axes

plt.subplot(2, 1, 2) # Add an axes to the current figure (bottom plot)
plt.plot(F, np.angle(Fy1), label="20 Hz") # Plot Fy1 vs F (phase)
plt.plot(F, np.angle(Fy2), label="30 Hz") # Plot Fy2 vs F (phase)
plt.xlim(0, FNy) # Display up to Nyquist frequency
plt.ylim(-pi, pi) # Set the y limits of the current axes
plt.xlabel("Frequency / Hz") # Set the label for the x-axis
plt.ylabel("Phase / rad") # Set the label for the y-axis
plt.legend(loc="upper right") # Place a legend on the axes
plt.grid(which="major", color="#AAAAAA", linestyle="-", linewidth=0.8) # Configure the major grid lines
plt.grid(which="minor", color="#BBBBBB", linestyle="--", linewidth=0.5) # Configure the minor grid lines
plt.minorticks_on() # Display minor ticks on the axes

plt.tight_layout() # Adjust subplots layout
plt.show() # Display all open figures