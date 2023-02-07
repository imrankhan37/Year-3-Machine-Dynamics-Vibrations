# -*- coding: utf-8 -*-
"""
Part I Autumn Term
Lecture 2: Frequency domain analysis

Fast Fourier Transform - spectral leakage.

Last modified on Thu Sep 15 04:57:34 2022
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

######################################
# Step 1: Display time domain signal #
######################################
t = np.linspace(0, 1-(1/512), 512) # Create time vector
y1 = sin(2 * pi * 10 * t) # Create signal y (20 Hz + 25 Hz)
y2 = sin(2 * pi * 15.5 * t) # Create signal x (20 Hz + 30 Hz)

plt.figure("Figure 1: Time domain signal") # Create a new figure
plt.title("Time domain signal") # Set a title for the axes
plt.plot(t, y1, label="10 Hz") # Plot y1 vs t
plt.plot(t, y2, label="15.5 Hz") # Plot y2 vs t
plt.xlim(0, 1) # Set the x limits of the current axes
plt.ylim(-1, 1) # Set the y limits of the current axes
plt.xlabel("Time / s") # Set the label for the x-axis
plt.ylabel("Amplitude / arb. unit") # Set the label for the y-axis
plt.legend() # Place a legend on the axes

###########################################
# Step 2: Display frequency domain signal #
###########################################
T = t[-1] # Define time period
Fs = (len(t)-1)/T # Define sampling frequency
F = np.linspace(0, Fs, len(t)) # Create frequency vector
Fy1 = fft(y1) # Take FFT of signal y1
Fy2 = fft(y2) # Take FFT of signal y2

fig = plt.figure("Figure 2: Frequency domain signal") # Create a new figure
fig.suptitle("Frequency domain signal") # Add a centered suptitle to the figure

plt.subplot(2, 1, 1) # Add an axes to the current figure (top plot)
plt.plot(F, np.abs(Fy1), label="10 Hz") # Plot Fy1 vs F (magnitude)
plt.plot(F, np.abs(Fy2), label="15.5 Hz") # Plot Fy2 vs F (magnitude)
plt.xlim(0, 50) # Set the x-limits of the current axes
plt.ylim(bottom=0) # Set the y limits of the current axes
_, top = plt.ylim()  # Return the current y-limits
plt.xlabel("Frequency / Hz") # Set the label for the x-axis
plt.ylabel("Amplitude / arb. unit") # Set the label for the y-axis
plt.legend(loc="upper right") # Place a legend on the axes

plt.subplot(2, 1, 2) # Add an axes to the current figure (bottom plot)
plt.plot(F, np.angle(Fy1), label="10 Hz") # Plot Fy1 vs F (phase)
plt.plot(F, np.angle(Fy2), label="15.5 Hz") # Plot Fy2 vs F (phase)
plt.xlim(0, 50) # Set the x-limits of the current axes
plt.ylim(-pi, pi) # Set the y limits of the current axes
plt.xlabel("Frequency / Hz") # Set the label for the x-axis
plt.ylabel("Phase / rad") # Set the label for the y-axis
plt.legend(loc="upper right") # Place a legend on the axes

plt.tight_layout() # Adjust subplots layout

################################
# Step 3: Apply Hanning window #
################################
Fy1_new = fft( np.multiply( np.hanning(len(y1)), y1 ) ) # Take FFT of signal y1 (after applying Hanning window)
Fy2_new = fft( np.multiply( np.hanning(len(y2)), y2 ) ) # Take FFT of signal y2 (after applying Hanning window)

fig = plt.figure("Figure 3: Frequency domain signal (after applying Hanning window)") # Create a new figure
fig.suptitle("Frequency domain signal (after applying Hanning window)") # Add a centered suptitle to the figure

plt.subplot(2, 1, 1) # Add an axes to the current figure (top plot)
plt.plot(F, np.abs(Fy1_new), label="10 Hz") # Plot Fy1_new vs F (magnitude)
plt.plot(F, np.abs(Fy2_new), label="15.5 Hz") # Plot Fy2_new vs F (magnitude)
plt.xlim(0, 50) # Set the x-limits of the current axes
plt.ylim(0, top) # Set the y-limits of the current axes
plt.xlabel("Frequency / Hz") # Set the label for the x-axis
plt.ylabel("Amplitude / arb. unit") # Set the label for the y-axis
plt.legend(loc="upper right") # Place a legend on the axes

plt.subplot(2, 1, 2) # Add an axes to the current figure (bottom plot)
plt.plot(F, np.angle(Fy1_new), label="10 Hz") # Plot Fy1_new vs F (phase)
plt.plot(F, np.angle(Fy2_new), label="15.5 Hz") # Plot Fy2_new vs F (phase)
plt.xlim(0, 50) # Set the x-limits of the current axes
plt.ylim(-pi, pi) # Set the y limits of the current axes
plt.xlabel("Frequency / Hz") # Set the label for the x-axis
plt.ylabel("Phase / rad") # Set the label for the y-axis
plt.legend(loc="upper right") # Place a legend on the axes

plt.tight_layout() # Adjust subplots layout
plt.show() # Display all open figures