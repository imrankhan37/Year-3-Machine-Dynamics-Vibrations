# -*- coding: utf-8 -*-
"""
Part I Autumn Term
Lecture 2: Frequency domain analysis

Fast Fourier Transform - low pass filter.

Last modified on Thu Sep 15 05:01:54 2022
@author: Dr Frederic Cegla, Amanda Lee
"""

###############################################################
# Import required functions, modules, packages, and libraries #
###############################################################
# Functions
from numpy import pi as pi, sin as sin         # Import NumPy constants and mathematical functions
from numpy.fft import fft as fft, ifft as ifft # Import NumPy discrete Fourier transform functions

# Modules, packages, and libraries
import numpy as np                             # Import the fundamental package for scientific computing with Python
import matplotlib.pyplot as plt                # Import the state-based interface for interactive plots and simple cases of programmatic plot generation

######################################
# Step 1: Display time domain signal #
######################################
t = np.linspace(0, 0.511, 512) # Create time vector
y1 = sin(2*pi*30*t) # Create signal y1
y2 = sin(2*pi*200*t) # Create signal y2
y = y1 + 0.2*y2 # Create signal y (y1 + y2, weighted)

plt.figure("Figure 1: Time domain signals") # Create a new figure
plt.title("Time domain signals") # Set a title for the axes
plt.plot(t, y) # Plot yw vs t
plt.xlim(0, 0.5) # Set the x limits of the current axes
plt.xlabel("Time / s") # Set the label for the x-axis
plt.ylabel("Amplitude / arb. unit") # Set the label for the y-axis

###########################################
# Step 2: Display frequency domain signal #
###########################################
T = t[-1] # Define time period
Fs = (len(t)-1)/T # Define sampling frequency
FNy = Fs/2 # Define Nyquist frequency
F = np.linspace(0, Fs, len(t)) # Create frequency vector
Fy = np.zeros(len(F), dtype="complex_") # Initiliase signal Fy
Fy = fft(y) # Take FFT of signal y

plt.figure("Figure 2: Frequency domain signals") # Create a new figure
plt.title("Frequency domain signals") # Add a centered suptitle to the figure
plt.semilogy(F, np.abs(Fy)) # Plot Fy vs F (magnitude)
plt.xlim(0, FNy) # Set the x limits of the current axes
plt.xlabel("Frequency / Hz") # Set the label for the x-axis
plt.ylabel("Amplitude / arb. unit") # Set the label for the y-axis
plt.grid(which="major", color="#AAAAAA", linestyle="-", linewidth=0.8) # Configure the major grid lines
plt.grid(which="minor", color="#BBBBBB", linestyle="--", linewidth=0.5) # Configure the minor grid lines
plt.minorticks_on() # Display minor ticks on the axes

######################################################
# Step 3: Display time domain signal (reconstructed) #
######################################################
y_new = np.real( ifft(Fy) ) # Take IFFT of signal Fy

plt.figure("Figure 3: Time domain signals") # Create a new figure
plt.title("Time domain signals") # Set a title for the axes
plt.plot(t, y_new) # Plot yw vs t
plt.xlim(0, 0.5) # Set the x limits of the current axes
plt.xlabel("Time / s") # Set the label for the x-axis
plt.ylabel("Amplitude / arb. unit") # Set the label for the y-axis

######################################################
# Step 4: Display frequency domain signal (filtered) #
######################################################
Fy[49:-49] = 10e-11 # Filter out selected high frequencies

plt.figure("Figure 4: Frequency domain signals") # Create a new figure
plt.title("Frequency domain signals") # Add a centered suptitle to the figure
plt.semilogy(F, np.abs(Fy)) # Plot Fy vs F (magnitude)
plt.xlim(0, FNy) # Set the x limits of the current axes
plt.xlabel("Frequency / Hz") # Set the label for the x-axis
plt.ylabel("Amplitude / arb. unit") # Set the label for the y-axis
plt.grid(which="major", color="#AAAAAA", linestyle="-", linewidth=0.8) # Configure the major grid lines
plt.grid(which="minor", color="#BBBBBB", linestyle="--", linewidth=0.5) # Configure the minor grid lines
plt.minorticks_on() # Display minor ticks on the axes

#################################################
# Step 5: Display time domain signal (filtered) #
#################################################
y_new = np.real( ifft(Fy) ) # Take IFFT of signal Fy

plt.figure("Figure 5: Time domain signals") # Create a new figure
plt.title("Time domain signals") # Set a title for the axes
plt.plot(t, y_new) # Plot yw vs t
plt.xlim(0, 0.5) # Set the x limits of the current axes
plt.xlabel("Time / s") # Set the label for the x-axis
plt.ylabel("Amplitude / arb. unit") # Set the label for the y-axis
plt.show() # Display all open figures