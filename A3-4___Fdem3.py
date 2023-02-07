# -*- coding: utf-8 -*-
"""
Part I Autumn Term
Lecture 3: Practical application of time domain and fourier domain processing

Aliasing.

Last modified on Thu Sep 15 05:22:56 2022
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

##########
# Step 1 #
##########
t = np.linspace(0, 1, 201) # Create time vector
y = sin(2 * pi * 20 * t) # Create signal y1 (50 Hz)

plt.figure("Figure 1: Time domain signals") # Create a new figure
plt.title("Time domain signals") # Set a title for the axes
plt.plot(t, y) # Plot y vs t
plt.xlim(0, 1)
plt.ylim(-1, 1)
plt.xlabel("Time / s") # Set the label for the x-axis
plt.ylabel("Amplitude / arb. unit") # Set the label for the y-axis
plt.show() # Display all open figures

##########
# Step 2 #
##########
T = t[-1] # Define time period
Fs = (len(t)-1)/T # Define sampling frequency
FNy = Fs/2 # Define Nyquist frequency
F = np.linspace(0, Fs, len(t)) # Create frequency vector
Fy = np.zeros(len(F), dtype="complex_")
Fy[:-1] = fft(y[:-1]) # Take FFT of signal y
Fy[-1] = Fy[0]

plt.figure("Figure 2: Frequency domain signals") # Create a new figure
plt.title("Frequency domain signals") # Add a centered suptitle to the figure
plt.plot(F, np.abs(Fy)/110) # Plot Fy vs F (magnitude)
plt.axvline(x=FNy, color="k", linestyle="--") # Display axis of symmetry at Nyquist frequency
plt.xlim(F[0], F[-1])
plt.ylim(bottom=0) # Set the y limits of the current axes
plt.xlabel("Frequency / Hz") # Set the label for the x-axis
plt.ylabel("Amplitude / arb. unit") # Set the label for the y-axis
plt.grid(which="major", color="#AAAAAA", linestyle="-", linewidth=0.8) # Configure the major grid lines
plt.grid(which="minor", color="#BBBBBB", linestyle="--", linewidth=0.5) # Configure the minor grid lines
plt.minorticks_on() # Display minor ticks on the axes
plt.show() # Display all open figures

##########
# Step 3 #
##########
t = np.linspace(0, 1, 61) # Create time vector
y = sin(2 * pi * 20 * t) # Create signal y1 (50 Hz)

plt.figure("Figure 3: Time domain signals") # Create a new figure
plt.title("Time domain signals") # Set a title for the axes
plt.plot(t, y) # Plot y vs t
plt.xlim(0, 1)
plt.ylim(-1, 1)
plt.xlabel("Time / s") # Set the label for the x-axis
plt.ylabel("Amplitude / arb. unit") # Set the label for the y-axis
plt.show() # Display all open figures

##########
# Step 4 #
##########
T = t[-1] # Define time period
Fs = (len(t)-1)/T # Define sampling frequency
FNy = Fs/2 # Define Nyquist frequency
F = np.linspace(0, Fs, len(t)) # Create frequency vector
Fy = np.zeros(len(F), dtype="complex_")
Fy[:-1] = fft(y[:-1]) # Take FFT of signal y
Fy[-1] = Fy[0]

plt.figure("Figure 4: Frequency domain signals") # Create a new figure
plt.title("Frequency domain signals") # Add a centered suptitle to the figure
plt.plot(F, np.abs(Fy)/110) # Plot Fy vs F (magnitude)
plt.axvline(x=FNy, color="k", linestyle="--") # Display axis of symmetry at Nyquist frequency
plt.xlim(F[0], F[-1])
plt.ylim(bottom=0) # Set the y limits of the current axes
plt.xlabel("Frequency / Hz") # Set the label for the x-axis
plt.ylabel("Amplitude / arb. unit") # Set the label for the y-axis
plt.grid(which="major", color="#AAAAAA", linestyle="-", linewidth=0.8) # Configure the major grid lines
plt.grid(which="minor", color="#BBBBBB", linestyle="--", linewidth=0.5) # Configure the minor grid lines
plt.minorticks_on() # Display minor ticks on the axes
plt.show() # Display all open figures

##########
# Step 5 #
##########
t = np.linspace(0, 1, 36) # Create time vector
y = sin(2 * pi * 20 * t) # Create signal y1 (50 Hz)

plt.figure("Figure 5: Time domain signals") # Create a new figure
plt.title("Time domain signals") # Set a title for the axes
plt.plot(t, y) # Plot y vs t
plt.xlim(0, 1)
plt.ylim(-1, 1)
plt.xlabel("Time / s") # Set the label for the x-axis
plt.ylabel("Amplitude / arb. unit") # Set the label for the y-axis
plt.show() # Display all open figures

##########
# Step 6 #
##########
T = t[-1] # Define time period
Fs = (len(t)-1)/T # Define sampling frequency
FNy = Fs/2 # Define Nyquist frequency
F = np.linspace(0, Fs, len(t)) # Create frequency vector
Fy = np.zeros(len(F), dtype="complex_")
Fy[:-1] = fft(y[:-1]) # Take FFT of signal y
Fy[-1] = Fy[0]

plt.figure("Figure 6: Frequency domain signals") # Create a new figure
plt.title("Frequency domain signals") # Add a centered suptitle to the figure
plt.plot(F, np.abs(Fy)/110) # Plot Fy vs F (magnitude)
plt.axvline(x=FNy, color="k", linestyle="--") # Display axis of symmetry at Nyquist frequency
plt.xlim(F[0], F[-1])
plt.ylim(bottom=0) # Set the y limits of the current axes
plt.xlabel("Frequency / Hz") # Set the label for the x-axis
plt.ylabel("Amplitude / arb. unit") # Set the label for the y-axis
plt.grid(which="major", color="#AAAAAA", linestyle="-", linewidth=0.8) # Configure the major grid lines
plt.grid(which="minor", color="#BBBBBB", linestyle="--", linewidth=0.5) # Configure the minor grid lines
plt.minorticks_on() # Display minor ticks on the axes
plt.show() # Display all open figures