# -*- coding: utf-8 -*-
"""
Part I Autumn Term
Lecture 2: Frequency domain analysis

Fast Fourier Transform.

Last modified on Thu Sep 15 02:04:04 2022
@author: Dr Frederic Cegla, Amanda Lee
"""

###############################################################
# Import required functions, modules, packages, and libraries #
###############################################################
# Functions
from numpy import pi as pi, exp as exp, sin as sin # Import NumPy constants and mathematical functions
from numpy.fft import fft as fft                   # Import NumPy discrete Fourier transform functions

# Modules, packages, and libraries
import numpy as np                                 # Import the fundamental package for scientific computing with Python
import matplotlib.pyplot as plt                    # Import the state-based interface for interactive plots and simple cases of programmatic plot generation

#######################################
# Step 1: Display time domain signals #
#######################################
t = np.linspace(0, 1-(1/512), 512) # Create time vector
y1 = sin(2 * pi * 20 * t) # Create signal y1 (20 Hz)
y2 = sin(2 * pi * 30 * t) # Create signal y2 (30 Hz)
y = y1 + y2 # Create signal y (y1 + y2)

fig = plt.figure("Figure 1: Time domain signals") # Create a new figure
fig.suptitle("Time domain signals") # Add a centered suptitle to the figure

plt.subplot(2, 1, 1) # Add an axes to the current figure (top plot)
plt.plot(t, y1, "b--", label="20 Hz") # Plot y1 vs t
plt.plot(t, y2, "r--", label="30 Hz") # Plot y2 vs t
plt.xlim(0, 1) # Set the x limits of the current axes
plt.ylim(-2, 2) # Set the y limits of the current axes
plt.xlabel("Time / s") # Set the label for the x-axis
plt.ylabel("Amplitude / arb. unit") # Set the label for the y-axis
plt.legend(loc="upper right") # Place a legend on the axes
plt.grid(which="major", color="#AAAAAA", linestyle="-", linewidth=0.8) # Configure the major grid lines
plt.grid(which="minor", color="#BBBBBB", linestyle="--", linewidth=0.5) # Configure the minor grid lines
plt.minorticks_on() # Display minor ticks on the axes

plt.subplot(2, 1, 2) # Add an axes to the current figure (bottom plot)
plt.plot(t, y, "g", label="20 Hz + 30 Hz") # Plot y vs t
plt.xlim(0, 1) # Set the x limits of the current axes
plt.ylim(-2, 2) # Set the y limits of the current axes
plt.xlabel("Time / s") # Set the label for the x-axis
plt.ylabel("Amplitude / arb. unit") # Set the label for the y-axis
plt.legend(loc="upper right") # Place a legend on the axes
plt.grid(which="major", color="#AAAAAA", linestyle="-", linewidth=0.8) # Configure the major grid lines
plt.grid(which="minor", color="#BBBBBB", linestyle="--", linewidth=0.5) # Configure the minor grid lines
plt.minorticks_on() # Display minor ticks on the axes

plt.tight_layout() # Adjust subplot layout

############################################
# Step 2: Display frequency domain signals #
############################################
T = t[-1] # Define time period
Fs = (len(t)-1)/T # Define sampling frequency
FNy = Fs/2 # Define Nyquist frequency
F = np.linspace(0, Fs, len(t)) # Create frequency vector
Fy1 = fft(y1) # Take FFT of signal y1
Fy2 = fft(y2) # Take FFT of signal y2
Fy = fft(y) # Take FFT of signal y

fig = plt.figure("Figure 2: Frequency domain signals") # Create a new figure
fig.suptitle("Frequency domain signals") # Add a centered suptitle to the figure

plt.subplot(2, 1, 1) # Add an axes to the current figure (top plot)
plt.plot(F, np.abs(Fy1), "+-", label="20 Hz") # Plot Fy1 vs F (magnitude)
plt.plot(F, np.abs(Fy2), "+-", label="30 Hz") # Plot Fy2 vs F (magnitude)
plt.plot(F, np.abs(Fy), "+--", label="20 Hz + 30 Hz") # Plot Fy vs F (magnitude)
# plt.plot(F, np.sqrt( np.real(Fy)**2 + np.imag(Fy)**2 ), "+--", label="20 Hz + 30 Hz (alt)") # Alternative command
plt.xlim(0, FNy) # Display up to Nyquist frequency
plt.ylim(bottom=0) # Set the y limits of the current axes
plt.xlabel("Frequency / Hz") # Set the label for the x-axis
plt.ylabel("Amplitude / arb. unit") # Set the label for the y-axis
plt.legend(loc="upper right") # Place a legend on the axes
plt.grid(which="major", color="#AAAAAA", linestyle="-", linewidth=0.8) # Configure the major grid lines
plt.grid(which="minor", color="#BBBBBB", linestyle="--", linewidth=0.5) # Configure the minor grid lines
plt.minorticks_on() # Display minor ticks on the axes

plt.subplot(2, 1, 2) # Add an axes to the current figure (bottom plot)
plt.plot(F, np.angle(Fy1), "+-", label="20 Hz") # Plot Fy1 vs F (phase)
plt.plot(F, np.angle(Fy2), "+-", label="30 Hz") # Plot Fy2 vs F (phase)
plt.plot(F, np.angle(Fy), "+--", label="20 Hz + 30 Hz") # Plot Fy vs F (phase)
# plt.plot(F, np.arctan2( np.imag(Fy), np.real(Fy) ), "+--", label="20 Hz + 30 Hz (alt)") # Alternative command
plt.xlim(0, FNy) # Display up to Nyquist frequency
plt.xlabel("Frequency / Hz") # Set the label for the x-axis
plt.ylabel("Phase / rad") # Set the label for the y-axis
plt.legend(loc="upper right") # Place a legend on the axes
plt.grid(which="major", color="#AAAAAA", linestyle="-", linewidth=0.8) # Configure the major grid lines
plt.grid(which="minor", color="#BBBBBB", linestyle="--", linewidth=0.5) # Configure the minor grid lines
plt.minorticks_on() # Display minor ticks on the axes

plt.tight_layout() # Adjust subplot layout

###########################################
# Step 3: Reconstruct time domain signals #
###########################################
mag_re20 = (exp(1j * 2 * pi * 20 * t) * Fy1[20]) / FNy # Reconstruct time domain signal (20 Hz)
mag_re30 = (exp(1j * 2 * pi * 30 * t) * Fy2[30]) / FNy # Reconstruct time domain signal (30 Hz)

fig = plt.figure("Figure 3: Time domain signals (reconstructed)") # Create a new figure
fig.suptitle("Time domain signals (reconstructed)") # Set a title for the axes

plt.subplot(2, 1, 1) # Add an axes to the current figure (top plot)
plt.plot(t, mag_re20, label="Reconstructed signal (20 Hz)") # Plot reconstructed signal (20 Hz)
plt.plot(t, y1, "--", label="Original signal (20 Hz)") # Plot original signal (20 Hz)
plt.xlim(0, 1) # Set the x limits of the current axes
plt.ylim(-1, 1) # Set the y limits of the current axes
plt.xlabel("Time / s") # Set the label for the x-axis
plt.ylabel("Amplitude / arb. unit") # Set the label for the y-axis
plt.legend(loc="upper right") # Place a legend on the axes

plt.subplot(2, 1, 2) # Add an axes to the current figure (bottom plot)
plt.plot(t, mag_re30, label="Reconstructed signal (30 Hz)") # Plot reconstructed signal (30 Hz)
plt.plot(t, y2, "--", label="Reconstructed signal (30 Hz)") # Plot original signal (30 Hz)
plt.xlim(0, 1) # Set the x limits of the current axes
plt.ylim(-1, 1) # Set the y limits of the current axes
plt.xlabel("Time / s") # Set the label for the x-axis
plt.ylabel("Amplitude / arb. unit") # Set the label for the y-axis
plt.legend(loc="upper right") # Place a legend on the axes

plt.tight_layout() # Adjust subplot layout
plt.show() # Display all open figures