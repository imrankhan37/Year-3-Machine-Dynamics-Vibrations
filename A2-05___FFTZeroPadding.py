# -*- coding: utf-8 -*-
"""
Part I Autumn Term
Lecture 2: Frequency domain analysis

Fast Fourier Transform - zero padding.

Last modified on Thu Sep 15 04:55:11 2022
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

############################################################
# Step 1: Signal Processing for short sample (time domain) #
############################################################
t = np.linspace(0, 0.05-(0.05/512), 512) # Create time vector
y1 = sin(2 * pi * 550 * t) # Create signal y1 (apply Hanning window to 550 Hz signal)
y2 = sin(2 * pi * 560 * t) # Create signal y2 (apply Hanning window to 560 Hz signal)
y = y1 + y2 # Create signal y (y1 + y2)

fig = plt.figure("Figure 1: Short sample") # Create a new figure
fig.suptitle("Short sample", fontweight="bold") # Set a title for the axes

plt.subplot(2, 1, 1) # Add an axes to the current figure (top plot)
plt.title("Time domain") # Set a title for the axes
plt.plot(t, y, label="550 Hz + 560 Hz") # Plot y vs t
plt.xlim(0, 0.05) # Set the x limits of the current axes
plt.ylim(-2, 2) # Set the y limits of the current axes
plt.xlabel("Time / s") # Set the label for the x-axis
plt.ylabel("Amplitude / arb. unit") # Set the label for the y-axis
plt.legend(loc="upper right") # Place a legend on the axes
plt.grid(which="major", color="#AAAAAA", linestyle="-", linewidth=0.8) # Configure the major grid lines
plt.grid(which="minor", color="#BBBBBB", linestyle="--", linewidth=0.5) # Configure the minor grid lines
plt.minorticks_on() # Display minor ticks on the axes

#################################################################
# Step 2: Signal Processing for short sample (frequency domain) #
#################################################################
T = t[-1] # Define time period
Fs = (len(t)-1)/T # Define sampling frequency
F = np.linspace(0, Fs, len(t)) # Create frequency vector
Fy = fft(y) # Take FFT of signal y

plt.subplot(2, 1, 2) # Add an axes to the current figure (bottom plot)
plt.title("Frequency domain") # Set a title for the axes
plt.plot(F, np.abs(Fy), "+-", label="550 Hz + 560 Hz") # Plot Fy vs F (magnitude)
plt.xlim(400, 700) # Display up to Nyquist frequency
plt.ylim(bottom=0) # Set the y limits of the current axes
plt.xlabel("Frequency / Hz") # Set the label for the x-axis
plt.ylabel("Amplitude / arb. unit") # Set the label for the y-axis
plt.legend(loc="upper right") # Place a legend on the axes
plt.grid(which="major", color="#AAAAAA", linestyle="-", linewidth=0.8) # Configure the major grid lines
plt.grid(which="minor", color="#BBBBBB", linestyle="--", linewidth=0.5) # Configure the minor grid lines
plt.minorticks_on() # Display minor ticks on the axes

plt.tight_layout() # Adjust subplots layout

###########################################################
# Step 3: Signal processing for long sample (time domain) #
###########################################################
t = np.linspace(0, 0.4-(0.4/4096), 4096) # Create time vector

y1 = sin(2 * pi * 550 * t) # Create signal y1 (apply Hanning window to 550 Hz signal)
y2 = sin(2 * pi * 560 * t) # Create signal y2 (apply Hanning window to 560 Hz signal)
y = y1 + y2 # Create signal y (y1 + y2)

fig = plt.figure("Figure 2: Long sample") # Create a new figure
fig.suptitle("Long sample", fontweight="bold") # Set a title for the axes

plt.subplot(2, 1, 1) # Add an axes to the current figure (top plot)
plt.title("Time domain") # Set a title for the axes
plt.plot(t, y, label="550 Hz + 560 Hz") # Plot y vs t
plt.xlim(0, 0.4) # Set the x limits of the current axes
plt.ylim(-2, 2) # Set the y limits of the current axes
plt.xlabel("Time / s") # Set the label for the x-axis
plt.ylabel("Amplitude / arb. unit") # Set the label for the y-axis
plt.legend(loc="upper right") # Place a legend on the axes
plt.grid(which="major", color="#AAAAAA", linestyle="-", linewidth=0.8) # Configure the major grid lines
plt.grid(which="minor", color="#BBBBBB", linestyle="--", linewidth=0.5) # Configure the minor grid lines
plt.minorticks_on() # Display minor ticks on the axes

################################################################
# Step 4: Signal Processing for long sample (frequency domain) #
################################################################
T = t[-1] # Define time period
Fs = (len(t)-1)/T # Define sampling frequency
F = np.linspace(0, Fs, len(t)) # Create frequency vector
Fy = fft(y) # Take FFT of signal y

plt.subplot(2, 1, 2) # Add an axes to the current figure (bottom plot)
plt.title("Frequency domain") # Set a title for the axes
plt.plot(F, np.abs(Fy), "+-", label="550 Hz + 560 Hz") # Plot Fy vs F (magnitude)
plt.xlim(400, 700) # Display up to Nyquist frequency
plt.ylim(bottom=0) # Set the y limits of the current axes
plt.xlabel("Frequency / Hz") # Set the label for the x-axis
plt.ylabel("Amplitude / arb. unit") # Set the label for the y-axis
plt.legend(loc="upper right") # Place a legend on the axes
plt.grid(which="major", color="#AAAAAA", linestyle="-", linewidth=0.8) # Configure the major grid lines
plt.grid(which="minor", color="#BBBBBB", linestyle="--", linewidth=0.5) # Configure the minor grid lines
plt.minorticks_on() # Display minor ticks on the axes

plt.tight_layout() # Adjust subplots layout

#######################################################
# Step 5: Zero padding for short sample (time domain) #
#######################################################
t = np.linspace(0, 0.4-(0.4/4096), 4096) # Create time vector

y1 = sin(2 * pi * 550 * t) # Create signal y1 (apply Hanning window to 550 Hz signal)
y2 = sin(2 * pi * 560 * t) # Create signal y2 (apply Hanning window to 560 Hz signal)
y = np.zeros(len(t)) # Initialise signal y with zeros
y[:512] = y1[:512] + y2[:512] # Create signal y (y1 + y2) with zero padding

fig = plt.figure("Figure 3: Zero-padded sample") # Create a new figure
fig.suptitle("Zero-padded sample", fontweight="bold") # Set a title for the axes

plt.subplot(2, 1, 1) # Add an axes to the current figure (top plot)
plt.title("Time domain") # Set a title for the axes
plt.plot(t, y, label="550 Hz + 560 Hz") # Plot y vs t
plt.xlim(0, 0.4) # Set the x limits of the current axes
plt.ylim(-2, 2) # Set the y limits of the current axes
plt.xlabel("Time / s") # Set the label for the x-axis
plt.ylabel("Amplitude / arb. unit") # Set the label for the y-axis
plt.legend(loc="upper right") # Place a legend on the axes
plt.grid(which="major", color="#AAAAAA", linestyle="-", linewidth=0.8) # Configure the major grid lines
plt.grid(which="minor", color="#BBBBBB", linestyle="--", linewidth=0.5) # Configure the minor grid lines
plt.minorticks_on() # Display minor ticks on the axes

############################################################
# Step 6: Zero padding for short sample (frequency domain) #
############################################################
T = t[-1] # Define time period
Fs = (len(t)-1)/T # Define sampling frequency
F = np.linspace(0, Fs, len(t)) # Create frequency vector
Fy = fft(y) # Take FFT of signal y

plt.subplot(2, 1, 2) # Add an axes to the current figure (bottom plot)
plt.title("Frequency domain") # Set a title for the axes
plt.plot(F, np.abs(Fy), "+-", label="550 Hz + 560 Hz") # Plot Fy vs F (magnitude)
plt.xlim(400, 700) # Display up to Nyquist frequency
plt.ylim(bottom=0) # Set the y limits of the current axes
plt.xlabel("Frequency / Hz") # Set the label for the x-axis
plt.ylabel("Amplitude / arb. unit") # Set the label for the y-axis
plt.legend(loc="upper right") # Place a legend on the axes
plt.grid(which="major", color="#AAAAAA", linestyle="-", linewidth=0.8) # Configure the major grid lines
plt.grid(which="minor", color="#BBBBBB", linestyle="--", linewidth=0.5) # Configure the minor grid lines
plt.minorticks_on() # Display minor ticks on the axes

plt.tight_layout() # Adjust subplots layout
plt.show() # Display all open figures