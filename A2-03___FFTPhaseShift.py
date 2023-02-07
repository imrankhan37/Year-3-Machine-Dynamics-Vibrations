# -*- coding: utf-8 -*-
"""
Part I Autumn Term
Lecture 2: Frequency domain analysis

Fast Fourier Transform - phase shift.

Last modified on Thu Sep 15 04:47:13 2022
@author: Dr Frederic Cegla, Amanda Lee
"""

###############################################################
# Import required functions, modules, packages, and libraries #
###############################################################
# Functions
from numpy import pi as pi, cos as cos # Import NumPy constants and mathematical functions
from numpy.fft import fft as fft       # Import NumPy discrete Fourier transform functions

# Modules, packages, and libraries
import numpy as np                     # Import the fundamental package for scientific computing with Python
import matplotlib.pyplot as plt        # Import the state-based interface for interactive plots and simple cases of programmatic plot generation

#######################################
# Step 1: Display time domain signals #
#######################################
t = np.linspace(0, 1-(1/512), 512) # Create time vector
y = cos(2*pi*20*t) # Create signal y (20 Hz)
y1 = cos(2*pi*20*t - pi/8) # Create signal y1 (20 Hz, phi=pi/8)
y2 = cos(2*pi*20*t - pi/4) # Create signal y2 (20 Hz, phi=pi/4)
y3 = cos(2*pi*20*t - 3*pi/8) # Create signal y3 (20 Hz, phi=3pi/8)
y4 = cos(2*pi*20*t - pi/2) # Create signal y4 (20 Hz, phi=pi/2)

plt.figure("Figure 1: Time domain signals") # Create a new figure
plt.title("Time domain signals") # Set a title for the axes
plt.plot(t, y, label="20 Hz, $\phi$ = 0") # Plot y vs t
plt.plot(t, y1, label="20 Hz, $\phi$ = $\pi$/8") # Plot y1 vs t
plt.plot(t, y2, label="20 Hz, $\phi$ = $\pi$/4") # Plot y2 vs t
plt.plot(t, y3, label="20 Hz, $\phi$ = 3$\pi$/8") # Plot y3 vs t
plt.plot(t, y4, label="20 Hz, $\phi$ = $\pi$/2") # Plot y4 vs t
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
Fy = fft(y) # Take FFT of signal y
Fy1 = fft(y1) # Take FFT of signal y1
Fy2 = fft(y2) # Take FFT of signal y2
Fy3 = fft(y3) # Take FFT of signal y3
Fy4 = fft(y4) # Take FFT of signal y4

fig = plt.figure("Figure 2: Frequency domain signals") # Create a new figure
fig.suptitle("Frequency domain signals") # Add a centered suptitle to the figure

plt.subplot(2, 1, 1) # Add an axes to the current figure (top plot)
plt.plot(F, np.abs(Fy), "+-", label="20 Hz, $\phi$ = 0") # Plot y vs t
plt.plot(F, np.abs(Fy1), "+-", label="20 Hz, $\phi$ = $\pi$/8") # Plot y1 vs t
plt.plot(F, np.abs(Fy2), "+-", label="20 Hz, $\phi$ = $\pi$/4") # Plot y2 vs t
plt.plot(F, np.abs(Fy3), "+-", label="20 Hz, $\phi$ = 3$\pi$/8") # Plot y3 vs t
plt.plot(F, np.abs(Fy4), "+-", label="20 Hz, $\phi$ = $\pi$/2") # Plot y4 vs t
plt.xlim(0, FNy) # Display up to Nyquist frequency
plt.ylim(bottom=0) # Set the y limits of the current axes
plt.xlabel("Frequency / Hz") # Set the label for the x-axis
plt.ylabel("Amplitude / arb. unit") # Set the label for the y-axis
plt.legend(loc="upper right") # Place a legend on the axes
plt.grid(which="major", color="#AAAAAA", linestyle="-", linewidth=0.8) # Configure the major grid lines
plt.grid(which="minor", color="#BBBBBB", linestyle="--", linewidth=0.5) # Configure the minor grid lines
plt.minorticks_on() # Display minor ticks on the axes

plt.subplot(2, 1, 2) # Add an axes to the current figure (bottom plot)
plt.plot(F, np.angle(Fy), "+-", label="20 Hz, $\phi$ = 0") # Plot y vs t
plt.plot(F, np.angle(Fy1), "+-", label="20 Hz, $\phi$ = $\pi$/8") # Plot y1 vs t
plt.plot(F, np.angle(Fy2), "+-", label="20 Hz, $\phi$ = $\pi$/4") # Plot y2 vs t
plt.plot(F, np.angle(Fy3), "+-", label="20 Hz, $\phi$ = 3$\pi$/8") # Plot y3 vs t
plt.plot(F, np.angle(Fy4), "+-", label="20 Hz, $\phi$ = $\pi$/2") # Plot y4 vs t
plt.xlim(0, FNy) # Display up to Nyquist frequency
plt.ylim(-pi, pi) # Set the y limits of the current axes
plt.xlabel("Frequency / Hz") # Set the label for the x-axis
plt.ylabel("Phase / rad") # Set the label for the y-axis
plt.legend(loc="upper right") # Place a legend on the axes
plt.grid(which="major", color="#AAAAAA", linestyle="-", linewidth=0.8) # Configure the major grid lines
plt.grid(which="minor", color="#BBBBBB", linestyle="--", linewidth=0.5) # Configure the minor grid lines
plt.minorticks_on() # Display minor ticks on the axes

plt.tight_layout() # Adjust subplots layout

##################################################################
# Step 3: Reconstruct phase shifts and find Fourier coefficients #
##################################################################
phase = [0, pi/8, pi/4, 3*pi/8, pi/2] # Define original phase shifts
phase_re = [np.angle(Fy)[20], np.angle(Fy1)[20], np.angle(Fy2)[20], np.angle(Fy3)[20], np.angle(Fy4)[20]] # Reconstruct phase shifts
a_k = [np.real(Fy[20]), np.real(Fy1[20]), np.real(Fy2[20]), np.real(Fy3[20]), np.real(Fy4[20])] # Find Fourier coefficients a_k
b_k = [np.imag(Fy[20]), np.imag(Fy1[20]), np.imag(Fy2[20]), np.imag(Fy3[20]), np.imag(Fy4[20])] # Find Fourier coefficients b_k

fig = plt.figure("Figure 3: Phase shifts and Fourier coefficients") # Create a new figure
fig.suptitle("Phase shifts and Fourier coefficients") # Add a centered suptitle to the figure

plt.subplot(2, 1, 1) # Add an axes to the current figure (top plot)
plt.plot(phase_re, "-", label="Reconstructed phase shifts") # Plot reconstructed phase shifts
plt.plot(np.negative(phase), "+", label="Original phase shifts") # Plot original phase shifts
plt.xlim(0, 4) # Set the x limits of the current axes
plt.ylim(-pi/2, 0) # Set the y limits of the current axes
plt.xticks(np.arange(1, 5, 1))  # Set the current tick locations and labels of the x-axis
plt.xlabel("Curve number") # Set the label for the x-axis
plt.ylabel("Phase / rad") # Set the label for the y-axis
plt.legend(loc="upper right") # Place a legend on the axes
plt.grid() # Configure the grid lines

plt.subplot(2, 1, 2) # Add an axes to the current figure (bottom plot)
plt.plot(phase, a_k, "+", label="$a_{k}$") # Plot a_k vs phase
plt.plot(phase, b_k, "+", label="$b_{k}$") # Plot b_k vs phase
plt.xlim(0, pi/2) # Set the x limits of the current axes
plt.xlabel("Phase / rad") # Set the label for the x-axis
plt.ylabel("Fourier coefficient") # Set the label for the y-axis
plt.legend(loc="upper right") # Place a legend on the axes
plt.grid() # Configure the grid lines

plt.tight_layout() # Adjust subplots layout
plt.show() # Display all open figures