# -*- coding: utf-8 -*-
"""
Part I Autumn Term
Lecture 3: Practical application of time domain and fourier domain processing

Windowing.

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
t = np.linspace(0, 0.04, 221) # Create time vector
y = 5.7 * sin(2*pi*550*t)

fig = plt.figure("Figure 1: Windowing of periodic signal") # Create a new figure
fig.suptitle("Windowing of periodic signal", fontweight="bold") # Set a title for the axes

plt.subplot(2, 2, 1) # Add an axes to the current figure (top left plot)
plt.title("Time domain (original)")
plt.plot(t, y, label="550 Hz, 22 cycles") # Plot y vs t
plt.xlim(0, 0.04)
plt.ylim(-6, 6)
plt.xlabel("Time / s") # Set the label for the x-axis
plt.ylabel("Amplitude / arb. unit") # Set the label for the y-axis
plt.legend(loc="upper right") # Place a legend on the axes
plt.grid(which="major", color="#AAAAAA", linestyle="-", linewidth=0.8) # Configure the major grid lines
plt.grid(which="minor", color="#BBBBBB", linestyle="--", linewidth=0.5) # Configure the minor grid lines
plt.minorticks_on() # Display minor ticks on the axes

##########
# Step 2 #
##########
T = t[-1] # Define time period
Fs = (len(t)-1)/T # Define sampling frequency
F = np.linspace(0, Fs, len(t)) # Create frequency vector
Fy = np.zeros(len(F), dtype="complex_")
Fy[:220] = fft(y[:220]) # Take FFT of signal y
Fy[220] = Fy[0]

plt.subplot(2, 2, 2) # Add an axes to the current figure (top right plot)
plt.title("Frequency domain (original)")
plt.plot(F, np.abs(Fy)/110, "+-", label="550 Hz, 22 cycles") # Plot Fy vs F (magnitude)
plt.xlim(0, 1000) # Display up to Nyquist frequency
plt.ylim(0, 6)
plt.xlabel("Frequency / Hz") # Set the label for the x-axis
plt.ylabel("Amplitude / arb. unit") # Set the label for the y-axis
plt.legend(loc="upper right") # Place a legend on the axes
plt.grid(which="major", color="#AAAAAA", linestyle="-", linewidth=0.8) # Configure the major grid lines
plt.grid(which="minor", color="#BBBBBB", linestyle="--", linewidth=0.5) # Configure the minor grid lines
plt.minorticks_on() # Display minor ticks on the axes

##########
# Step 3 #
##########
yw = np.multiply( y, np.hanning(len(y)) ) # Create signal y (apply Hanning window to 20 Hz signal)

plt.subplot(2, 2, 3) # Add an axes to the current figure (top left plot)
plt.title("Time domain (windowed)")
plt.plot(t, yw, label="550 Hz, 22 cycles") # Plot y vs t
plt.xlim(0, 0.04)
plt.ylim(-6, 6)
plt.xlabel("Time / s") # Set the label for the x-axis
plt.ylabel("Amplitude / arb. unit") # Set the label for the y-axis
plt.legend(loc="upper right") # Place a legend on the axes
plt.grid(which="major", color="#AAAAAA", linestyle="-", linewidth=0.8) # Configure the major grid lines
plt.grid(which="minor", color="#BBBBBB", linestyle="--", linewidth=0.5) # Configure the minor grid lines
plt.minorticks_on() # Display minor ticks on the axes

##########
# Step 4 #
##########
Fyw = np.zeros(len(F), dtype="complex_")
Fyw[:220] = fft(yw[:220]) # Take FFT of signal y
Fyw[220] = Fyw[0]

plt.subplot(2, 2, 4) # Add an axes to the current figure (top right plot)
plt.title("Frequency domain (windowed)")
plt.plot(F, np.abs(Fyw)/110, "+-", label="550 Hz, 22 cycles") # Plot Fy vs F (magnitude)
plt.xlim(0, 1000) # Display up to Nyquist frequency
plt.ylim(0, 6)
plt.xlabel("Frequency / Hz") # Set the label for the x-axis
plt.ylabel("Amplitude / arb. unit") # Set the label for the y-axis
plt.legend(loc="upper right") # Place a legend on the axes
plt.grid(which="major", color="#AAAAAA", linestyle="-", linewidth=0.8) # Configure the major grid lines
plt.grid(which="minor", color="#BBBBBB", linestyle="--", linewidth=0.5) # Configure the minor grid lines
plt.minorticks_on() # Display minor ticks on the axes

plt.subplots_adjust(wspace=0.5, hspace=0.5)
plt.show() # Display all open figures

##########
# Step 5 #
##########
t = np.linspace(0, 0.04, 221) # Create time vector
y = 5.7 * sin(2*pi*562.5*t)

fig = plt.figure("Figure 2: Windowing of non-periodic signal") # Create a new figure
fig.suptitle("Windowing of non-periodic signal", fontweight="bold") # Set a title for the axes

plt.subplot(2, 2, 1) # Add an axes to the current figure (top left pltot)
plt.title("Time domain (original)")
plt.plot(t, y, label="562.5 Hz, 22.5 cycles") # pltot y vs t
plt.xlim(0, 0.04)
plt.ylim(-6, 6)
plt.xlabel("Time / s") # Set the label for the x-axis
plt.ylabel("Ampltitude / arb. unit") # Set the label for the y-axis
plt.legend(loc="upper right") # pltace a legend on the axes
plt.grid(which="major", color="#AAAAAA", linestyle="-", linewidth=0.8) # Configure the major grid lines
plt.grid(which="minor", color="#BBBBBB", linestyle="--", linewidth=0.5) # Configure the minor grid lines
plt.minorticks_on() # Displtay minor ticks on the axes

##########
# Step 6 #
##########
T = t[-1] # Define time period
Fs = (len(t)-1)/T # Define samplting frequency
F = np.linspace(0, Fs, len(t)) # Create frequency vector
Fy = np.zeros(len(F), dtype="complex_")
Fy[:220] = fft(y[:220]) # Take FFT of signal y
Fy[220] = Fy[0]

plt.subplot(2, 2, 2) # Add an axes to the current figure (top right pltot)
plt.title("Frequency domain (original)")
plt.plot(F, np.abs(Fy)/110, "+-", label="562.5 Hz, 22.5 cycles") # pltot Fy vs F (magnitude)
plt.xlim(0, 1000) # Displtay up to Nyquist frequency
plt.ylim(0, 6)
plt.xlabel("Frequency / Hz") # Set the label for the x-axis
plt.ylabel("Ampltitude / arb. unit") # Set the label for the y-axis
plt.legend(loc="upper right") # pltace a legend on the axes
plt.grid(which="major", color="#AAAAAA", linestyle="-", linewidth=0.8) # Configure the major grid lines
plt.grid(which="minor", color="#BBBBBB", linestyle="--", linewidth=0.5) # Configure the minor grid lines
plt.minorticks_on() # Displtay minor ticks on the axes

##########
# Step 7 #
##########
yw = np.multiply( y, np.hanning(len(y)) ) # Create signal y (applty Hanning window to 20 Hz signal)

plt.subplot(2, 2, 3) # Add an axes to the current figure (top left pltot)
plt.title("Time domain (windowed)")
plt.plot(t, yw, label="562.5 Hz, 22.5 cycles") # pltot y vs t
plt.xlim(0, 0.04)
plt.ylim(-6, 6)
plt.xlabel("Time / s") # Set the label for the x-axis
plt.ylabel("Ampltitude / arb. unit") # Set the label for the y-axis
plt.legend(loc="upper right") # pltace a legend on the axes
plt.grid(which="major", color="#AAAAAA", linestyle="-", linewidth=0.8) # Configure the major grid lines
plt.grid(which="minor", color="#BBBBBB", linestyle="--", linewidth=0.5) # Configure the minor grid lines
plt.minorticks_on() # Displtay minor ticks on the axes

##########
# Step 8 #
##########
Fyw = np.zeros(len(F), dtype="complex_")
Fyw[:220] = fft(yw[:220]) # Take FFT of signal y
Fyw[220] = Fyw[0]

plt.subplot(2, 2, 4) # Add an axes to the current figure (top right pltot)
plt.title("Frequency domain (windowed)")
plt.plot(F, np.abs(Fyw)/110, "+-", label="562.5 Hz, 22.5 cycles") # pltot Fy vs F (magnitude)
plt.xlim(0, 1000) # Displtay up to Nyquist frequency
plt.ylim(0, 6)
plt.xlabel("Frequency / Hz") # Set the label for the x-axis
plt.ylabel("Ampltitude / arb. unit") # Set the label for the y-axis
plt.legend(loc="upper right") # pltace a legend on the axes
plt.grid(which="major", color="#AAAAAA", linestyle="-", linewidth=0.8) # Configure the major grid lines
plt.grid(which="minor", color="#BBBBBB", linestyle="--", linewidth=0.5) # Configure the minor grid lines
plt.minorticks_on() # Displtay minor ticks on the axes

plt.subplots_adjust(wspace=0.5, hspace=0.5)
plt.show() # Displtay all open figures