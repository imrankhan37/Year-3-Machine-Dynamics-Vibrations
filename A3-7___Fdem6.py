# -*- coding: utf-8 -*-
"""
Part I Autumn Term
Lecture 3: Practical application of time domain and fourier domain processing

Frequency resolution and zero padding.

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
t = np.linspace(0, 0.05, 256) # Create time vector
y = sin(2*pi*550*t)
yw = np.multiply( y, np.hanning(len(y)) ) # Create signal y (apply Hanning window to 20 Hz signal)

plt.figure("Figure 1: Time domain signals") # Create a new figure
plt.title("Time domain signals") # Set a title for the axes
plt.plot(t, yw) # Plot yw vs t
plt.xlim(t[0], t[-1])
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
Fy = fft(yw) # Take FFT of signal y

plt.figure("Figure 2: Frequency domain signals") # Create a new figure
plt.title("Frequency domain signals") # Add a centered suptitle to the figure
plt.plot(F, np.abs(Fy)/128) # Plot Fy vs F (magnitude)
plt.xlim(0, 1000)
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
t1 = np.linspace(0, 0.4, 2048) # Create time vector
yw1 = np.zeros(2048)
yw1[:len(yw)] = yw

plt.figure("Figure 3: Time domain signals") # Create a new figure
plt.title("Time domain signals") # Set a title for the axes
plt.plot(t1, yw1) # Plot yw vs t
plt.xlim(t[0], t[-1])
plt.xlabel("Time / s") # Set the label for the x-axis
plt.ylabel("Amplitude / arb. unit") # Set the label for the y-axis
plt.show() # Display all open figures

##########
# Step 4 #
##########
T = t1[-1] # Define time period
Fs = (len(t1)-1)/T # Define sampling frequency
FNy = Fs/2 # Define Nyquist frequency
F = np.linspace(0, Fs, len(t1)) # Create frequency vector
Fy = np.zeros(len(F), dtype="complex_")
Fy = fft(yw1) # Take FFT of signal y

plt.figure("Figure 4: Frequency domain signals") # Create a new figure
plt.title("Frequency domain signals") # Add a centered suptitle to the figure
plt.plot(F, np.abs(Fy)/1024) # Plot Fy vs F (magnitude)
plt.xlim(0, 1000)
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
t = np.linspace(0, 0.05, 256) # Create time vector
y1 = sin(2*pi*545*t)
y2 = sin(2*pi*555*t)
y = y1 + y2
yw = np.multiply( y, np.hanning(len(y)) ) # Create signal y (apply Hanning window to 20 Hz signal)

plt.figure("Figure 5: Time domain signals") # Create a new figure
plt.title("Time domain signals") # Set a title for the axes
plt.plot(t, yw) # Plot yw vs t
plt.xlim(t[0], t[-1])
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
Fy = fft(yw) # Take FFT of signal y

plt.figure("Figure 6: Frequency domain signals") # Create a new figure
plt.title("Frequency domain signals") # Add a centered suptitle to the figure
plt.plot(F, np.abs(Fy)/128) # Plot Fy vs F (magnitude)
plt.xlim(0, 1000)
plt.ylim(bottom=0) # Set the y limits of the current axes
plt.xlabel("Frequency / Hz") # Set the label for the x-axis
plt.ylabel("Amplitude / arb. unit") # Set the label for the y-axis
plt.grid(which="major", color="#AAAAAA", linestyle="-", linewidth=0.8) # Configure the major grid lines
plt.grid(which="minor", color="#BBBBBB", linestyle="--", linewidth=0.5) # Configure the minor grid lines
plt.minorticks_on() # Display minor ticks on the axes
plt.show() # Display all open figures

##########
# Step 7 #
##########
t1 = np.linspace(0, 0.4, 2048) # Create time vector
yw1 = np.zeros(2048)
yw1[:len(yw)] = yw

plt.figure("Figure 7: Time domain signals") # Create a new figure
plt.title("Time domain signals") # Set a title for the axes
plt.plot(t1, yw1) # Plot yw vs t
plt.xlim(t[0], t[-1])
plt.xlabel("Time / s") # Set the label for the x-axis
plt.ylabel("Amplitude / arb. unit") # Set the label for the y-axis
plt.show() # Display all open figures

##########
# Step 8 #
##########
T = t1[-1] # Define time period
Fs = (len(t1)-1)/T # Define sampling frequency
FNy = Fs/2 # Define Nyquist frequency
F = np.linspace(0, Fs, len(t1)) # Create frequency vector
Fy = np.zeros(len(F), dtype="complex_")
Fy = fft(yw1) # Take FFT of signal y

plt.figure("Figure 8: Frequency domain signals") # Create a new figure
plt.title("Frequency domain signals") # Add a centered suptitle to the figure
plt.plot(F, np.abs(Fy)/1024) # Plot Fy vs F (magnitude)
plt.xlim(0, 1000)
plt.ylim(bottom=0) # Set the y limits of the current axes
plt.xlabel("Frequency / Hz") # Set the label for the x-axis
plt.ylabel("Amplitude / arb. unit") # Set the label for the y-axis
plt.grid(which="major", color="#AAAAAA", linestyle="-", linewidth=0.8) # Configure the major grid lines
plt.grid(which="minor", color="#BBBBBB", linestyle="--", linewidth=0.5) # Configure the minor grid lines
plt.minorticks_on() # Display minor ticks on the axes
plt.show() # Display all open figures

##########
# Step 9 #
##########
t = np.linspace(0, 0.4, 2048) # Create time vector
y1 = sin(2*pi*545*t)
y2 = sin(2*pi*555*t)
y = y1 + y2
yw = np.multiply( y, np.hanning(len(y)) ) # Create signal y (apply Hanning window to 20 Hz signal)

plt.figure("Figure 9: Time domain signals") # Create a new figure
plt.title("Time domain signals") # Set a title for the axes
plt.plot(t, yw) # Plot yw vs t
plt.xlim(t[0], t[-1])
plt.xlabel("Time / s") # Set the label for the x-axis
plt.ylabel("Amplitude / arb. unit") # Set the label for the y-axis
plt.show() # Display all open figures

###########
# Step 10 #
###########
T = t[-1] # Define time period
Fs = (len(t)-1)/T # Define sampling frequency
FNy = Fs/2 # Define Nyquist frequency
F = np.linspace(0, Fs, len(t)) # Create frequency vector
Fy = np.zeros(len(F), dtype="complex_")
Fy = fft(yw) # Take FFT of signal y

plt.figure("Figure 10: Frequency domain signals") # Create a new figure
plt.title("Frequency domain signals") # Add a centered suptitle to the figure
plt.plot(F, np.abs(Fy)/128) # Plot Fy vs F (magnitude)
plt.xlim(0, 1000)
plt.ylim(bottom=0) # Set the y limits of the current axes
plt.xlabel("Frequency / Hz") # Set the label for the x-axis
plt.ylabel("Amplitude / arb. unit") # Set the label for the y-axis
plt.grid(which="major", color="#AAAAAA", linestyle="-", linewidth=0.8) # Configure the major grid lines
plt.grid(which="minor", color="#BBBBBB", linestyle="--", linewidth=0.5) # Configure the minor grid lines
plt.minorticks_on() # Display minor ticks on the axes
plt.show() # Display all open figures