# -*- coding: utf-8 -*-
"""
Part II Spring Term
Lecture 3: Plate vibration, disc vibration, and FRFs

Last modified on Wed Sep 14 01:40:45 2022
@author: Dr Frederic Cegla, Amanda Lee
"""

###############################################################
# Import required functions, modules, packages, and libraries #
###############################################################
# Functions
from numpy import pi as pi                     # Import NumPy constants and mathematical functions
from numpy.fft import fft as fft, ifft as ifft # Import NumPy discrete Fourier transform functions

# Modules, packages, and libraries
import numpy as np                             # Import the fundamental package for scientific computing with Python
import matplotlib.pyplot as plt                # Import the state-based interface for interactive plots and simple cases of programmatic plot generation

####################################
# Step 1: Time domain (excitation) #
####################################
t = np.linspace(0, 1-(1/512), 512) # Create time vector
y = np.zeros(len(t)) # Initialise signal y
p = t[:5] # Initialise signal p
y[50:55] = np.multiply( np.ones(len(p)), np.hanning(len(p)) ) # Create signal y (apply Hanning window to p signal)

fig = plt.figure("Figure 1: SDOF impact response signals") # Create a new figure
fig.suptitle("SDOF impact response signals", fontweight="bold") # Add a centered suptitle to the figure 

plt.subplot(2, 2, 1) # Add an axes to the current figure (top left plot)
plt.title("Time domain (excitation)") # Set a title for the axes
plt.plot(t, y, linewidth=2) # Plot y vs t
plt.xlim(0, 1) # Set the x limits of the current axes
plt.ylim(bottom=0) # Set the y limits of the current axes
plt.xlabel("Time / s") # Set the label for the x-axis
plt.ylabel("Amplitude / arb. unit") # Set the label for the y-axis

#########################################
# Step 2: Frequency domain (excitation) #
#########################################
T = t[-1] # Define time period
Fs = (len(t)-1)/T # Define sampling frequency
FNy = Fs/2 # Define Nyquist frequency
F = np.linspace(0, Fs, len(t)) # Create frequency vector
Fy = fft(y) # Take FFT of signal y

plt.subplot(2, 2, 2) # Add an axes to the current figure (top right plot)
plt.title("Frequency domain (excitation)") # Set a title for the axes
plt.plot(F, np.abs(Fy), linewidth=2) # Plot Fy vs F (magnitude)
plt.xlim(0, FNy) # Display up to Nyquist frequency
plt.ylim(bottom=0) # Set the y limits of the current axes
plt.xlabel("Frequency / Hz") # Set the label for the x-axis
plt.ylabel("Amplitude / arb. unit") # Set the label for the y-axis

#######################################
# Step 3: Frequency domain (response) #
#######################################
m1 = 0.05 # Mass in kg
f1 = 100 # Natural frequency in Hz
c1 = 0.1 # Damping ratio

w = 2 * pi * F # Frequency vector
w1 = 2 * pi * f1 # Natural frequency in rad/s
k1 = m1 * (w1**2) # Stiffness
x = w / w1 # Frequency ratio
a = 1 - (x**2) # Real denominator
b = 2 * c1 * x # Imaginary denominator
den = a + (1j*b) # Real and imaginary denominator
h = 1 / k1 / den # Freqeuncy response function

plt.subplot(2, 2, 4) # Add an axes to the current figure (bottom right plot)
plt.title("Frequency domain (response)") # Set a title for the axes
plt.plot(F, np.abs(h), linewidth=2) # Plot h vs F (magnitude)
plt.xlim(0, FNy) # Display up to Nyquist frequency
plt.ylim(bottom=0) # Set the y limits of the current axes
plt.xlabel("Frequency / Hz") # Set the label for the x-axis
plt.ylabel("Amplitude / arb. unit") # Set the label for the y-axis

##################################
# Step 4: Time domain (response) #
##################################
Fy1 = h * Fy # Create signal Fy1
Fy1_new = np.zeros(np.size(Fy1), dtype="complex_") # Initialise an array to store new spectrum
FNy_idx = int( len(F)/2 ) - 1 # Find the Nyquist frequency array index
Fy1_new[1 : FNy_idx+1] = Fy1[1 : FNy_idx+1] # Copy spectrum up to Nyquist frequency
Fy1_new[FNy_idx+1 :] = np.conj( Fy1[FNy_idx+1 : 0 : -1] ) # Recreate spectrum above Nyquist frequency
y1 = np.real( ifft(Fy1_new) ) # Take IFFT of signal Fy_new

plt.subplot(2, 2, 3) # Add an axes to the current figure (bottom left plot)
plt.title("Time domain (response)") # Set a title for the axes
plt.plot(t, y1, linewidth=2) # Plot y1 vs t
plt.xlim(0, 1) # Set the x limits of the current axes
plt.xlabel("Time / s") # Set the label for the x-axis
plt.ylabel("Amplitude / arb. unit") # Set the label for the y-axis

plt.subplots_adjust(wspace=0.5, hspace=0.5) # Adjust the subplot layout parameters
plt.show() # Display all open figures