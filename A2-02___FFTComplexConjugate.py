# -*- coding: utf-8 -*-
"""
Part I Autumn Term
Lecture 2: Frequency domain analysis

Fast Fourier Transform - complex conjugate.

Last modified on Thu Sep 15 02:11:19 2022
@author: Dr Frederic Cegla, Amanda Lee
"""

###############################################################
# Import required functions, modules, packages, and libraries #
###############################################################
# Functions
from numpy import pi as pi                     # Import NumPy constants and mathematical functions
from numpy.fft import fft as fft, ifft as ifft # Import NumPy discrete Fourier transform functions

# Modules, packages, and libraries
import soundfile as sf                         # Import audio library
import numpy as np                             # Import the fundamental package for scientific computing with Python
import matplotlib.pyplot as plt                # Import the state-based interface for interactive plots and simple cases of programmatic plot generation

######################################
# Step 1: Display time domain signal #
######################################
(data, Fs) = sf.read("Chimes.wav") # Read data and extract sampling frequency from sound file
FNy = Fs/2 # Define Nyquist frequency
T = (len(data)-1)/Fs # Define time period
t = np.linspace(0, T, len(data)) # Create time vector
y = data[:, 0] # Extract data from channel 1 only (note that the loaded wave file is stereo)

plt.figure("Figure 1: Time domain signal") # Create a new figure
plt.title("Time domain signal") # Set a title for the axes
plt.plot(t, y) # Plot y vs t
plt.xlim(t[0], t[-1]) # Set the x limits of the current axes
plt.xlabel("Time / s") # Set the label for the x-axis
plt.ylabel("Amplitude / arb. unit") # Set the label for the y-axis

###########################################
# Step 2: Display frequency domain signal #
###########################################
F = np.linspace(0, Fs, len(t)) # Create frequency vector
Fy = fft(y) # Take FFT of signal y

fig = plt.figure("Figure 2: Frequency domain signal") # Create a new figure
fig.suptitle("Frequency domain signal") # Add a centered suptitle to the figure

plt.subplot(2, 1, 1) # Add an axes to the current figure (top plot)
plt.plot(F, np.abs(Fy)) # Plot Fy vs F (magnitude)
plt.xlim(0, FNy) # Display up to Nyquist frequency
plt.ylim(bottom=0) # Set the y limits of the current axes
plt.xlabel("Frequency / Hz") # Set the label for the x-axis
plt.ylabel("Amplitude / arb. unit") # Set the label for the y-axis

plt.subplot(2, 1, 2) # Add an axes to the current figure (bottom plot)
plt.plot(F, np.angle(Fy)) # Plot Fy vs F (phase)
plt.xlim(0, FNy) # Display up to Nyquist frequency
plt.ylim(-pi, pi) # Set the y limits of the current axes
plt.xlabel("Frequency / Hz") # Set the label for the x-axis
plt.ylabel("Phase / rad") # Set the label for the y-axis

plt.tight_layout()

##################################################
# Step 3: Recreate upper half frequency spectrum #
##################################################
Fy_new = np.zeros(np.size(Fy), dtype="complex_") # Initialise an array to store new spectrum
FNy_idx = int( len(F)/2 ) - 1 # Find the Nyquist frequency array index
Fy_new[1 : FNy_idx+1] = Fy[1 : FNy_idx+1] # Copy spectrum up to Nyquist frequency
Fy_new[FNy_idx+1 :] = np.conj( Fy[FNy_idx+1 : 0 : -1] ) # Recreate spectrum above Nyquist frequency

plt.figure("Figure 3: Frequency domain signal (reconstructed)") # Create a new figure
plt.title("Frequency domain signal (reconstructed)") # Set a title for the axes
plt.plot(F, np.abs(Fy_new), label="Reconstructed signal") # Plot reconstructed signal (Fy_new vs F; magnitude)
plt.plot(F, np.abs(Fy), ":", label="Original signal") # Plot original signal (Fy vs F; magnitude)
plt.axvline(x=FNy, color="k", linestyle="--") # Display axis of symmetry at Nyquist frequency
plt.xlim(F[0], F[-1]) # Set the x limits of the current axes
plt.ylim(bottom=0) # Set the y limits of the current axes
plt.xlabel("Frequency / Hz") # Set the label for the x-axis
plt.ylabel("Amplitude / arb. unit") # Set the label for the y-axis
plt.legend() # Place a legend on the axes

###############################################
# Step 4: Recover original time domain signal #
###############################################
y_new = np.real( ifft(Fy_new) ) # Take IFFT of signal Fy_new

plt.figure("Figure 4: Time domain signal (reconstructed)") # Create a new figure
plt.title("Time domain signal (reconstructed)") # Set a title for the axes
plt.plot(t, y_new, label="Reconstructed signal") # Plot reconstructed signal (y_new vs t)
plt.plot(t, y, ":", label="Original signal") # Plot original signal (y vs t)
plt.xlim(t[0], t[-1]) # Set the x limits of the current axes
plt.xlabel("Time / s") # Set the label for the x-axis
plt.ylabel("Amplitude / arb. unit") # Set the label for the y-axis
plt.legend() # Place a legend on the axes
plt.show() # Display all open figures