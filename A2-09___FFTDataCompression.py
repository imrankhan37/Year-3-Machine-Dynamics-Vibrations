# -*- coding: utf-8 -*-
"""
Part I Autumn Term
Lecture 2: Frequency domain analysis

Fast Fourier Transform - data compression.

Last modified on Thu Sep 15 04:59:26 2022
@author: Dr Frederic Cegla, Amanda Lee
"""

###############################################################
# Import required functions, modules, packages, and libraries #
###############################################################
# Functions
from numpy import pi as pi                     # Import NumPy constants and mathematical functions
from numpy.fft import fft as fft, ifft as ifft # Import NumPy discrete Fourier transform functions

# Modules, packages, and libraries
import soundfile as sf                         # Import audio library to read and write sounds
import sounddevice as sd                       # Import audio library to play and record sounds
import numpy as np                             # Import the fundamental package for scientific computing with Python
import matplotlib.pyplot as plt                # Import the state-based interface for interactive plots and simple cases of programmatic plot generation

######################################
# Step 1: Display time domain signal #
######################################
(y, Fs) = sf.read("Dadadida.wav") # Read data and extract sampling frequency from sound file
FNy = Fs/2 # Define Nyquist frequency
T = (len(y)-1)/Fs # Define time period
t = np.linspace(0, T, len(y)) # Create time vector

sd.play(y, Fs) # Playback of audio data y with a sampling frequency of Fs
sd.wait() # Block the Python interpreter until playback is finished

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

plt.tight_layout() # Adjust subplots layout

##########################################
# Step 3: Filter out selected amplitudes #
##########################################
val = 0.002 * max( abs(Fy) ) # Set amplitude threshold to 0.2% of maximum value

idx = [] # Initialise array to store indices
for i in range(0, len(F)): # Loop through freqeuncy vector
    if (abs(Fy[i]) > val): # If current amplitude is greater than threshold value
        idx += [i] # Add current index to existing array
idx = np.array(idx) # Convert index list to NumPy array

comp_ratio = len(y) / len(idx) # Find audio compression ratio of original size to filtered size
print("The compression ratio is", comp_ratio) # Print audio compression ratio

Fy_new = np.zeros(np.size(Fy), dtype="complex_") # Initialise an array to store new spectrum
Fy_new[idx] = Fy[idx] # Copy spectrum below selected maximum amplitude

plt.figure("Figure 3: Frequency domain signal (compressed)") # Create a new figure
plt.title("Frequency domain signal (compressed)") # Set a title for the axes
plt.plot(F, np.abs(Fy_new), label="Compressed signal") # Plot compressed signal (Fy_new vs F; magnitude)
plt.plot(F, np.abs(Fy), ":", label="Original signal") # Plot original signal (Fy vs F; magnitude)
plt.xlim(0, FNy) # Display up to Nyquist frequency
plt.ylim(bottom=0) # Set the y limits of the current axes
plt.xlabel("Frequency / Hz") # Set the label for the x-axis
plt.ylabel("Amplitude / arb. unit") # Set the label for the y-axis
plt.legend() # Place a legend on the axes

###############################################
# Step 4: Recover original time domain signal #
###############################################
y_new = np.real( ifft(Fy_new) ) # Take IFFT of signal Fy_new
sf.write("DadadidaCompressed.wav", y_new, Fs) # Write audio data to file

sd.play(y_new, Fs) # Playback of audio data y_new with a sampling frequency of Fs
sd.wait() # Block the Python interpreter until playback is finished

plt.figure("Figure 4: Time domain signal (compressed)") # Create a new figure
plt.title("Time domain signal (compressed)") # Set a title for the axes
plt.plot(t, y_new, label="Compressed signal") # Plot reconstructed signal (y_new vs t)
plt.plot(t, y, ":", label="Original signal") # Plot original signal (y vs t)
plt.xlim(t[0], t[-1]) # Set the x limits of the current axes
plt.xlabel("Time / s") # Set the label for the x-axis
plt.ylabel("Amplitude / arb. unit") # Set the label for the y-axis
plt.legend() # Place a legend on the axes
plt.show() # Display all open figures

########################################
# Step 5: Save audio data to text file #
########################################
with open("A2-09___FullySampledAudioData.txt", "w") as f: # Open file for writing in text mode
    for i in range(0, len(t)): # Loop through time vector
        f.write(f"{t[i]} {y[i]}\n") # Write corresponding t and y values to file
f.close() # Close the text file

with open("A2-09___SparseFreqeuncySpectrum.txt", "w") as f: # Open file for writing in text mode
    for i in idx[:len(idx)//2]: # Loop through index vector
        f.write(f"{np.round(F[i],2)}   {np.round(Fy_new[i],2)}\n") # Write corresponding F and Fy_new values to file
f.close() # Close the text file