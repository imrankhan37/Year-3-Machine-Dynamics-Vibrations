# -*- coding: utf-8 -*-
"""
Part I Autumn Term
Lecture 2: Frequency domain analysis

Fast Fourier Transform - interpolated zero padding.

Last modified on Thu Sep 15 05:08:52 2022
@author: Dr Frederic Cegla, Amanda Lee
"""

###############################################################
# Import required functions, modules, packages, and libraries #
###############################################################
# Functions
from numpy import pi as pi, sin as sin                                          # Import NumPy constants and mathematical functions
from numpy.fft import fft as fft, ifft as ifft, fft2 as fft2, ifft2 as ifft2    # Import NumPy discrete Fourier transform functions

# Modules, packages, and libraries
import numpy as np                 # Import the fundamental package for scientific computing with Python
import matplotlib.image as mpimg   # Import the image module for basic image loading, rescaling and display operations
import matplotlib.pyplot as plt    # Import the state-based interface for interactive plots and simple cases of programmatic plot generation

##########################################################
# Step 1: Signal Processing (time and frequency domains) #
##########################################################
t = np.linspace(0, 1-(1/512), 512) # Create time vector
y = np.multiply( sin(2*pi*20*t), np.hanning(len(t)) ) # Create signal y (apply Hanning window to 20 Hz signal)

T = t[-1] # Define time period
Fs = (len(t)-1)/T # Define sampling frequency
FNy = Fs/2 # Define Nyquist frequency
F = np.linspace(0, Fs, len(t)) # Create frequency vector
Fy = fft(y) # Take FFT of signal y

fig = plt.figure("Figure 1: Original signal") # Create a new figure
fig.suptitle("Original signal", fontweight="bold") # Add a centered suptitle to the figure

plt.subplot(2, 1, 1) # Add an axes to the current figure (top plot)
plt.title("Time domain") # Set a title for the axes
plt.plot(t, y) # Plot y vs t
plt.xlim(0, 1) # Set the x limits of the current axes
plt.ylim(-1, 1) # Set the y limits of the current axes
plt.xlabel("Time / s") # Set the label for the x-axis
plt.ylabel("Amplitude / arb. unit") # Set the label for the y-axis

plt.subplot(2, 1, 2) # Add an axes to the current figure (bottom plot)
plt.title("Frequency domain") # Set a title for the axes
plt.plot(F, np.abs(Fy)) # Plot Fy vs F (magnitude)
plt.xlim(0, FNy) # Display up to Nyquist frequency
plt.ylim(bottom=0) # Set the y limits of the current axes
plt.xlabel("Frequency / Hz") # Set the label for the x-axis
plt.ylabel("Amplitude / arb. unit") # Set the label for the y-axis

plt.tight_layout() # Adjust subplots layout

#########################################################
# Step 2: Signal processing (interpolated zero padding) #
#########################################################
factor = 8
N = len(Fy)
df = 1
F2 = np.arange(0, ((factor*N)-1)*df + 1, df)
idx1 = int(N/2 + 1)
idx2 = int(-N/2)
Fy2 = np.zeros(N*factor, dtype="complex_")
Fy2[:idx1] = factor * Fy[:idx1]
Fy2[idx2:] = factor * Fy[idx2:]

F2_max = F2[-1]
T = (len(F2)-1)/F2_max
t2 = np.linspace(0, T, len(F2))
y2 = np.real( ifft(Fy2) )

fig = plt.figure("Figure 2: Interpolated zero padding in signal processing") # Create a new figure
fig.suptitle("Interpolated zero padding in signal processing", fontweight="bold") # Set a title for the axes

plt.subplot(2, 1, 1) # Add an axes to the current figure (top plot)
plt.title("Time domain") # Set a title for the axes
plt.plot(t, y, "+", label="Without interpolation") # Plot y vs t
plt.plot(t2, y2, label="With interpolation") # Plot y2 vs t2
plt.xlim(0, 1) # Set the x limits of the current axes
plt.ylim(-1, 1) # Set the y limits of the current axes
plt.xlabel("Time / s") # Set the label for the x-axis
plt.ylabel("Amplitude / arb. unit") # Set the label for the y-axis
plt.legend(loc="upper right") # Place a legend on the axes

plt.subplot(2, 1, 2) # Add an axes to the current figure (bottom plot)
plt.title("Frequency domain") # Set a title for the axes
plt.plot(F, np.abs(Fy), "+", label="Without interpolation") # Plot Fy vs F (magnitude)
plt.plot(F2, np.abs(Fy2), label="With interpolation") # Plot Fy2 vs F2 (magnitude)
plt.xlim(0, FNy) # Display up to Nyquist frequency
plt.ylim(bottom=0) # Set the y limits of the current axes
plt.xlabel("Frequency / Hz") # Set the label for the x-axis
plt.ylabel("Amplitude / arb. unit") # Set the label for the y-axis
plt.legend(loc="upper right") # Place a legend on the axes

plt.tight_layout() # Adjust subplots layout

###############################################
# Step 3: Image processing (RGB to grayscale) #
###############################################
horse = mpimg.imread("A2-06___HorseDetail.jpg")[:256, :256, :] # Read image

R, G, B = horse[:,:,0], horse[:,:,1], horse[:,:,2] # Extract RGB colour of image
horseG = 0.2989 * R + 0.5870 * G + 0.1140 * B # Adjust RGB colour of image to grayscale

fig = plt.figure("Figure 3: Original image") # Create a new figure
fig.suptitle("Original image", fontweight="bold") # Set a title for the axes

plt.subplot(1, 2, 1) # Add an axes to the current figure (left plot)
plt.title("RGB") # Set a title for the axes
plt.imshow(horse) # Display data as an image

plt.subplot(1, 2, 2) # Add an axes to the current figure (right plot)
plt.title("Grayscale") # Set a title for the axes
plt.imshow(horseG) # Display data as an image

plt.subplots_adjust(wspace=0.5) # Adjust subplots layout
fig.canvas.manager.window.showMaximized() # Show maximised figure window

########################################################
# Step 4: Image processing (interpolated zero padding) #
########################################################
factor = 8
N_im = len(horseG)
horseG_big = np.zeros( (N_im*factor, N_im*factor) )
for i in range(N_im):
    for j in range(N_im):
        for a in range(8):
            for b in range(8):
                horseG_big[(i*factor)+a, (j*factor)+b] = horseG[i, j]

Fy_im = fft2(horseG)
idx1 = int(N_im/2 + 1)
idx2 = int(-N_im/2)
Fy2_im = np.zeros( (N_im*factor, N_im*factor), dtype="complex_" )
Fy2_im[:idx1, :idx1] = (factor**2) * Fy_im[:idx1, :idx1]
Fy2_im[:idx1, idx2:] = (factor**2) * Fy_im[:idx1, idx2:]
Fy2_im[idx2:, :idx1] = (factor**2) * Fy_im[idx2:, :idx1]
Fy2_im[idx2:, idx2:] = (factor**2) * Fy_im[idx2:, idx2:]
horseG_new = np.real( ifft2(Fy2_im) )

fig = plt.figure("Figure 4: Interpolated zero padding in image processing") # Create a new figure
fig.suptitle("Interpolated zero padding in image processing", fontweight="bold") # Set a title for the axes

plt.subplot(1, 2, 1) # Add an axes to the current figure (left plot)
plt.title("Without interpolation") # Set a title for the axes
plt.imshow(horseG_big) # Display data as an image

plt.subplot(1, 2, 2) # Add an axes to the current figure (right plot)
plt.title("With interpolation") # Set a title for the axes
plt.imshow(horseG_new) # Display data as an image

plt.subplots_adjust(wspace=0.5) # Adjust subplots layout
fig.canvas.manager.window.showMaximized() # Show maximised figure window
plt.show() # Display all open figures