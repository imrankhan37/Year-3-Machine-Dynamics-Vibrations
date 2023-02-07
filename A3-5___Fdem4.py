# -*- coding: utf-8 -*-
"""
Part I Autumn Term
Lecture 3: Practical application of time domain and fourier domain processing

Fourier transform.

Last modified on Thu Sep 15 05:56:16 2022
@author: Dr Frederic Cegla, Amanda Lee
"""

###############################################################
# Import required functions, modules, packages, and libraries #
###############################################################
# Functions
from numpy.fft import fft as fft, ifft as ifft  # Import NumPy discrete Fourier transform functions

# Modules, packages, and libraries
import numpy as np                              # Import the fundamental package for scientific computing with Python
import matplotlib.pyplot as plt                 # Import the state-based interface for interactive plots and simple cases of programmatic plot generation
from matplotlib.widgets import Button as Button # Import matplotlib button widget

############################
# Function: Next frequency #
############################
def NextFrequency():
    """
    This function extracts the next frequency in the spectrum when called by button.
    
    Parameters
    ----------
    None.

    Returns
    -------
    bnext : Button object of matplotlib.widgets module
        Next button.
    """
    global fcount # Set fcount as global variable
    global ax1, ax2, ax3, ax4 # Set axes as global variables
    global line1, line2, line3, line4 # Set lines as global variables
    
    plt.ion() # Turn on interactive plot mode
    fig = plt.figure("Figure 1: SDOF impact signals") # Create a new figure
    fig.suptitle("SDOF impact signals", fontweight="bold") # Set a title for the axes

    ax1 = plt.subplot(2, 2, 1) # Add an axes to the current figure (top plot)
    ax1.set_title("Time domain") # Set a title for the axes
    line5, = ax1.plot(t, y, "b-", linewidth=4) # Plot y vs t
    line1, = ax1.plot(t, ny, "r--") # Plot ny vs t
    ax1.set_xlim(0, 1) # Set the x limits of the current axes
    ax1.set_ylim(-0.3, 0.3) # Set the y limits of the current axes
    ax1.set_xlabel("Time / s") # Set the label for the x-axis
    ax1.set_ylabel("Amplitude / arb. unit") # Set the label for the y-axis
    ax1.grid(which="major", color="#AAAAAA", linestyle="-", linewidth=0.8) # Configure the major grid lines
    ax1.grid(which="minor", color="#BBBBBB", linestyle="--", linewidth=0.5) # Configure the minor grid lines
    ax1.minorticks_on() # Display minor ticks on the axes

    ax2 = plt.subplot(2, 2, 3) # Add an axes to the current figure (top plot)
    ax2.set_title("Time domain") # Set a title for the axes
    line2, = ax2.plot(t, nny, "r-") # Plot nny vs t
    ax2.set_xlim(0, 1) # Set the x limits of the current axes
    ax2.set_ylim(-0.3, 0.3) # Set the y limits of the current axes
    ax2.set_xlabel("Time / s") # Set the label for the x-axis
    ax2.set_ylabel("Amplitude / arb. unit") # Set the label for the y-axis
    ax2.grid(which="major", color="#AAAAAA", linestyle="-", linewidth=0.8) # Configure the major grid lines
    ax2.grid(which="minor", color="#BBBBBB", linestyle="--", linewidth=0.5) # Configure the minor grid lines
    ax2.minorticks_on() # Display minor ticks on the axes

    ax3 = plt.subplot(2, 2, 2) # Add an axes to the current figure (top plot)
    ax3.set_title("Frequency domain") # Set a title for the axes
    line6, = ax3.plot(F, np.abs(f)/16, "b+-", linewidth=4) # Plot f vs F (magnitude)
    line3, = ax3.plot(F, np.abs(ff)/16, "r+--") # Plot ff vs F (magnitude)
    ax3.set_xlim(0, FNy) # Set the x limits of the current axes
    ax3.set_ylim(0, 0.2) # Set the y limits of the current axes
    ax3.set_xlabel("Frequency / Hz") # Set the label for the x-axis
    ax3.set_ylabel("Amplitude / arb. unit") # Set the label for the y-axis
    ax3.grid(which="major", color="#AAAAAA", linestyle="-", linewidth=0.8) # Configure the major grid lines
    ax3.grid(which="minor", color="#BBBBBB", linestyle="--", linewidth=0.5) # Configure the minor grid lines
    ax3.minorticks_on() # Display minor ticks on the axes

    ax4 = plt.subplot(2, 2, 4) # Add an axes to the current figure (top plot)
    ax4.set_title("Frequency domain") # Set a title for the axes
    line4, = ax4.plot(F, np.abs(fff)/16, "r+-") # Plot fff vs F (magnitude)
    ax4.set_xlim(0, FNy) # Set the x limits of the current axes
    ax4.set_ylim(0, 0.2) # Set the y limits of the current axes
    ax4.set_xlabel("Freqeuncy / Hz") # Set the label for the x-axis
    ax4.set_ylabel("Amplitude / arb. unit") # Set the label for the y-axis
    ax4.grid(which="major", color="#AAAAAA", linestyle="-", linewidth=0.8) # Configure the major grid lines
    ax4.grid(which="minor", color="#BBBBBB", linestyle="--", linewidth=0.5) # Configure the minor grid lines
    ax4.minorticks_on() # Display minor ticks on the axes
    
    plt.subplots_adjust(hspace=0.5) # Adjust subplots layout
    plt.subplots_adjust(bottom=0.2) # Add margin to plot for button
    fig.canvas.manager.window.showMaximized() # Display maximised figure window
    plt.show() # Display all open figures
    
    class Frequency: # Create frequency class object
        def nextfrequency(self, event): # Define next frequency function
            global fcount # Set fcount as global variable
            global ax1, ax2, ax3, ax4 # Set axes as global variables
            global line1, line2, line3, line4 # Set lines as global variables
            fcount += 2 # Add 2 to fcount
            
            ff = np.copy(f) # Initialise signal ff
            ff[(fcount-1) : (len(F)+1-fcount)] = 0 # Add data to signal ff
            fff = np.copy(ff) # Initialise signal fff
            fff[:(fcount-2)] = 0 # Add data to signal fff
            fff[(len(F)+1-(fcount-2)):] = 0 # Add data to signal fff

            ny = np.zeros(len(F)) # Initialise signal ny
            ny[:32] = np.real( ifft(ff[:32]) ) # Take IFFT of signal Fy_new
            ny[-1] = ny[0] # Edit signal ny
            nny = np.zeros(len(F)) # Initialise signal nny
            nny[:32] = np.real( ifft(fff[:32]) ) # Take IFFT of signal Fy_new
            nny[-1] = nny[0] # Edit signal nny
                        
            ax1.clear() # Clear axes 1
            ax2.clear() # Clear axes 2
            ax3.clear() # Clear axes 3
            ax4.clear() # Clear axes 4
            
            ax1.set_title("Time domain") # Set a title for the axes
            line5, = ax1.plot(t, y, "b-", linewidth=4) # Plot y vs t
            line1, = ax1.plot(t, ny, "r--") # Plot ny vs t
            ax1.set_xlim(0, 1) # Set the x limits of the current axes
            ax1.set_ylim(-0.3, 0.3) # Set the y limits of the current axes
            ax1.set_xlabel("Time / s") # Set the label for the x-axis
            ax1.set_ylabel("Amplitude / arb. unit") # Set the label for the y-axis
            ax1.grid(which="major", color="#AAAAAA", linestyle="-", linewidth=0.8) # Configure the major grid lines
            ax1.grid(which="minor", color="#BBBBBB", linestyle="--", linewidth=0.5) # Configure the minor grid lines
            ax1.minorticks_on() # Display minor ticks on the axes

            ax2 = plt.subplot(2, 2, 3) # Add an axes to the current figure (top plot)
            ax2.set_title("Time domain") # Set a title for the axes
            line2, = ax2.plot(t, nny, "r-") # Plot nny vs t
            ax2.set_xlim(0, 1) # Set the x limits of the current axes
            ax2.set_ylim(-0.3, 0.3) # Set the y limits of the current axes
            ax2.set_xlabel("Time / s") # Set the label for the x-axis
            ax2.set_ylabel("Amplitude / arb. unit") # Set the label for the y-axis
            ax2.grid(which="major", color="#AAAAAA", linestyle="-", linewidth=0.8) # Configure the major grid lines
            ax2.grid(which="minor", color="#BBBBBB", linestyle="--", linewidth=0.5) # Configure the minor grid lines
            ax2.minorticks_on() # Display minor ticks on the axes

            ax3 = plt.subplot(2, 2, 2) # Add an axes to the current figure (top plot)
            ax3.set_title("Frequency domain") # Set a title for the axes
            line6, = ax3.plot(F, np.abs(f)/16, "b+-", linewidth=4) # Plot f vs F (magnitude)
            line3, = ax3.plot(F, np.abs(ff)/16, "r+--") # Plot ff vs F (magnitude)
            ax3.set_xlim(0, FNy) # Set the x limits of the current axes
            ax3.set_ylim(0, 0.2) # Set the y limits of the current axes
            ax3.set_xlabel("Frequency / Hz") # Set the label for the x-axis
            ax3.set_ylabel("Amplitude / arb. unit") # Set the label for the y-axis
            ax3.grid(which="major", color="#AAAAAA", linestyle="-", linewidth=0.8) # Configure the major grid lines
            ax3.grid(which="minor", color="#BBBBBB", linestyle="--", linewidth=0.5) # Configure the minor grid lines
            ax3.minorticks_on() # Display minor ticks on the axes

            ax4 = plt.subplot(2, 2, 4) # Add an axes to the current figure (top plot)
            ax4.set_title("Frequency domain") # Set a title for the axes
            line4, = ax4.plot(F, np.abs(fff)/16, "r+-") # Plot fff vs F (magnitude)
            ax4.set_xlim(0, FNy) # Set the x limits of the current axes
            ax4.set_ylim(0, 0.2) # Set the y limits of the current axes
            ax4.set_xlabel("Freqeuncy / Hz") # Set the label for the x-axis
            ax4.set_ylabel("Amplitude / arb. unit") # Set the label for the y-axis
            ax4.grid(which="major", color="#AAAAAA", linestyle="-", linewidth=0.8) # Configure the major grid lines
            ax4.grid(which="minor", color="#BBBBBB", linestyle="--", linewidth=0.5) # Configure the minor grid lines
            ax4.minorticks_on() # Display minor ticks on the axes
            
            plt.subplots_adjust(hspace=0.5) # Adjust subplots layout
            plt.subplots_adjust(bottom=0.2) # Add margin to plot for button
            fig.canvas.draw_idle() # Redraw the current figure
            fig.canvas.flush_events() # Speed up graph processing
            
    callback = Frequency() # Set callback handle
    axnext = plt.axes([0.7, 0.02, 0.2, 0.075]) # Set next button position
    bnext = Button(axnext, "Next Frequency") # Create next button
    bnext.on_clicked(callback.nextfrequency) # Activate next button
    return bnext # Return button widget handle

###########################
# Demo: Fourier transform #
###########################
t = np.linspace(0, 1, 33) # Create time vector
y = np.zeros(len(t)) # Initialise signal y
y[:8] = t[:8] # Add data to signal y
y[8:24] = 0.5 - t[8:24] # Add data to signal y
y[24:] = t[24:] - 1 # Add data to signal y

fcount = 3 # Set fcount intial value
T = t[-1] # Define time period
Fs = (len(t)-1)/T # Define sampling frequency
FNy = Fs/2 # Define Nyquist frequency
F = np.linspace(0, Fs, len(t)) # Create frequency vector
f = np.zeros(len(F), dtype="complex_") # Initialise signal f
f[:32] = fft( y[:32] ) # Take FFT of signal y

ff = np.copy(f) # Initialise signal ff
ff[(fcount-1) : (len(F)+1-fcount)] = 0 # Add data to signal ff
fff = np.copy(ff) # Initialise signal fff
fff[:(fcount-2)] = 0 # Add data to signal fff
fff[(len(F)+1-(fcount-2)):] = 0 # Add data to signal fff

ny = np.zeros(len(F)) # Initialise signal ny
ny[:32] = np.real( ifft(ff[:32]) ) # Take IFFT of signal Fy_new
ny[-1] = ny[0] # Edit signal ny
nny = np.zeros(len(F)) # Initialise signal nny
nny[:32] = np.real( ifft(fff[:32]) ) # Take IFFT of signal Fy_new
nny[-1] = nny[0] # Edit signal nny

bnext = NextFrequency() # Set up next frequency button and interactive plot