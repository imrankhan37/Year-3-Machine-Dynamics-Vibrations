# -*- coding: utf-8 -*-
"""
Part I Autumn Term
Lecture 3: Practical application of time domain and fourier domain processing

Microphone recording and its spectrum.

Last modified on Thu Sep 15 01:29:12 2022
@author: Dr Frederic Cegla, Amanda Lee
"""

###############################################################
# Import required functions, modules, packages, and libraries #
###############################################################
# Functions
from numpy.fft import fft as fft                # Import NumPy discrete Fourier transform functions
from matplotlib.widgets import Button as Button # Import matplotlib button widget

# Modules, packages, and libraries
import soundfile as sf                          # Import audio library to read and write sounds
import sounddevice as sd                        # Import audio library to play and record sounds
import numpy as np                              # Import the fundamental package for scientific computing with Python
import matplotlib.pyplot as plt                 # Import the state-based interface to matplotlib for interactive plots and simple cases of programmatic plot generation

###############################
# Function 1: Audio recording #
###############################
def AudioRec(T, Fs, NumChannels):
    """
    This function records an audio object when called by button.
    
    Parameters
    ----------
    T : float
        Duration.
    Fs : int
        Sampling frequency, in the most cases this will be 44100 or 48000 frames per second.
    NumChannels : int
        Number of channels.
    
    Returns
    -------
    y : list
        Audio data.
    brec : Button object of matplotlib.widgets module
        Record button.
    bplay: Button object of matplotlib.widgets module
        Play button.
    """
    # Global variables
    global y, Fy # Declare global variables for time domain signal y and frequency domain spectrum Fy
    
    # Record audio
    y = sd.rec(int(T*Fs), samplerate=Fs, channels=NumChannels) # Record audio data from your sound device into a NumPy array
    sd.wait() # Block the Python interpreter until playback is finished
    
    # Save audio
    sf.write("TestE.wav", y, Fs) # Write audio data to file
    
    # Plot spectrum
    plt.ion() # Turn on interactive plot mode
    fig = plt.figure() # Create a new figure
    fig.suptitle("Microphone recording", fontweight="bold") # Add a centered suptitle to the figure
    
    plt.subplot(2, 1, 1) # Add an axes to the current figure (top plot)
    plt.title("Time domain signal") # Set a title for the axes
    t = np.linspace(0, T, len(y)) # Create time vector
    l1, = plt.plot(t, y) # Plot y vs t
    plt.xlim(0, T) # Set the x-limits of the current axes
    plt.xlabel("Time [s]") # Set the label for the x-axis
    plt.ylabel("Amplitude [arb. unit]") # Set the label for the y-axis
    
    plt.subplot(2, 1, 2) # Add an axes to the current figure (bottom plot)
    plt.title("Frequency domain spectrum") # Set a title for the axes
    F = np.linspace(0, Fs, len(t)) # Create frequency vector
    Fy = np.zeros(len(y)) # Initialise frequency domain spectrum
    Fy = np.transpose( fft( np.multiply( np.hanning(len(y)), np.transpose(y) ) ) ) # Take FFT of signal y (after applying Hanning window)
    l2, = plt.plot(F[:int(len(F)/2)], np.abs(Fy[:int(len(F)/2)])) # Plot Fy vs F (magnitude) up to Nyquist frequency
    plt.xlim(0, 5000) # Set the x-limits of the current axes
    plt.ylim(bottom=0) # Set the y-limits of the current axes
    plt.xlabel("Frequency [Hz]") # Set the label for the x-axis
    plt.ylabel("Amplitude [arb. unit]") # Set the label for the y-axis
    
    class Audio: # Create Audio class object
        def rec(self, event): # Define rec function
            global y, Fy # Declare global variables for time domain signal y and frequency domain spectrum Fy
            y = sd.rec(int(T*Fs), samplerate=Fs, channels=NumChannels) # Record audio data from your sound device into a NumPy array
            sd.wait() # Block the Python interpreter until playback is finished
            Fy = np.zeros(len(y)) # Initialise frequency domain spectrum
            Fy = np.transpose( fft( np.multiply( np.hanning(len(y)), np.transpose(y) ) ) ) # Take FFT of signal y (after applying Hanning window)
            sf.write("TestE.wav", y, Fs) # Write audio data to file
            l1.set_ydata(y) # Update time domain signal y in plot
            l2.set_ydata(np.abs(Fy[:int(len(F)/2)])) # Update frequency domain spectrum Fy in plot
            fig.canvas.draw_idle() # Update plot
            
        def play(self, event): # Define play function
            AudioPlay(y, Fs) # Play audio
            
    callback = Audio() # Set callback handle
    axrec = plt.axes([0.7, 0.05, 0.1, 0.075]) # Set record button position
    axplay = plt.axes([0.81, 0.05, 0.1, 0.075]) # Set play button position
    brec = Button(axrec, "Record") # Create record button
    bplay = Button(axplay, "Play") # Create play button
    brec.on_clicked(callback.rec) # Activate record button
    bplay.on_clicked(callback.play) # Activate play button
    
    plt.subplots_adjust(hspace=0.8, bottom=0.2)
    plt.show() # Display all open figures
    return y, brec, bplay # Return audio data and button widget handles

##############################
# Function 2: Audio playback #
##############################
def AudioPlay(y, Fs):
    """
    This function plays the recorded audio object when called by button.
    
    Parameters
    ----------
    y : list
        Audio data.
    Fs : int
        Sampling frequency, in the most cases this will be 44100 or 48000 frames per second.

    Returns
    -------
    None.
    """
    # Play audio
    sd.play(y, Fs) # Playback of audio data y with a sampling frequency of Fs
    sd.wait() # Block the Python interpreter until playback is finished
    return None # Return NoneType object

###############################################
# Demo: Microphone recording and its spectrum #
###############################################
# Record and play audio
T = 2                                         # Recording duration in s
Fs = 44100                                    # Sampling frequency in Hz
NumChannels = 1                               # Number of input channels
y, brec, bplay = AudioRec(T, Fs, NumChannels) # Record and play audio