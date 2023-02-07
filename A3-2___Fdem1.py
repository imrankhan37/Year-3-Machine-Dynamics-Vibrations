# -*- coding: utf-8 -*-
"""
Part I Autumn Term
Lecture 3: Practical application of time domain and fourier domain processing

Building a signal from Fourier components.

Last modified on Thu Sep 15 05:22:56 2022
@author: Dr Frederic Cegla, Amanda Lee
"""

###############################################################
# Import required functions, modules, packages, and libraries #
###############################################################
# Functions
from numpy.fft import fft as fft, ifft as ifft   # Import NumPy discrete Fourier transform functions

# Modules, packages, and libraries
import numpy as np                     # Import the fundamental package for scientific computing with Python
import matplotlib.pyplot as plt        # Import the state-based interface for interactive plots and simple cases of programmatic plot generation

##########
# Step 1 #
##########
n = 33
t = np.linspace(0, 1, n)
y = np.copy(t)
y[8:24] = 0.5 - t[8:24]
y[24:] = t[24:] - 1

plt.figure(1)
plt.axhline(y=0, color="k", linewidth=0.8)
plt.plot(t, y)
plt.xlim(0, t[-1])
ylim = plt.ylim()
plt.grid(which="major", color="#AAAAAA", linestyle="-", linewidth=0.8)

##########
# Step 2 #
##########
f = np.zeros(n, dtype="complex_")
f[:-1] = fft( y[:-1] )

count = 0
start = 0
end = n
fig = plt.figure(2)
ax = fig.add_subplot(111)
ax.axhline(y=0, color="k", linewidth=0.8)
ax.set_xlim(0, t[-1])
ax.set_ylim(ylim)
ax.grid(which="major", color="#AAAAAA", linestyle="-", linewidth=0.8)
while (start < (n-1)//2):
    if (count > 0):
        ff = np.copy(f)
        ff[start:end] = 0
        ny = np.zeros(n)
        ny[:-1] = np.real( ifft(ff[:-1]) )
        ny[-1] = ny[0]
        if (count == 1):
            line, = ax.plot(t, ny)
        else:
            line.set_ydata(ny)
            fig.canvas.draw_idle()
            fig.canvas.flush_events()
    plt.pause(2)
    count += 1
    start += 2
    end -= 2
    
##########
# Step 3 #
##########
T = t[-1]
Fs = (n-1)/T
FNy = Fs/2
F = np.linspace(0, Fs, n)
f[-1] = f[0]

plt.figure(3)
plt.plot(F, np.abs(f)/16)
plt.xlim(0, FNy)
plt.ylim(bottom=0)
plt.show()