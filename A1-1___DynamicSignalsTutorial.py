# -*- coding: utf-8 -*-
"""
Part I Autumn Term
Tutorial Questions Lecture 1: Dynamic Signals

This script demonstrates the phenomenon of beats as dynamic signals (Question 1).
Note that there are no codes for Questions 2-5.

Last modified on Mon Sep 12 13:06:37 2022
@author: Dr Frederic Cegla, Amanda Lee
"""

###############################################################
# Import required functions, modules, packages, and libraries #
###############################################################
# Functions
from numpy import pi as pi, sin as sin      # Import NumPy constants and mathematical functions

# Modules, packages, and libraries
import numpy as np                          # Import the fundamental package for scientific computing with Python
import matplotlib.pyplot as plt             # Import the state-based interface to matplotlib for interactive plots and simple cases of programmatic plot generation
import tkinter as tk                        # Import the standard Python interface to the Tcl/Tk GUI toolkit
import tkinter.messagebox as messagebox     # Import Tkinter message prompts module
import tkinter.simpledialog as simpledialog # Import standard Tkinter input dialogs module

#####################
# Question 1: Beats #
#####################
# Define signals
t  = np.linspace(0, 1, 1000)         # Create time vector in s
f1 = 20                              #---> Edit frequency 1 in Hz
f2 = 25                              #---> Edit frequency 2 in Hz
f3 = 30                              #---> Edit frequency 3 in Hz
y  = sin(2*pi*f1*t) + sin(2*pi*f2*t) # Create signal y (f1 + f2)
x  = sin(2*pi*f1*t) + sin(2*pi*f3*t) # Create signal x (f1 + f3)

# Plot signals
fig1 = plt.figure(); fig1.suptitle("Time domain signals of mixed sine waves", fontweight="bold")

ax1 = fig1.add_subplot(2, 1, 1); ax1.set_title(f"Signal y: {f1} Hz + {f2} Hz")
line1, = ax1.plot(t, y, "r"); line2 = ax1.axvline(x=(1/(f2-1)), color="k", linestyle="--")
ax1.set_xlim(t[0], t[-1]); ax1.set_ylim(-2, 2)
ax1.set_xlabel("Time [s]"); ax1.set_ylabel("Amplitude [arb. unit]")
ax1.axhline(y=0, color="k", zorder=1); ax1.axvline(x=0, color="k", zorder=1)
ax1.grid(which="major")
ax1.grid(which="minor", color="#BBBBBB", linestyle="--", linewidth=0.5); ax1.minorticks_on()

ax2 = fig1.add_subplot(2, 1, 2); ax2.set_title(f"Signal x: {f1} Hz + {f3} Hz")
line3, = ax2.plot(t, x, "b"); line4 = ax2.axvline(x=1/(f3-f1), color="k", linestyle="--")
ax2.set_xlim(t[0], t[-1]); ax2.set_ylim(-2, 2)
ax2.set_xlabel("Time [s]"); ax2.set_ylabel("Amplitude [arb. unit]")
ax2.axhline(y=0, color="k", zorder=1); ax2.axvline(x=0, color="k", zorder=1)
ax2.grid(which="major")
ax2.grid(which="minor", color="#BBBBBB", linestyle="--", linewidth=0.5); ax2.minorticks_on()

fig1.tight_layout(); plt.show()

# Dialog box for beat periods and beat frequencies
while True:
    root = tk.Tk(); root.lift()
    Ty = simpledialog.askfloat(title="Signal y", prompt="Input the beat period of signal y in seconds:"); root.destroy()
    if not Ty and not Ty==0:
        break
    elif Ty == 1/(f2-f1):
        root = tk.Tk(); root.lift()
        messagebox.showinfo(title="Correct answer",
                            message=f"Correct answer.\nThe beat period of signal y is {1/(f2-f1)} s,"
                            f" which is equivalent to a beat frequency of {f2-f1} Hz."); root.destroy()
        break
    else:
        root = tk.Tk(); root.lift()
        messagebox.showerror(title="Wrong answer", message="Wrong answer.\nPlease try again."); root.destroy()
        continue

while True:
    root = tk.Tk(); root.lift()
    Tx = simpledialog.askfloat(title="Signal x", prompt="Input the beat period of signal x in seconds:"); root.destroy()
    if not Tx and not Tx==0:
        break
    elif Tx == 1/(f3-f1):
        root = tk.Tk(); root.lift()
        messagebox.showinfo(title="Correct answer",
                            message=f"Correct answer.\nThe beat period of signal x is {1/(f3-f1)} s,"
                            f" which is equivalent to a beat frequency of {f3-f1} Hz."); root.destroy()
        break
    else:
        root = tk.Tk(); root.lift()
        messagebox.showerror(title="Wrong answer", message="Wrong answer.\nPlease try again."); root.destroy()
        continue
            
###############################
# Question 2: Saw-tooth waves #
###############################
# No code for Question 2

##############################################
# Question 3: Dynamic range of A/D converter #
##############################################
# No code for Question 3

#####################################
# Question 4: Fourier decomposition #
#####################################
# No code for Question 4

######################################
# Question 5: Range of A/D converter #
######################################
# No code for Question 5