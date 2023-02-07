# -*- coding: utf-8 -*-
"""
Part I Autumn Term
Tutorial Questions Lecture 5: Vibration of multi-degree-of-freedom systems

This script models the vibrations of a rigid box (Question 2), an engine and
gearbox mount (Question 3), and a general two-degree-of-freedom system
(Question 4).
Note that there is no code for Questions 1.

Last modified on Thu Sep 15 03:46:06 2022
@author: Dr Frederic Cegla, Dr Ludovic Renson, Amanda Lee
"""

################################################
# Select question(s) to run and display output #
################################################
# Questions
run_Q1 = None           # No code for Question 1
run_Q2 = 1  #---> Edit "1" (True) or "0" (False)
run_Q3 = 0  #---> Edit "1" (True) or "0" (False)
run_Q4 = 0  #---> Edit "1" (True) or "0" (False)

###############################################################
# Import required functions, modules, packages, and libraries #
###############################################################
# Functions
from numpy import pi as pi, sin as sin, cos as cos # Import NumPy constants and mathematical functions

# Modules, packages, and libraries
import numpy as np                                 # Import the fundamental package for scientific computing with Python
import numpy.linalg as LA                          # Import NumPy linear algebra functions package
import scipy.linalg as linalg                      # Import SciPy linear algebra functions package
import tkinter as tk                               # Import the standard Python interface to the Tcl/Tk GUI toolkit
import tkinter.messagebox as messagebox            # Import Tkinter message prompts module
import tkinter.simpledialog as simpledialog        # Import standard Tkinter input dialogs module
import matplotlib.image as mpimg                   # Import the image module for basic image loading, rescaling and display operations
import matplotlib.pyplot as plt                    # Import the state-based interface to matplotlib for interactive plots and simple cases of programmatic plot generation

####################################
# Question 1: Four-wheeled trailer #
####################################
if run_Q1:
    pass

#########################
# Question 2: Rigid box #
#########################
if run_Q2:
    # System parameters
    m  = 5
    I  = 2.5
    ka = 2000
    kb = ka
    kc = 4000
    kd = kc
    a  = 0.3
    b  = 0.4
    c  = 0.4
    d  = 0.3
    
    # System matrices
    M = np.array( [[m, 0, 0],
                   [0, m, 0],
                   [0, 0, I]] )
    K = np.array( [[ (kc+kd),          0,               -((kc*c)-(kd*d))                                  ],
                   [ 0,                (ka+kb),         -((ka*a)-(kb*b))                                  ],
                   [-((kc*c)-(kd*d)), -((ka*a)-(kb*b)),  ((ka*(a**2))+(kb*(b**2))+(kc*(c**2))+(kd*(d**2)))]] )
    
    # Eigenvalues, eigenvectors, and natural frequencies
    w2, X = linalg.eigh(K, M)
    w     = np.sqrt(w2)
    f     = w / (2*pi)

########################################
# Question 3: Engine and gearbox mount #
########################################
if run_Q3:
    # System parameters
    mass = 53
    ix = 1.1
    iy = 2.7
    iz = 2.6
    
    # Location of springs relative to centre of mass
    ax = -0.215
    ay = -0.125
    az = -0.175
    bx = -0.215
    by =  0.125
    bz = -0.175
    cx =  0.285
    cy =  0
    cz =  0
    
    # Spring stiffness
    kax = 30
    kay = 30
    kaz = 20
    kbx = 30
    kby = 30
    kbz = 20
    kcx = 35
    kcy = 35
    kcz = 30
    
    # Mass matrix
    m = np.zeros((6,6))
    m[0,0] = mass
    m[1,1] = mass
    m[2,2] = mass
    m[3,3] = ix
    m[4,4] = iy
    m[5,5] = iz
    
    # Stiffness matrix
    k = np.zeros((6,6))
    k[0,0] =  kax + kbx + kcx
    k[0,4] =  kax*az + kbx*bz + kcx*cz
    k[0,5] =  kax*ay + kbx*by + kcx*cy
    k[1,1] =  kay + kby + kcy
    k[1,3] = -(kay*az + kby*bz + kcy*cz)
    k[1,5] =  kay*ax + kby*bx + kcy*cx
    k[2,2] =  kaz + kbz + kcz
    k[2,3] = -(kaz*ay + kbz*by + kcz*cy)
    k[2,4] = -(kaz*ax + kbz*bx + kcz*cx)
    k[3,1] = -(kay*az + kby*bz + kcy*cz)
    k[3,2] = -(kaz*ay + kbz*by + kcz*cy)
    k[3,3] =  kay*az*az + kby*bz*bz + kaz*ay*ay + kbz*by*by + kcz*cy*cy + kcy*cz*cz
    k[3,4] =  kaz*ay*ax + kbz*by*bx + kcz*cy*cx
    k[3,5] = -(kay*az*ax + kby*bz*bx + kcy*cz*cx)
    k[4,0] =  kax*az + kbx*bz + kcx*cz
    k[4,2] = -(kaz*ax + kbz*bx + kcz*cx)
    k[4,3] =  kaz*ax*ay + kbz*bx*by + kcz*cx*cy
    k[4,4] =  kax*az*az + kbx*bz*bz + kaz*ax*ax + kbz*bx*bx + kcx*cz*cz + kcz*cx*cx
    k[4,5] =  kax*az*ay + kbx*bz*by + kcx*cy*cz
    k[5,0] =  kax*ay + kbx*by + kcx*cy
    k[5,1] =  kay*ax + kby*bx + kcy*cx
    k[5,3] = -(kay*az*ax + kby*bz*bx + kcy*cz*cx)
    k[5,4] =  kax*ay*az + kbx*by*bz + kcx*cy*cz
    k[5,5] =  kax*ay*ay + kbx*by*by + kay*ax*ax + kby*bx*bx + kcx*cy*cy + kcy*cx*cx
    k = 1000*k
    
    # Eigenvalues, eigenvectors, and natural frequencies
    w2, vec = linalg.eigh(k, m)
    w = np.sqrt(w2)
    f = w / (2*pi)
    
    # Interactive simulation
    plt.ion()
    fig1 = plt.figure(); fig1.suptitle("Engine and gearbox mount", fontweight="bold")
    ax1 = fig1.add_subplot(111, projection='3d')
    def MyEngine(ax1):
        ax1.plot([-3, 3], [0, 0], [0, 0], "k", linewidth=5); ax1.plot([0, 0], [-3, 3], [0, 0], "k", linewidth=5); ax1.plot([0, 0], [0, 0], [-3, 3], "k", linewidth=5)
        ax1.set_xlabel("x-axis"); ax1.set_ylabel("y-axis"); ax1.set_zlabel("z-axis")
        ax1.set_xlim(-1, 1); ax1.set_ylim(-1, 1); ax1.set_zlim(-1, 1)
        ax1.set_xticks([]); ax1.set_yticks([]); ax1.set_zticks([])
        return ax1
    ax1  = MyEngine(ax1)
    
    r    = [-1,1]
    X, Y = np.meshgrid(r, r)
    one  = np.ones((2,2))
    box  = np.array( [[X,Y,one], [X,Y,-one], [X,one,Y], [X,-one,Y], [one,X,Y], [-one,X,Y]] )
    for i in range(6):
        ax1.plot_surface(box[i][0], box[i][1], box[i][2], alpha=0.5)
    
    while True:
        root = tk.Tk(); root.lift()
        mode = simpledialog.askinteger(title="Mode number", prompt="Start the animation by entering a mode number\nbetween 1 and 6 (inclusive):"); root.destroy()
        if not mode and not mode==0:
            plt.close(); break
        else:
            if   mode<1: mode = 1
            elif mode>6: mode = 6
            else       : pass
        
        fig1.suptitle(f"Mode {mode}\nNatural frequency = {round(f[mode-1],1)} Hz", fontweight="bold")
        t = np.linspace(0, 3, 61)
        for ti in t:
            dx  = 0.1 * vec[0,mode-1] * cos(2*pi*10*ti)
            dy  = 0.1 * vec[1,mode-1] * cos(2*pi*10*ti)
            dz  = 0.1 * vec[2,mode-1] * cos(2*pi*10*ti)
            dtx = 0.1 * vec[3,mode-1] * cos(2*pi*10*ti)
            dty = 0.1 * vec[4,mode-1] * cos(2*pi*10*ti)
            dtz = 0.1 * vec[5,mode-1] * cos(2*pi*10*ti)
            
            Rx = np.array( [[        1,         0,         0],
                            [        0,  cos(dtx), -sin(dtx)],
                            [        0,  sin(dtx),  cos(dtx)]] )
            Ry = np.array( [[ cos(dty),         0,  sin(dty)],
                            [        0,         1,         0],
                            [-sin(dty),         0,  cos(dty)]] )
            Rz = np.array( [[ cos(dtz), -sin(dtz),         0],
                            [ sin(dtz),  cos(dtz),         0],
                            [        0,         0,         1]] )
            
            ax1.clear(); ax1 = MyEngine(ax1)
            for i in range(6):
                XX    = np.array( [box[i][j].flatten() for j in range(3)] )
                tempX = LA.multi_dot([Ry, Rz, XX])
                YY    = np.array( [tempX[j] for j in range(3)] )
                tempY = LA.multi_dot([Rx, Rz, YY])
                ZZ    = np.array( [tempY[j] for j in range(3)] )
                tempZ = LA.multi_dot([Rx, Ry, ZZ])
                XYZ   = np.array( [np.reshape(tempZ[j],(2,2)) for j in range(3)] )
                ax1.plot_surface(XYZ[0]+dx, XYZ[1]+dy, XYZ[2]+dz, alpha=0.5)
            fig1.canvas.draw_idle(); plt.pause(0.1)

###########################################
# Question 4: Two-degree-of-freedom model #
###########################################
if run_Q4:
    # System parameters
    M1 = 1
    M2 = 1
    k1 = 0.39478e6
    k2 = 1.57910e6
    c1 = 10
    c2 = 10
    
    # Frequency range
    f = np.linspace(0, 400, 4001)
    omega = 2*pi*f
    
    # Frequency response function
    res = np.zeros((2, len(f)), dtype="complex_")
    for i in range(len(f)):
        A = np.zeros((2,2), dtype="complex_")
        A[0,0] = k1 + k2 - (M1*(omega[i]**2)) + ((c1+c2)*omega[i]*1j)
        A[0,1] = -k2 - (c2*omega[i]*1j)
        A[1,0] = -k2 - (c2*omega[i]*1j)
        A[1,1] = k2 - (M2*(omega[i]**2)) + (c2*omega[i]*1j)
        b = np.array([[1], [0]])
        x = LA.lstsq(A, b, rcond=None)[0]
        res[:,i] = np.squeeze(x)
    
    # Interactive simulation
    plt.ion()
    fig2 = plt.figure(); fig2.suptitle("Two-degree-of-freedom model", fontweight="bold"); gs = fig2.add_gridspec(3,2)
    ax2 = fig2.add_subplot(gs[0,0]); ax2.set_title("Frequency response function")
    line1, = ax2.semilogy(f, np.abs(res[0,:]), "r", picker=True, pickradius=2)
    line2, = ax2.semilogy(f, np.abs(res[1,:]), "b", picker=True, pickradius=2)
    ylim = ax2.get_ylim()
    line3, = ax2.plot([0,0], [ylim[0], ylim[1]], "k--")
    line4, = ax2.plot([], [], "ko")
    ax2.set_xlim(f[0], f[-1]); ax2.set_ylim(ylim)
    ax2.set_xlabel("Frequency [Hz]"); ax2.set_ylabel("Log receptance [no unit]")
    
    ax3 = fig2.add_subplot(gs[1,0]); ax3.set_title("Mass-spring system")
    img = mpimg.imread("A5-2___MassSpringSystem.jpg"); ax3.imshow(img)
    ax3.get_xaxis().set_ticks([]); ax3.get_yaxis().set_ticks([])
    
    ax4 = fig2.add_subplot(gs[2, 0]); ax4.set_title("Mass motion at selected frequency")
    fill1, = ax4.fill([0.8, 0.8, 1.2, 1.2], [0.4, 0.64, 0.64, 0.4], facecolor="r", edgecolor="k")
    fill2, = ax4.fill([1.8, 1.8, 2.2, 2.2], [0.4, 0.64, 0.64, 0.4], facecolor="b", edgecolor="k")
    ax4.get_xaxis().set_ticks([]); ax4.get_yaxis().set_ticks([])
    ax4.set_xlim(0,3); ax4.set_ylim(0,1)
    
    ax5 = fig2.add_subplot(gs[:, 1]); ax5.set_title("Nyquist plot")
    ax5.plot(np.real(res[0,:]), np.imag(res[0,:]), "r+-")
    ax5.plot(np.real(res[1,:]), np.imag(res[1,:]), "b+-")
    line5, = ax5.plot([0, np.real(res[0,0])], [0, np.imag(res[0,0])], "ko--")
    line6, = ax5.plot([0, np.real(res[1,0])], [0, np.imag(res[1,0])], "ko--")
    ax5.set_xlabel("Real axis [arb. unit]"); ax5.set_ylabel("Imaginary axis [arb. unit]")
    ax5.axhline(y=0, color="k", zorder=1); ax5.axvline(x=0, color="k", zorder=1)
    ax5.axis("equal")
    
    def onpick(event):
        myLine = event.artist
        xData = myLine.get_xdata()
        yData = myLine.get_ydata()
        idx = event.ind
        points = tuple(zip(xData[idx], yData[idx]))
        myIdx = idx[0]
        myXPoint = points[0][0]
        
        ax3.set_title(f"Mass motion at {int(myXPoint)} Hz")
        line1.set_ydata( np.abs(res[0,:]) )
        line2.set_ydata( np.abs(res[1,:]) )
        line3.set_xdata( [myXPoint, myXPoint] )
        line4.set_xdata( [myXPoint, myXPoint] )
        line4.set_ydata( [np.abs(res[0,myIdx]), np.abs(res[1,myIdx])] )
        line5.set_xdata( [0, np.real(res[0,myIdx])] )
        line5.set_ydata( [0, np.imag(res[0,myIdx])] )
        line6.set_xdata( [0, np.real(res[1,myIdx])] )
        line6.set_ydata( [0, np.imag(res[1,myIdx])] )
        
        t = np.linspace(0, 5*myXPoint, 1000)
        for i in t:
            x   = res[:,myIdx]
            x   = 0.5*x/max(abs(x))
            x1  = x[0] * cos(2*pi*myXPoint*i); x1+=1
            x2  = x[1] * cos(2*pi*myXPoint*i); x2+=2
            dx1 = [x1-0.2, x1-0.2, x1+0.2, x1+0.2]
            dx2 = [x2-0.2, x2-0.2, x2+0.2, x2+0.2]
            
            fill1.set_xy(np.array([dx1, [0.4, 0.64, 0.64, 0.4]]).T)
            fill2.set_xy(np.array([dx2, [0.4, 0.64, 0.64, 0.4]]).T)
            fig2.canvas.draw_idle(); plt.pause(0.25)
        
    fig2.canvas.mpl_connect("pick_event", onpick)
    fig2.subplots_adjust(hspace=0.5, wspace=0.5); fig2.canvas.manager.window.showMaximized(); plt.show()
    
    root = tk.Tk(); root.lift()
    messagebox.showinfo(title="Help dialog",\
                        message=("Click any point in the top left plot to"
                                 " select a frequency at which the mass motion"
                                 " is simulated in the bottom left plot. Wait"
                                 " for the simulation to finish before"
                                 " selecting a new frequency.")); root.destroy()