#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 10:18:27 2021

@author: suraj
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from mpl_toolkits.mplot3d import Axes3D
from scipy.linalg import orth

sigma = 10.0
rho = 28.0
beta = 8.0 / 3.0

xState = -9.42
yState = -9.43
zState = 28.3

#sigma = 16.0
#rho = 120.1
#beta = 4.0

#xState = 22.8 #-9.42
#yState = 35.7 #-9.43
#zState = 114.9 #28.3

def f(state, t):
    x, y, z = state  # Unpack the state vector
    return sigma * (y - x), x * (rho - z) - y, x * y - beta * z  # Derivatives

def df(state, t, x,y,z):
    dx, dy, dz = state  # Unpack the state vector
    return sigma*(dy - dx), dx*(rho - z) - dy - x*dz, y*dx  + x*dy - beta*dz  # Derivatives

dt = 0.001

# The number of iterations to throw away
nTransients = 10
# The number of time steps to integrate over
nIterates = 10000

# The main loop that generates the orbit, storing the states


x0 = [xState, yState, zState]

t = np.linspace(0,nIterates*dt,nIterates+1)
xt = odeint(f, x0, t)

e1 = np.array([1.0,0.0,0.0])
t = np.linspace(0,nTransients*dt,nTransients+1)
dxt = odeint(df, e1, t, args=(xState, yState, zState))

fig = plt.figure()
ax = fig.gca(projection="3d")
ax.plot(xt[:, 0], xt[:, 1], xt[:, 2])
plt.draw()
plt.show()

#%%
# Estimate the LCEs
# The number of iterations to throw away
nTransients = 100
# The number of iterations to over which to estimate
#  This is really the number of pull-backs
nIterates = 10000
# The number of iterations per pull-back
nItsPerPB = 20

x0 = [xState, yState, zState]

# Initial tangent vectors
e1 = np.array([1.0,0.0,0.0])
e2 = np.array([0.0,1.0,0.0])
e3 = np.array([0.0,0.0,1.0])

for n in range(nTransients):
#    for i in range(nItsPerPB):
#        t1 = np.linspace(0,dt,2)
    t1 = np.linspace(0,nItsPerPB*dt,nItsPerPB)
    x = odeint(f, x0, t1)
    x0 = x[-1]
    
    e1t = odeint(df, e1, t1, args=(x0[0], x0[1], x0[2]))
    e1 = e1t[-1]
    
    e2t = odeint(df, e2, t1, args=(x0[0], x0[1], x0[2]))
    e2 = e2t[-1]
    
    e3t = odeint(df, e3, t1, args=(x0[0], x0[1], x0[2]))
    e3 = e3t[-1]
        
    # Normalize the tangent vector
    d = np.linalg.norm(e1)
    e1 = e1/d
    
    # Pull-back: Remove any e1 component from e2
    dote1e2 = np.sum(e1*e2)
    e2 = e2 - dote1e2*e1
    
    # Normalize second tangent vector
    d = np.linalg.norm(e2)
    e2 = e2/d
    
    # Pull-back: Remove any e1 and e2 components from e3
    dote1e3 = np.sum(e1*e3)
    dote2e3 = np.sum(e2*e3)
    
    e3 = e3 - dote1e3*e1 -dote2e3*e2
    
    # Normalize second tangent vector
    d = np.linalg.norm(e3)
    e3 = e3/d

#%%
# Okay, now we're ready to begin the estimation
LCE1 = 0.0
LCE2 = 0.0
LCE3 = 0.0

for n in range(nIterates):
    
#    for i in range(nItsPerPB):
#        t1 = np.linspace(0,dt,2)
    t1 = np.linspace(0,nItsPerPB*dt,nItsPerPB)
    x = odeint(f, x0, t1)
    x0 = x[-1]
    
    e1t = odeint(df, e1, t1, args=(x0[0], x0[1], x0[2]))
    e1 = e1t[-1]
    
    e2t = odeint(df, e2, t1, args=(x0[0], x0[1], x0[2]))
    e2 = e2t[-1]
    
    e3t = odeint(df, e3, t1, args=(x0[0], x0[1], x0[2]))
    e3 = e3t[-1]
        
    # Normalize the tangent vector
    d = np.linalg.norm(e1)
    e1 = e1/d
    
    # Accumulate the first tangent vector's length change factor
    LCE1 += np.log(d)

    # Pull-back: Remove any e1 component from e2
    dote1e2 = np.sum(e1*e2)
    e2 = e2 - dote1e2*e1
    
    # Normalize second tangent vector
    d = np.linalg.norm(e2)
    e2 = e2/d
    
    # Accumulate the second tangent vector's length change factor
    LCE2 += np.log(d)

    # Pull-back: Remove any e1 and e2 components from e3
    dote1e3 = np.sum(e1*e3)
    dote2e3 = np.sum(e2*e3)
    
    e3 = e3 - dote1e3*e1 -dote2e3*e2   
    
    # Normalize second tangent vector
    d = np.linalg.norm(e3)
    e3 = e3/d
    
    # Accumulate the third tangent vector's length change factor
    LCE3 += np.log(d)

#%%
# Convert to per-iterate, per-second LCEs and to base-2 logs
IntegrationTime = dt * float(nItsPerPB) * float(nIterates)
LCE1 = LCE1 / IntegrationTime
LCE2 = LCE2 / IntegrationTime
LCE3 = LCE3 / IntegrationTime
        
#%%
print(f'LCE1 = {round(LCE1,4)}')
print(f'LCE2 = {round(LCE2,4)}')
print(f'LCE3 = {round(LCE3,4)}')