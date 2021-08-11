#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  5 10:46:00 2020

@author: suraj
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from mpl_toolkits.mplot3d import Axes3D

#%%
def rhs(u,a,b,c): 
    x,y,z = u
    r = np.zeros(3)
    
    r[0] = -a*(x - y)
    r[1] = x*(b - z) - y
    r[2] = x*y - c*z
    
    return r
    
    
def rk4(dt,u,a,b,c):
    r1 = rhs(u,a,b,c)
    k1 = dt*r1
    
    r2 = rhs(u,a,b,c)
    k2 = dt*r2
    
    r3 = rhs(u,a,b,c)
    k3 = dt*r3
    
    r4 = rhs(u,a,b,c)
    k4 = dt*r4
    
    un = u + (k1 + 2.0*(k2 + k3) + k4)/6.0
    
    return un

def ft(dt,u,a,b,c):
    r = rhs(u,a,b,c)
    
    un = u + dt*r
    
    return un

#%%
# weakly chaotic case parameters
a = 10.0
b = 28.0
c = 8/3
u0 = np.array([-9.42, -9.43, 28.3]) # weakly chaotic case

# strongly chaotic case parameters
#a = 16.0
#b = 120.1
#c = 4.0
#u0 = np.array([22.8, 35.7, 114.9]) # strongly chaotic case

ne = 3
tmax = 30
dt = 0.001
nt = int(tmax/dt)
t = np.linspace(0,tmax,nt+1)

uall_ft = np.zeros((ne,nt+1))
uall_rk4 = np.zeros((ne,nt+1))

k = 0
uall_ft[:,k] = u0
uall_rk4[:,k] = u0

for k in range(1,nt+1):
    up = uall_ft[:,k-1]
    un = ft(dt,up,a,b,c)
    uall_ft[:,k] = un
    
    up = uall_rk4[:,k-1]
    un = rk4(dt,up,a,b,c)
    uall_rk4[:,k] = un

uall_ab3 = np.zeros((ne,nt+1))
k = 0
uall_ab3[:,k] = u0
k = 1
uall_ab3[:,k] = uall_ab3[:,k-1] + dt*rhs(uall_ab3[:,k-1],a,b,c)
k = 2
uall_ab3[:,k] = uall_ab3[:,k-1] + dt*rhs(uall_ab3[:,k-1],a,b,c)

temp_matrix = np.zeros((ne,nt+1))

for k in range(3,nt+1):
    r1 = rhs(uall_ab3[:,k-1],a,b,c)
    r2 = rhs(uall_ab3[:,k-2],a,b,c)
    r3 = rhs(uall_ab3[:,k-3],a,b,c)
    temp= (23/12) * r1 - (16/12) * r2 + (5/12) * r3
    temp_matrix[:,k] = temp
    uall_ab3[:,k] = uall_ab3[:,k-1] + dt*temp 

np.savez('data_weak_chaos_long_lead_time', t = t, u = uall_ab3, a = a, b = b, c = c, u0 = u0)
#np.savez('data_strong_chaos', t = t, u = uall_ab3, a = a, b = b, c = c, u0 = u0)


#%%
#rho = 28.0
#sigma = 10.0
#beta = 8.0 / 3.0
#
#def f(state, t):
#    x, y, z = state  # Unpack the state vector
#    return sigma * (y - x), x * (rho - z) - y, x * y - beta * z  # Derivatives
#
#state0 = [1.0, 1.0, 1.0]
#ts = np.arange(0.0, 40.0, 0.01)
#
#states = odeint(f, state0, ts)

#%%
nrows = 3  ## rows of subplot
ncols = 1  ## colmns of subplot
label = [r'$X$',r'$Y$',r'$Z$']
fig, axs = plt.subplots(nrows, ncols, sharex=True,  figsize=(8,5))
fig.subplots_adjust(hspace=0.15)  
           
for i in range(0,nrows):           
    axs[i].plot(t,uall_ab3[i,:], color='black', linestyle='-', label=r'$y_'+str(2*i+1)+'$'+' (True)')
    axs[i].set_xlim([0, t[-1]])
    axs[i].set_ylabel(label[i], fontsize = 14)
    axs[i].yaxis.set_major_locator(plt.MaxNLocator(3))
    axs[i].yaxis.set_label_coords(-0.05, 0.5)       
  
axs[i].set_xlabel(r'$t$', fontsize = 14)
    
fig.tight_layout()    
plt.show()
fig.savefig('weak_2d.png', dpi=200)

#%% Plot the Lorenz attractor using a Matplotlib 3D projection
fig = plt.figure()
ax = fig.gca(projection='3d')

# Make the line multi-coloured by plotting it in segments of length s which
# change in colour across the whole time series.
s = 100
c = np.linspace(0,1,nt+1)
for i in range(0,nt-s,s):
    ax.plot(uall_ab3[0,i:i+s+1], uall_ab3[1,i:i+s+1], uall_ab3[2,i:i+s+1], color=(1,c[i],0), alpha=0.8)
#    ax.plot(uall_rk4[i:i+s+1,0], uall_rk4[i:i+s+1,1], uall_rk4[i:i+s+1,2], color=(1,c[i],0), alpha=0.8)
    
#    ax.plot(uall_ab3[i:i+s+1,0], uall_ab3[i:i+s+1,1], uall_ab3[i:i+s+1,2], color='k', alpha=0.8)
#    ax.plot(uall_rk4[i:i+s+1,0], uall_rk4[i:i+s+1,1], uall_rk4[i:i+s+1,2], color='r', alpha=0.8)

# Remove all the axis clutter, leaving just the curve.
#ax.set_axis_off()

plt.show()
fig.savefig('weak_3d.png', dpi=200)
