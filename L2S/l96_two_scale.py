# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 17:41:10 2019

@author: Suraj
"""
#%% Import libraries
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
from scipy.integrate import simps
import pyfftw

from numpy.random import seed
seed(1)
#from tensorflow import set_random_seed
#set_random_seed(2)
import pandas as pd
import time as clck
import os

from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import ticker

font = {'family' : 'Times New Roman',
        'weight' : 'bold',
        'size'   : 14}    
plt.rc('font', **font)

import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']

#%%
#@jit(nopython=True, cache=True)
def l96_truth_step(X, Y, h, F, b, c):
    """
    Calculate the time increment in the X and Y variables for the Lorenz '96 "truth" model.

    Args:
        X (1D ndarray): Values of X variables at the current time step
        Y (1D ndarray): Values of Y variables at the current time step
        h (float): Coupling constant
        F (float): Forcing term
        b (float): Spatial scale ratio
        c (float): Time scale ratio

    Returns:
        dXdt (1D ndarray): Array of X increments, dYdt (1D ndarray): Array of Y increments
    """
    K = X.size
    J = Y.size // K
    dXdt = np.zeros(X.shape)
    dYdt = np.zeros(Y.shape)
    
    Yr = np.reshape(Y,[K,J])
    k = np.arange(K)
    dXdt[k] = -X[k - 1] * (X[k - 2] - X[(k + 1) % K]) - X[k] + F - h * c / b * np.sum(Yr, axis=1)
    
    j = np.arange(J*K)
    Xr = np.array([X,]*J).T
    Xr = Xr.flatten()
    dYdt[j] = -c * b * Y[(j + 1) % (J * K)] * (Y[(j + 2) % (J * K)] - Y[j-1]) - c * Y[j] + h * c / b * Xr[j]
    
#    for k in range(K):
#        dXdt[k] = -X[k - 1] * (X[k - 2] - X[(k + 1) % K]) - X[k] + F - h * c / b * np.sum(Y[k * J: (k + 1) * J])
#    for j in range(J * K):
#        dYdt[j] = -c * b * Y[(j + 1) % (J * K)] * (Y[(j + 2) % (J * K)] - Y[j-1]) - c * Y[j] + h * c / b * X[int(j / J)]

    return dXdt, dYdt

    
def rk4uc(ne,J,x,y,f,h,c,b,time_step):
    k1_dxdt = np.zeros(x.shape)
    k2_dxdt = np.zeros(x.shape)
    k3_dxdt = np.zeros(x.shape)
    k4_dxdt = np.zeros(x.shape)
    k1_dydt = np.zeros(y.shape)
    k2_dydt = np.zeros(y.shape)
    k3_dydt = np.zeros(y.shape)
    k4_dydt = np.zeros(y.shape)
    
    k1_dxdt[:], k1_dydt[:] = l96_truth_step(x, y, h, f, b, c)
    k2_dxdt[:], k2_dydt[:] = l96_truth_step(x + k1_dxdt * time_step / 2,
                                            y + k1_dydt * time_step / 2,
                                            h, f, b, c)
    k3_dxdt[:], k3_dydt[:] = l96_truth_step(x + k2_dxdt * time_step / 2,
                                            y + k2_dydt * time_step / 2,
                                            h, f, b, c)
    k4_dxdt[:], k4_dydt[:] = l96_truth_step(x + k3_dxdt * time_step,
                                            y + k3_dydt * time_step,
                                            h, f, b, c)
    x += (k1_dxdt + 2 * k2_dxdt + 2 * k3_dxdt + k4_dxdt) / 6 * time_step
    y += (k1_dydt + 2 * k2_dydt + 2 * k3_dydt + k4_dydt) / 6 * time_step
        
    return x,y

#def rk4uc(ne,J,x,y,f,h,c,b,time_step):
#    k1_dxdt = np.zeros(x.shape)
#    k2_dxdt = np.zeros(x.shape)
#    k3_dxdt = np.zeros(x.shape)
#    k4_dxdt = np.zeros(x.shape)
#    k1_dydt = np.zeros(y.shape)
#    k2_dydt = np.zeros(y.shape)
#    k3_dydt = np.zeros(y.shape)
#    k4_dydt = np.zeros(y.shape)
#    
#    k1_dxdt[:], k1_dydt[:] = l96_truth_step(x, y, h, f, b, c)
#    k2_dxdt[:], k2_dydt[:] = l96_truth_step(x + k1_dxdt * time_step / 2,
#                                            y,h, f, b, c)
#    k3_dxdt[:], k3_dydt[:] = l96_truth_step(x + k2_dxdt * time_step / 2,
#                                            y,h, f, b, c)
#    k4_dxdt[:], k4_dydt[:] = l96_truth_step(x + k3_dxdt * time_step,
#                                            y,h, f, b, c)
#    x += (k1_dxdt + 2 * k2_dxdt + 2 * k3_dxdt + k4_dxdt) / 6 * time_step
#    y += (k1_dydt + 2 * k2_dydt + 2 * k3_dydt + k4_dydt) / 6 * time_step
#        
#    return x,y


def dx_dt(ne,u,fr,dt):
    v = np.zeros(ne+3)
    v[2:ne+2] = u
    v[1] = v[ne+1]
    v[0] = v[ne]
    v[ne+2] = v[2]
    
    r = np.zeros(ne)
    
#    ys = np.sum(y,axis=1)
    r = -v[1:ne+1]*(v[0:ne] - v[3:ne+3]) - v[2:ne+2] + fr 
    
    return r

#%% Main program:
ne = 8
J = 32
fr = 20.0
c = 10.0
b = 10.0
h = 1.0

fact = 0.1
std = 1.0

dt = 0.001
tmax = 20.0
tinit = 5.0
ns = int(tinit/dt)
nt = int(tmax/dt)

nf = 10         # frequency of observation
nb = int(nt/nf) # number of observation time
oib = [nf*k for k in range(nb+1)]

u = np.zeros(ne)
utrue = np.zeros((ne,nt+1))
uinit = np.zeros((ne,ns+1))
ysuminit = np.zeros((ne,ns+1))
ysum = np.zeros((ne,nt+1))

ti = np.linspace(-tinit,0,ns+1)
t = np.linspace(0,tmax,nt+1)
tobs = np.linspace(0,tmax,nb+1)
x = np.linspace(1,ne,ne)

X,T = np.meshgrid(x,t,indexing='ij')
Xi,Ti = np.meshgrid(x,ti,indexing='ij')

#%%
#-----------------------------------------------------------------------------#
# generate true solution trajectory
#-----------------------------------------------------------------------------#
#u[:] = fr
#u[int(ne/2)-1] = fr + 0.01
#uinit[:,0] = u
#
#y = np.zeros(ne*J) #2*fact*fr*np.random.random_sample(ne*J) - fact*fr

u = np.zeros(ne)
u[0] = 1.0
uinit[:,0] = u

y = np.zeros(ne*J) #2*fact*fr*np.random.random_sample(ne*J) - fact*fr 
y[0] = 1.0

#%%

# generate initial condition at t = 0
for k in range(1,ns+1):
    un, yn = rk4uc(ne,J,u,y,fr,h,c,b,dt)
    uinit[:,k] = un
    ynr = np.reshape(yn,[ne,J])
    ysuminit[:,k] = np.sum(ynr,axis=1)
    u = np.copy(un)
    y = np.copy(yn)

# assign inital condition
u = uinit[:,-1]
utrue[:,0] = uinit[:,-1]
ysum[:,0] = ysuminit[:,-1]

# generate true forward solution
for k in range(1,nt+1):
    un, yn = rk4uc(ne,J,u,y,fr,h,c,b,dt)
    utrue[:,k] = un
    ynr = np.reshape(yn,[ne,J])
    ysum[:,k] = np.sum(ynr,axis=1)
    u = np.copy(un)
    y = np.copy(yn)

#%%
# assign inital condition
u = uinit[:,-1]
ysum_check = np.zeros((ne,nt+1))

y = np.zeros(ne*J)

# generate true forward solution
for k in range(nt):
    u = np.copy(utrue[:,k])
    dxdt = dx_dt(ne,u,fr,dt)
    ysum_check[:,k] = dxdt - (utrue[:,k+1] - utrue[:,k])/dt 
    
#    k1_dxdt = np.zeros(x.shape)
#    k2_dxdt = np.zeros(x.shape)
#    k3_dxdt = np.zeros(x.shape)
#    k4_dxdt = np.zeros(x.shape)
#    k1_dydt = np.zeros(y.shape)
#    k2_dydt = np.zeros(y.shape)
#    k3_dydt = np.zeros(y.shape)
#    k4_dydt = np.zeros(y.shape)
#    
#    x = np.copy(utrue[:,k])
#    
#    k1_dxdt[:], k1_dydt[:] = l96_truth_step(x, y, h, fr, b, c)
#    k2_dxdt[:], k2_dydt[:] = l96_truth_step(x + k1_dxdt * dt / 2,
#                                            y  ,
#                                            h, fr, b, c)
#    k3_dxdt[:], k3_dydt[:] = l96_truth_step(x + k2_dxdt * dt / 2,
#                                            y ,
#                                            h, fr, b, c)
#    k4_dxdt[:], k4_dydt[:] = l96_truth_step(x + k3_dxdt * dt,
#                                            y ,
#                                            h, fr, b, c)
#    dudt = (k1_dxdt + 2 * k2_dxdt + 2 * k3_dxdt + k4_dxdt) / 6 
#    dydt = (k1_dydt + 2 * k2_dydt + 2 * k3_dydt + k4_dydt) / 6 
#    
#    ysum_check[:,k] = dudt - (utrue[:,k+1] - utrue[:,k])/dt
        
#%%
np.savez(f'data_cyclic_{ne}_{J}.npz',utrue=utrue,ysum=ysum_check)
    
#%%
vmin = -15
vmax = 15
fig, ax = plt.subplots(2,1,figsize=(8,5))
cs = ax[0].contourf(Ti,Xi,uinit,40,cmap='jet',vmin=vmin,vmax=vmax)
m = plt.cm.ScalarMappable(cmap='jet')
m.set_array(uinit)
m.set_clim(vmin, vmax)
fig.colorbar(m,ax=ax[0],ticks=np.linspace(vmin, vmax, 6))

cs = ax[1].contourf(T,X,utrue,40,cmap='jet',vmin=vmin,vmax=vmax)
m = plt.cm.ScalarMappable(cmap='jet')
m.set_array(utrue)
m.set_clim(vmin, vmax)
fig.colorbar(m,ax=ax[1],ticks=np.linspace(vmin, vmax, 6))

fig.tight_layout()
plt.show()
fig.savefig(f'contour_cyclic_{ne}_{J}.png', dpi=200)

#%%
data = np.load(f'data_cyclic_{ne}_{J}.npz')
yp = data['ysum']
fig, ax = plt.subplots(4,1,sharex=True,figsize=(10,6))

n = [0,4]
for i in range(2):
    ax[i].plot(t,utrue[n[i],:],'k-')
    ax[i+2].plot(t,ysum[n[i],:],'r-')
#    ax[i+2].plot(t,yp[n[i],:],'g-')
    ax[i].set_xlim([0,tmax])
    ax[i].set_ylabel(r'$x_{'+str(n[i]+1)+'}$')
    ax[i+2].set_ylabel(r'$y_{'+str(n[i]+1)+'}$')
    
ax[i].set_xlabel(r'$t$')
line_labels = ['$X$','$Y$']
plt.figlegend( line_labels,  loc = 'lower center', borderaxespad=-0.2, ncol=4, labelspacing=0.)
fig.tight_layout()
plt.show() 
fig.savefig(f'ts_cyclic_{ne}_{J}.png', dpi=200)