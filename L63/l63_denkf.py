#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  5 12:06:30 2020

@author: suraj
"""

import numpy as np
from numpy.random import seed
seed(222)
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from mpl_toolkits.mplot3d import Axes3D

font = {'family' : 'Times New Roman',
        'size'   : 14}    
plt.rc('font', **font)

import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']

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
#a = 10.0
#b = 28.0
#c = 8/3
#u0 = np.array([-9.42, -9.43, 28.3]) # weakly chaotic case
#u0 = np.array([1.0,1.0,1.0]) # weakly chaotic case
    
# strongly chaotic case parameters
a = 16.0
b = 120.1
c = 4.0
u0 = np.array([22.8, 35.7, 114.9]) # strongly chaotic case
figname = 'l63_denkf_strong_chaos.png'

ne = 3 # number of states
npe = 3 # number of ensembeles
tmax = 10
dt = 0.001
nt = int(tmax/dt)
t = np.linspace(0,tmax,nt+1)

freq_obs = 20         # frequency of observation
nb = int(nt/freq_obs) # number of observation time
tb = np.linspace(0,tmax,nb+1)

utrue = np.zeros((3,nt+1))
k = 0
utrue[:,k] = u0
k = 1
utrue[:,k] = utrue[:,k-1] + dt*rhs(utrue[:,k-1],a,b,c)
k = 2
utrue[:,k] = utrue[:,k-1] + dt*rhs(utrue[:,k-1],a,b,c)

for k in range(3,nt+1):
    r1 = rhs(utrue[:,k-1],a,b,c)
    r2 = rhs(utrue[:,k-2],a,b,c)
    r3 = rhs(utrue[:,k-3],a,b,c)
    temp= (23/12) * r1 - (16/12) * r2 + (5/12) * r3
    utrue[:,k] = utrue[:,k-1] + dt*temp 


#%%
#-----------------------------------------------------------------------------#
# generate observations
#-----------------------------------------------------------------------------#
mean = 0.0
sd2 = 1.0e1 # added noise (variance)
sd1 = np.sqrt(sd2) # added noise (standard deviation)

oib = [freq_obs*k for k in range(nb+1)]

uobs = utrue[:,oib] + np.random.normal(mean,sd1,[ne,nb+1])

#-----------------------------------------------------------------------------#
# generate erroneous soltions trajectory
#-----------------------------------------------------------------------------#
uw = np.zeros((ne,nt+1))
k = 0
si2 = 1.0e0 # initial condition (variance)
si1 = np.sqrt(si2) # initial condition (standard deviation)

k = 0
u = utrue[:,k] + np.random.normal(mean,si1,ne)
uw[:,k] = u
k = 1
uw[:,k] = uw[:,k-1] + dt*rhs(uw[:,k-1],a,b,c)
k = 2
uw[:,k] = uw[:,k-1] + dt*rhs(uw[:,k-1],a,b,c)

for k in range(1,nt+1):
    r1 = rhs(uw[:,k-1],a,b,c)
    r2 = rhs(uw[:,k-2],a,b,c)
    r3 = rhs(uw[:,k-3],a,b,c)
    temp= (23/12) * r1 - (16/12) * r2 + (5/12) * r3
    uw[:,k] = uw[:,k-1] + dt*temp 

#%%
nrows = 3  ## rows of subplot
ncols = 1  ## colmns of subplot
label = [r'$X$',r'$Y$',r'$Z$']
fig, axs = plt.subplots(nrows, ncols, sharex=True,  figsize=(8,5))
fig.subplots_adjust(hspace=0.15)  
           
for i in range(0,nrows):           
    axs[i].plot(t,utrue[i,:], color='k', linestyle='-', label='True')
    axs[i].plot(t,uw[i,:], color='g', linestyle='-', label='Wrong')
    axs[i].plot(tb,uobs[i,:],'ro',fillstyle='none', markersize=4,markeredgewidth=1)
    axs[i].set_xlim([0, t[-1]])
    axs[i].set_ylabel(label[i], fontsize = 14)
    axs[i].yaxis.set_major_locator(plt.MaxNLocator(3))
    axs[i].yaxis.set_label_coords(-0.05, 0.5)       
  
axs[i].set_xlabel(r'$t$', fontsize = 14)
line_labels = ['True','Wrong','Observations']
#line_labels = ['Observation','True','Wrong','EnKF']
plt.figlegend( line_labels,  loc = 'lower center', borderaxespad=-0.2, ncol=4, labelspacing=0.)
    
fig.tight_layout()    
plt.show()

#%%
#-----------------------------------------------------------------------------#
# EnKF model
#-----------------------------------------------------------------------------#    

# number of observation vector
me = 3
freq = int(ne/me)
oin = [freq*i-1 for i in range(1,me+1)]
roin = np.int32(np.linspace(0,me-1,me))

dh = np.zeros((me,ne))
dh[roin,oin] = 1.0

H = np.zeros((me,ne))
H[roin,oin] = 1.0

#%%
cn = 1.0/np.sqrt(npe-1)
lambd = 1.0

z = np.zeros((me,nb+1)) # all observations
ue = np.zeros((ne,npe,nt+1)) # all ensambles
ua = np.zeros((ne,nt+1)) # mean analyssi solution (to store)
uf = np.zeros(ne)    
Af = np.zeros((ne,npe))   # Af data

for k in range(nb+1):
    z[:,k] = uobs[oin,k]

# initial ensemble
k = 0
se2 = 0.0 #np.sqrt(sd2)
se1 = np.sqrt(se2)

for n in range(npe):
    ue[:,n,k] = uw[:,k] + np.random.normal(mean,si1,ne)       
    
ua[:,k] = np.sum(ue[:,:,k],axis=1)
ua[:,k] = ua[:,k]/npe

k = 1
for n in range(npe):
    ue[:,n,k] = ue[:,n,k-1] + dt*rhs(ue[:,n,k-1],a,b,c)

ua[:,k] = np.sum(ue[:,:,k],axis=1)
ua[:,k] = ua[:,k]/npe

k = 2
for n in range(npe):
    ue[:,n,k] = ue[:,n,k-1] + dt*rhs(ue[:,n,k-1],a,b,c)    

ua[:,k] = np.sum(ue[:,:,k],axis=1)
ua[:,k] = ua[:,k]/npe

kobs = 1

for k in range(3,nt+1):
    # forecast afor all ensemble fields
    for n in range(npe):
        r1 = rhs(ue[:,n,k-1],a,b,c)
        r2 = rhs(ue[:,n,k-2],a,b,c)
        r3 = rhs(ue[:,n,k-3],a,b,c)
        temp= (23/12) * r1 - (16/12) * r2 + (5/12) * r3
        ue[:,n,k] = ue[:,n,k-1] + dt*temp 
    
    # mean analysis for plotting
    ua[:,k] = np.sum(ue[:,:,k],axis=1)
    ua[:,k] = ua[:,k]/npe
    
    if k == oib[kobs]:
        # compute mean of the forecast fields
        uf[:] = np.sum(ue[:,:,k],axis=1)   
        uf[:] = uf[:]/npe
        
        # compute Af dat
        for n in range(npe):
            Af[:,n] = ue[:,n,k] - uf[:]
            
        pf = Af @ Af.T
        pf[:,:] = pf[:,:]/(npe-1)
        
        dp = dh @ pf
        cc = dp @ dh.T     

        for i in range(me):
            cc[i,i] = cc[i,i] + sd2     
        
        ph = pf @ dh.T
        
        ci = np.linalg.pinv(cc) # ci: inverse of cc matrix
        
        km = ph @ ci # compute Kalman gain
        
        # analysis update    
        kmd = km @ (z[:,kobs] - uf[oin])
        ua[:,k] = uf[:] + kmd[:]
        
        # ensemble correction
        ha = dh @ Af
        
        ue[:,:,k] = Af[:,:] - 0.5*(km @ dh @ Af) + ua[:,k].reshape(-1,1)
        
        #multiplicative inflation (optional): set lambda=1.0 for no inflation
        #ue[:,:,k] = ua[:,k].reshape(-1,1) + lambd*(ue[:,:,k] - ua[:,k].reshape(-1,1))
        
        kobs = kobs+1

#%%
nrows = 3  ## rows of subplot
ncols = 1  ## colmns of subplot
label = [r'$X$',r'$Y$',r'$Z$']
fig, axs = plt.subplots(nrows, ncols, sharex=True,  figsize=(10,6))
fig.subplots_adjust(hspace=0.15)  
           
for i in range(0,nrows):           
    axs[i].plot(t,utrue[i,:], color='k', linestyle='-', label='True')
    axs[i].plot(t,uw[i,:], color='g', linestyle='-', label='Wrong')
    axs[i].plot(t,ua[i,:], color='b', linestyle='--', label='Analysis')
#    axs[i].plot(tb,uobs[i,:],'ro',fillstyle='none', markersize=4,markeredgewidth=1)
    axs[i].set_xlim([0, t[-1]])
    axs[i].set_ylabel(label[i], fontsize = 14)
    axs[i].yaxis.set_major_locator(plt.MaxNLocator(3))
    axs[i].yaxis.set_label_coords(-0.05, 0.5)       
  
axs[i].set_xlabel(r'$t$', fontsize = 14)
line_labels = ['True','Wrong','Analysis']#,'Observations']
#line_labels = ['Observation','True','Wrong','EnKF']
plt.figlegend( line_labels,  loc = 'lower center', borderaxespad=-0.2, ncol=4, labelspacing=0.)
    
fig.tight_layout()    
plt.show()
fig.savefig(figname)