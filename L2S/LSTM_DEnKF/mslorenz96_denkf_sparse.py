#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 22:15:00 2020

@author: suraj
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
from scipy.integrate import simps
import pyfftw

from numpy.random import seed
seed(1)
#from tensorflow import set_random_seed
#set_random_seed(2)

import time as clck
import os

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import load_model
import keras.backend as kb
kb.set_floatx('float64')

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

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
def coeff_determination(y_true, y_pred):
    SS_res =  kb.sum(kb.square(y_true-y_pred ))
    SS_tot = kb.sum(kb.square( y_true - kb.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + kb.epsilon()) )

def create_training_data_lstm(features,labels,lookback):
    # m : number of snapshots 
    # n: number of states
    m, n = features.shape
    ytrain = [labels[i,:] for i in range(lookback-1,m)]
    ytrain = np.array(ytrain)    
    
    xtrain = np.zeros((m-lookback+1,lookback,n))
    for i in range(m-lookback+1):
        a = features[i,:]
        for j in range(1,lookback):
            a = np.vstack((a,features[i+j,:]))
        xtrain[i,:,:] = a
    return xtrain , ytrain

def rhs_dx_dt(ne,u,ys,fr,h,c,dt):
    v = np.zeros(ne+3)
    v[2:ne+2] = u
    v[1] = v[ne+1]
    v[0] = v[ne]
    v[ne+2] = v[2]
    
    r = np.zeros(ne)
    
#    ys = np.sum(y,axis=1)
    r = -v[1:ne+1]*(v[0:ne] - v[3:ne+3]) - v[2:ne+2] + fr - (h*c/b)*ys
    
    return r*dt
    
def rk4uc(ne,J,u,y,fr,h,c,b,dt):
    k1x = rhs_dx_dt(ne,u,y,fr,h,c,dt)
    k2x = rhs_dx_dt(ne,u+0.5*k1x,y,fr,h,c,dt)
    k3x = rhs_dx_dt(ne,u+0.5*k2x,y,fr,h,c,dt)
    k4x = rhs_dx_dt(ne,u+0.5*k3x,y,fr,h,c,dt)
    
    un = u + (k1x + 2.0*(k2x + k3x) + k4x)/6.0
    
    return un
       
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

ttrain_start = 0.0
ttrain_end = 10.0
ntrains = int(ttrain_start/dt) 
ntrain = int((ttrain_end-ttrain_start)/dt)

tstart = 10.0
tend = 20.0
nts = int(tstart/dt) 
ntest = int((tend-tstart)/dt)

training = True
lookback = 6

#%%    
data = np.load(f'../data_cyclic_{ne}_{J}.npz')
utrue = data['utrue']
ysum = data['ysum']

mean = 0.0
sd2 = 1.0e0 # added noise (variance)
sd1 = np.sqrt(sd2) # added noise (standard deviation)

uobsfull = utrue[:,:] + np.random.normal(mean,sd1,[ne,nt+1])

features_train = utrue[:,ntrains:ntrains+ntrain+1].T
labels_train = ysum[:,ntrains:ntrains+ntrain+1].T

xtrain, ytrain = create_training_data_lstm(features_train,labels_train,lookback)

#%%
data = np.copy(xtrain)
labels = np.copy(ytrain)

p,q,r = data.shape
data2d = data.reshape(p*q,r)

scalerIn = MinMaxScaler(feature_range=(-1,1))
scalerIn = scalerIn.fit(data2d)
data2d = scalerIn.transform(data2d)
data = data2d.reshape(p,q,r)

scalerOut = MinMaxScaler(feature_range=(-1,1))
scalerOut = scalerOut.fit(labels)
labels = scalerOut.transform(labels)

xtrain_sc = np.copy(data)
ytrain_sc = np.copy(labels)

x_train, x_valid, y_train, y_valid = train_test_split(xtrain_sc, ytrain_sc, test_size=0.2 , shuffle= True)

mx,lx,nx = xtrain.shape 
my,ny = ytrain.shape

#%%
#-----------------------------------------------------------------------------#
# generate erroneous soltions trajectory
#-----------------------------------------------------------------------------#
model = load_model('lstm_bestmodel_8_32_3_80.hd5')

nt = ntest
nf = 10         # frequency of observation
nb = int(ntest/nf) # number of observation time
oib = [nf*k for k in range(nb+1)]
uobs = uobsfull[:,nts:nts+ntest+1][:,oib]

mean = 0.0
si2 = 1.0e-2
si1 = np.sqrt(si2)

uw = np.zeros((ne,ntest+1))
ysml = np.zeros((ne,ntest+1))

uw[:,:lookback] = utrue[:,nts:nts+lookback]
ysml[:,:lookback] = ysum[:,nts:nts+lookback]

xtest = np.zeros((1,lookback,nx))
for j in range(lookback):
    xtest[0,j,:] = uw[:,j] 
    
# generate true forward solution
for k in range(lookback,ntest+1):
    u = uw[:,k-1]
    ys = ysml[:,k-1]
    un = rk4uc(ne,J,u,ys,fr,h,c,b,dt)
    uw[:,k] = un
#    for j in range(lookback-1):
#        xtest[0,j,:] = xtest[0,j+1,:]
    xtest[0,:-1,:] = xtest[0,1:,:]
    xtest[0,lookback-1,:] = un 
    xtest_sc = scalerIn.transform(xtest[0])
    xtest_sc = xtest_sc.reshape(1,lookback,nx)
    ytest_sc = model.predict(xtest_sc)
    ytest = scalerOut.inverse_transform(ytest_sc)
    ysml[:,k] = ytest

t = np.linspace(tstart,tend,ntest+1)
x = np.linspace(1,ne,ne)
X,T = np.meshgrid(x,t,indexing='ij')

vmin = -15
vmax = 15
fig, ax = plt.subplots(3,1,figsize=(6,7.5))
cs = ax[0].contourf(T,X,utrue[:,nts:nts+ntest+1],40,cmap='jet',vmin=vmin,vmax=vmax)
m = plt.cm.ScalarMappable(cmap='jet')
m.set_array(utrue)
m.set_clim(vmin, vmax)
fig.colorbar(m,ax=ax[0],ticks=np.linspace(vmin, vmax, 6))

cs = ax[1].contourf(T,X,uw,40,cmap='jet',vmin=vmin,vmax=vmax)
m = plt.cm.ScalarMappable(cmap='jet')
m.set_array(utrue)
m.set_clim(vmin, vmax)
fig.colorbar(m,ax=ax[1],ticks=np.linspace(vmin, vmax, 6))

diff = utrue[:,nts:nts+ntest+1] - uw
cs = ax[2].contourf(T,X,diff,40,cmap='jet',vmin=vmin,vmax=vmax)
m = plt.cm.ScalarMappable(cmap='jet')
m.set_array(utrue)
m.set_clim(vmin, vmax)
fig.colorbar(m,ax=ax[2],ticks=np.linspace(vmin, vmax, 6))

fig.tight_layout()
plt.show()
fig.savefig(f'f_lstm_{tstart}_{tend}.png',dpi=200)   

print(np.linalg.norm(diff))

np.savez('data_lstm.npz',T=T,X=X,utrue=utrue,uw=uw)

#%%
#-----------------------------------------------------------------------------#
# EnKF model
#-----------------------------------------------------------------------------#    
# number of observation vector
me = 4
freq = int(ne/me)
oin = [freq*i-1 for i in range(1,me+1)]
roin = np.int32(np.linspace(0,me-1,me))
print(oin)

dh = np.zeros((me,ne))
dh[roin,oin] = 1.0

H = np.zeros((me,ne))
H[roin,oin] = 1.0

#%%
# number of ensemble 
npe = 10
cn = 1.0/np.sqrt(npe-1)
lambd = 1.02

z = np.zeros((me,nb+1))
#zf = np.zeros((me,npe,nb+1))
DhX = np.zeros((me,npe))
DhXm = np.zeros(me)

ua = np.zeros((ne,nt+1)) # mean analyssi solution (to store)
uf = np.zeros(ne)        # mean forecast
sc = np.zeros((ne,npe))   # square-root of the covariance matrix
Af = np.zeros((ne,npe))   # Af data
ue = np.zeros((ne,npe,nt+1)) # all ensambles
yse = np.zeros((ne,npe,nt+1)) # all ensambles
ph = np.zeros((ne,me))

km = np.zeros((ne,me))
kmd = np.zeros((ne,npe))

cc = np.zeros((me,me))
ci = np.zeros((me,me))

for k in range(nb+1):
    z[:,k] = uobs[oin,k]
#    for n in range(npe):
#        zf[:,n,k] = z[:,k] + np.random.normal(mean,sd1,me)

#%%
# initial condition for all ensembles
se2 = 1.0e-0 #np.sqrt(sd2)
se1 = np.sqrt(se2)

for n in range(npe):
    for k in range(lookback):
        ue[:,n,k] = uw[:,k] + np.random.normal(mean,si1,ne)       
        yse[:,n,k] = ysum[:,k]

xtest = np.zeros((npe,lookback,nx))
for n in range(npe):
    for k in range(lookback):
        xtest[n,k,:] = ue[:,n,k] 

for k in range(lookback):
    ua[:,k] = np.sum(ue[:,:,k],axis=1)
    ua[:,k] = ua[:,k]/npe

#%%
kobs = 1

# RK4 scheme
for k in range(1,nt+1):
    # forecast afor all ensemble fields
    for n in range(npe):
        u[:] = ue[:,n,k-1]
        ys[:] = yse[:,n,k-1]
        un = rk4uc(ne,J,u,ys,fr,h,c,b,dt)
        ue[:,n,k] = un #+ np.random.normal(mean,se1,ne)
        xtest[n,:-1,:] = xtest[n,1:,:]
        xtest[n,lookback-1,:] = un
        xtest_sc = scalerIn.transform(xtest[n])
        xtest_sc = xtest_sc.reshape(1,lookback,nx)
        ytest_sc = model.predict(xtest_sc)
        ytest = scalerOut.inverse_transform(ytest_sc)
        yse[:,n,k] = ytest
            
    # mean analysis for plotting
    ua[:,k] = np.sum(ue[:,:,k],axis=1)
    ua[:,k] = ua[:,k]/npe
    
    if k == oib[kobs]:
        print(k)
        # compute mean of the forecast fields
        uf[:] = np.sum(ue[:,:,k],axis=1)   
        uf[:] = uf[:]/npe
        
        # compute Af dat
        for n in range(npe):
            Af[:,n] = ue[:,n,k] - uf[:]
        
        da = dh @ Af
        
        cc = da @ da.T/(npe-1)  
        
        for i in range(me):
            cc[i,i] = cc[i,i] + sd2 
        
        ci = np.linalg.pinv(cc)
        
        km = Af @ da.T @ ci/(npe-1)
                
        # analysis update    
        kmd = km @ (z[:,kobs] - uf[oin])
        ua[:,k] = uf[:] + kmd[:]
        
        # ensemble correction
        ha = dh @ Af
        
        ue[:,:,k] = Af[:,:] - 0.5*(km @ dh @ Af) + ua[:,k].reshape(-1,1)
        
        #multiplicative inflation (optional): set lambda=1.0 for no inflation
        ue[:,:,k] = ua[:,k].reshape(-1,1) + lambd*(ue[:,:,k] - ua[:,k].reshape(-1,1))
        
        kobs = kobs+1

np.savez(f'data_{me}_{lambd}.npz',t=t,tobs=tobs,T=T,X=X,utrue=utrue,uobs=uobs,uw=uw,ua=ua,oin=oin)
    
#%%

fig, ax = plt.subplots(3,1,sharex=True,figsize=(6,5))

n = [1,4,7]
for i in range(3):
    ax[i].plot(t,utrue[n[i],nts:nts+ntest+1],'k-')
    ax[i].plot(t,uw[n[i],:],'b--')
    ax[i].plot(t,ua[n[i],:],'g-.')
#    if i == 0:
#        ax[i].plot(tobs,uobs[n[i],:],'ro',fillstyle='none', markersize=6,markeredgewidth=2)

    ax[i].set_xlim([t[0],t[-1]])
    ax[i].set_ylabel(r'$u_{'+str(n[i]+1)+'}$')

ax[i].set_xlabel(r'$t$')
line_labels = ['True','Wrong','EnKF']
plt.figlegend( line_labels,  loc = 'lower center', borderaxespad=-0.2, ncol=4, labelspacing=0.)
fig.tight_layout()
plt.show() 
fig.savefig(f'm_lstmda_{me}_{tstart}_{tend}_{lambd}.png', dpi=200)

#%%
vmin = -15
vmax = 15
fig, ax = plt.subplots(3,1,figsize=(6,7.5))

cs = ax[0].contourf(T,X,utrue[:,nts:nts+ntest+1],40,cmap='jet',vmin=vmin,vmax=vmax)
m = plt.cm.ScalarMappable(cmap='jet')
m.set_array(utrue)
m.set_clim(vmin, vmax)
fig.colorbar(m,ax=ax[0],ticks=np.linspace(vmin, vmax, 6))
ax[0].set_title('True')

cs = ax[1].contourf(T,X,ua,40,cmap='jet',vmin=vmin,vmax=vmax)
m = plt.cm.ScalarMappable(cmap='jet')
m.set_array(ua)
m.set_clim(vmin, vmax)
fig.colorbar(m,ax=ax[1],ticks=np.linspace(vmin, vmax, 6))
ax[1].set_title('EnKF')

diff = ua - utrue[:,nts:nts+ntest+1]
cs = ax[2].contourf(T,X,diff,40,cmap='jet',vmin=vmin,vmax=vmax)
m = plt.cm.ScalarMappable(cmap='jet')
m.set_array(ua)
m.set_clim(vmin, vmax)
fig.colorbar(m,ax=ax[2],ticks=np.linspace(vmin, vmax, 6))
ax[2].set_title('Difference')

fig.tight_layout()
plt.show() 
fig.savefig(f'f_lstmda_{me}_{tstart}_{tend}_{lambd}.png',dpi=200)   

print(np.linalg.norm(diff))


l2norm = np.array([me,nf,sd2,np.linalg.norm(diff)])
f=open(f'l2norm_{me}_{lambd}.dat','ab')
np.savetxt(f,l2norm.reshape([1,-1]))    
f.close()

    

 
























































