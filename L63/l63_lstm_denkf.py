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

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import load_model
import keras.backend as K
K.set_floatx('float64')

from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split

font = {'family' : 'Times New Roman',
        'size'   : 14}    
plt.rc('font', **font)

import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']

#%%
def create_training_data_lstm(features,labels, m, n, lookback):
    # m : number of snapshots 
    # n: number of states
    ytrain = [labels[i,:] for i in range(lookback-1,m)]
    ytrain = np.array(ytrain)    
    
    xtrain = np.zeros((m-lookback+1,lookback,n))
    for i in range(lookback-1,m):
        a = features[i,:]
        for j in range(1,lookback):
            a = np.vstack((a,features[i-j,:]))
        xtrain[i-lookback+1,:,:] = a
    return xtrain , ytrain

def rhs(u,a,b,c): 
    x,y,z = u
    r = np.zeros(3)
    
    r[0] = -a*(x - y)
    r[1] = x*(b - z) - y
    r[2] = x*y - c*z
#    r[2] = 0.0
    
    return r

def rhs_lstm(u,a,b,c): 
    x,y,z = u
    r = np.zeros(3)
    
    r[0] = -a*(x - y)
    r[1] = x*(b - z) - y
#    r[2] = x*y - c*z
    r[2] = 0.0
    
    return r

#%%
# weakly chaotic case parameters
a = 10.0
b = 28.0
c = 8/3
u0 = np.array([-9.42, -9.43, 28.3]) # weakly chaotic case
modelname = 'l63_lstm_weak_chaos.hd5'
data = np.load('data_weak_chaos_long_lead_time.npz')
figname = 'l63_lstm_denkf_weak_chaos.png'
    
# strongly chaotic case parameters
#a = 16.0
#b = 120.1
#c = 4.0
#u0 = np.array([22.8, 35.7, 114.9]) # strongly chaotic case
##modelname = 'l63_lstm_strong_chaos.hd5'
#modelname = 'l63_lstm_strong_chaos.hd5'
#data = np.load('data_strong_chaos.npz')
#figname = 'l63_lstm_denkf_strong_chaos'

ne = 3 # number of states
npe = 10 # number of ensembeles
tmax = 30
ttrain = 20.0
dt = 0.001
nt = int(tmax/dt)
t = np.linspace(0,tmax,nt+1)

freq_obs = 100         # frequency of observation
nb = int(nt/freq_obs) # number of observation time
tb = np.linspace(0,tmax,nb+1)


#%%

nttrain = int(ttrain/dt)

Training = True
lookback = 6

utrue = data['u']   

ut = utrue.T

features = utrue[:,:nttrain].T

#labels = (utrue[2,1:nttrain+1].T - utrue[2,0:nttrain].T)/dt
labels = np.zeros((nttrain+1,1))
for k in range(3,nttrain+2):
    r1 = utrue[0,k-1]*utrue[1,k-1] - c*utrue[2,k-1]
    r2 = utrue[0,k-2]*utrue[1,k-2] - c*utrue[2,k-2]
    r3 = utrue[0,k-3]*utrue[1,k-3] - c*utrue[2,k-3]
    
    labels[k-1,0] = (23.0/12.0)*r1 - (16.0/12.0)*r2 + (5.0/12.0)*r3
    
labels = labels.reshape([-1,1])
slope = (utrue[:,1:] - utrue[:,:-1])/dt
z_slope = slope[-1,:nttrain+1]

#%%
diff = z_slope - labels.flatten()

#%%
xt, yt = create_training_data_lstm(features, labels, nttrain, ne, lookback)

data = np.copy(xt) # modified GP as the input data
labels = np.copy(yt)

# Scaling data
p,q,r = data.shape
data2d = data.reshape(p*q,r)


scalerIn = MinMaxScaler(feature_range=(-1,1))
scalerIn = scalerIn.fit(data2d)
data2d = scalerIn.transform(data2d)
data = data2d.reshape(p,q,r)

scalerOut = MinMaxScaler(feature_range=(-1,1))
scalerOut = scalerOut.fit(labels)
labels = scalerOut.transform(labels)

#%%
#-----------------------------------------------------------------------------#
# generate observations
#-----------------------------------------------------------------------------#
mean = 0.0
sd2 = 1.0e0 # added noise (variance)
sd1 = np.sqrt(sd2) # added noise (standard deviation)
sd2mag = 1.0

id_name = f'l63_lstm_denkf_weak_chaos_1_{freq_obs}'

oib = [freq_obs*k for k in range(nb+1)]

uobs = utrue[:,oib] + sd2mag*np.random.normal(mean,sd1,[ne,nb+1])

#-----------------------------------------------------------------------------#
# generate erroneous soltions trajectory
#-----------------------------------------------------------------------------#
uw = np.zeros((ne,nt+1))
k = 0
si2 = 1.0e0 # initial condition (variance)
si1 = np.sqrt(si2) # initial condition (standard deviation)

model = load_model(modelname)

for k in range(lookback):
    uw[:,k] = utrue[:,k]
    
xtest = np.zeros((1,lookback,ne))
for n in range(0,lookback):
    xtest[0,n,:] = uw[:,lookback-1-n]

for k in range(lookback,nt+1):
    r1 = rhs_lstm(uw[:,k-1],a,b,c)
    r2 = rhs_lstm(uw[:,k-2],a,b,c)
    r3 = rhs_lstm(uw[:,k-3],a,b,c)
    temp= (23/12) * r1 - (16/12) * r2 + (5/12) * r3
    uw[:2,k] = uw[:2,k-1] + dt*temp[:2] 
    
    xtest_sc = scalerIn.transform(xtest[0])
    xtest_sc = xtest_sc.reshape(1,lookback,ne)
    ytest_sc = model.predict(xtest_sc)
    ytest = scalerOut.inverse_transform(ytest_sc) # residual/ correction
    uw[2,k] = uw[2,k-1] + ytest*dt
    
    for q in range(lookback-1,0,-1):
        xtest[0,q,:] = xtest[0,q-1,:]
    
    xtest[0,0,:] = uw[:,k]

#%%
nrows = 3  ## rows of subplot
ncols = 1  ## colmns of subplot
label = [r'$X$',r'$Y$',r'$Z$']
fig, axs = plt.subplots(nrows, ncols, sharex=True,  figsize=(8,5))
fig.subplots_adjust(hspace=0.15)  
           
for i in range(0,nrows):           
    axs[i].plot(t,utrue[i,:nt+1], color='k', linestyle='-', label='True')
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
model = load_model(modelname)

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
se2 = 0.0 #np.sqrt(sd2)
se1 = np.sqrt(se2)

for k in range(lookback):
    for n in range(npe):
        ue[:,n,k] = uw[:,k] + np.random.normal(mean,si1,ne)       
        
    ua[:,k] = np.sum(ue[:,:,k],axis=1)
    ua[:,k] = ua[:,k]/npe

xtest = np.zeros((npe,lookback,ne))
for n in range(npe):
    for k in range(0,lookback):
        xtest[n,k,:] = ue[:,n,lookback-1-k]
    
kobs = 1

for k in range(lookback,nt+1):
    # forecast afor all ensemble fields
    for n in range(npe):
        
        r1 = rhs_lstm(ue[:,n,k-1],a,b,c)
        r2 = rhs_lstm(ue[:,n,k-2],a,b,c)
        r3 = rhs_lstm(ue[:,n,k-3],a,b,c)
        temp= (23/12) * r1 - (16/12) * r2 + (5/12) * r3
        ue[:2,n,k] = ue[:2,n,k-1] + dt*temp[:2]  
        
        xtest_sc = scalerIn.transform(xtest[n])
        xtest_sc = xtest_sc.reshape(1,lookback,ne)
        ytest_sc = model.predict(xtest_sc)
        ytest = scalerOut.inverse_transform(ytest_sc) # residual/ correction
        ue[2,n,k] = ue[2,n,k-1] + dt*ytest
        
        for q in range(lookback-1,0,-1):
            xtest[n,q,:] = xtest[n,q-1,:]
        
        xtest[n,0,:] = ue[:,n,k]
    
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
fig, axs = plt.subplots(nrows, ncols, sharex=True,  figsize=(10,6.25))
fig.subplots_adjust(hspace=0.15)  

ntp = nt
           
for i in range(0,nrows):           
    axs[i].plot(t[:ntp],utrue[i,:ntp], color='k', linestyle='-', label='True')
    axs[i].plot(t[:ntp],uw[i,:ntp], color='g', linestyle='-', label='Wrong')
    axs[i].plot(t[:ntp],ua[i,:ntp], color='b', linestyle='--', label='Analysis')
    axs[i].plot(tb,uobs[i,:],'ro',fillstyle='none', markersize=4,markeredgewidth=1)
    axs[i].set_xlim([0, t[ntp-1]])
    axs[i].set_ylabel(label[i], fontsize = 14)
    axs[i].yaxis.set_major_locator(plt.MaxNLocator(3))
    axs[i].yaxis.set_label_coords(-0.05, 0.5)       
  
axs[i].set_xlabel(r'$t$', fontsize = 14)
line_labels = ['True','LSTM','Analysis','Observations']
#line_labels = ['Observation','True','Wrong','EnKF']
plt.figlegend(line_labels,  loc = 'lower center', borderaxespad=-0.5, ncol=4, labelspacing=0.)
    
fig.tight_layout()    
plt.show()
fig.savefig(id_name+'.png', dpi=300)

#%%
np.savez(id_name, utrue = utrue, ulstm = uw, ua = ua, uobs = uobs, t = t, tb = tb)