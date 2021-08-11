#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 11 10:34:23 2020

@author: suraj
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
from scipy.integrate import simps
import pyfftw

from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

import time as clck
import os

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import load_model
import keras.backend as K
K.set_floatx('float64')

from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib
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
    SS_res =  K.sum(K.square(y_true-y_pred ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

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

#%%
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

#%% main parameters:
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
nt = int(tmax/dt)

t = np.linspace(0,tmax,nt+1)
x = np.linspace(1,ne,ne)
X,T = np.meshgrid(x,t,indexing='ij')

ttrain = 10.0
ntrain = int(ttrain/dt)

nf = 1         # frequency of training samples
nb = int(ntrain/nf) # number of training time
oib = [nf*k for k in range(nb+1)]

training = True
lookback = 6

#%%
data = np.load(f'../data_cyclic_{ne}_{J}.npz')
utrue = data['utrue']
ysum = data['ysum']

#%%
features_train = utrue[:,oib].T 
labels_train = ysum[:,oib].T

xtrain, ytrain = create_training_data_lstm(features_train,labels_train,lookback)

#%%
# Scaling data
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

x_train, x_valid, y_train, y_valid = train_test_split(xtrain_sc, ytrain_sc, 
                                                      test_size=0.2 , shuffle= True)

#%%
mx,lx,nx = x_train.shape # m is number of training samples, n is number of output features [i.e., n=nr]
my,ny = y_train.shape
nh = 80
model_name = f'lstm_bestmodel_{ne}_{J}_3_{nh}.hd5'
    
if training:
    # create the LSTM architecture
    model = Sequential()
    #model.add(Dropout(0.2))
    model.add(LSTM(nh, input_shape=(lookback, nx), return_sequences=True, activation='relu'))
    model.add(LSTM(nh, input_shape=(lookback, nx), return_sequences=True, activation='relu'))
#    model.add(LSTM(80, input_shape=(lookback, nx), return_sequences=True, activation='relu'))
    #model.add(LSTM(40, input_shape=(lookback, n+1), return_sequences=True, activation='relu', kernel_initializer='uniform'))
    model.add(LSTM(nh, input_shape=(lookback, nx), activation='relu'))
    model.add(Dense(ny))
    
    # compile model
    #model.compile(loss='mean_absolute_percentage_error', optimizer='adam', metrics=['mean_absolute_percentage_error'])
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])
    
    # run the model
    history = model.fit(x_train, y_train, epochs=1600, batch_size=128, validation_data= (x_valid,y_valid))
    #history = model.fit(xtrain, ytrain, epochs=600, batch_size=32, validation_split=0.2)
    
    # evaluate the model
    scores = model.evaluate(x_train, y_train, verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    plt.figure()
    epochs = range(1, len(loss) + 1)
    plt.semilogy(epochs, loss, 'b', label='Training loss')
    plt.semilogy(epochs, val_loss, 'r', label='Validationloss')
    plt.title('Training and validation loss')
    plt.legend()
    filename = f'loss_{ne}_{J}_3_{nh}.png'
    plt.savefig(filename, dpi = 200)
    plt.show()
    
    # Save the model
    filename = model_name
    model.save(filename)

mx,lx,nx = xtrain.shape # m is number of training samples, n is number of output features [i.e., n=nr]
my,ny = ytrain.shape

#%%     
model = load_model(model_name)

for ts in [0.0,10.0]:
    tstart = ts
    tend = tstart + 10.0
    nts = int(tstart/dt) 
    ntest = int((tend-tstart)/dt)
    
    uml = np.zeros((ne,ntest+1))
    ysml = np.zeros((ne,ntest+1))
    
    uml[:,:lookback] = utrue[:,nts:nts+lookback]
    ysml[:,:lookback] = ysum[:,nts:nts+lookback]
    
    xtest = np.zeros((1,lookback,nx))
    for j in range(lookback):
        xtest[0,j,:] = uml[:,j] 
        
    # generate true forward solution
    for k in range(lookback,ntest+1):
        u = uml[:,k-1]
        ys = ysml[:,k-1]
        un = rk4uc(ne,J,u,ys,fr,h,c,b,dt)
        uml[:,k] = un
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
    
    cs = ax[1].contourf(T,X,uml,40,cmap='jet',vmin=vmin,vmax=vmax)
    m = plt.cm.ScalarMappable(cmap='jet')
    m.set_array(utrue)
    m.set_clim(vmin, vmax)
    fig.colorbar(m,ax=ax[1],ticks=np.linspace(vmin, vmax, 6))
    
    diff = utrue[:,nts:nts+ntest+1] - uml
    cs = ax[2].contourf(T,X,diff,40,cmap='jet',vmin=vmin,vmax=vmax)
    m = plt.cm.ScalarMappable(cmap='jet')
    m.set_array(utrue)
    m.set_clim(vmin, vmax)
    fig.colorbar(m,ax=ax[2],ticks=np.linspace(vmin, vmax, 6))
    
    fig.tight_layout()
    plt.show()
    fig.savefig('lstm_'+str(tstart)+'_'+str(tend)+'.png',dpi=300)
    
    l2norm = np.array([tstart, tend, np.linalg.norm(diff)])
    f=open(f'l2norm_{tstart}_{tend}.dat','ab')
    np.savetxt(f,l2norm.reshape([1,-1]))    
    f.close()










