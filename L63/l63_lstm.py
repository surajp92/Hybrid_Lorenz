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
from tensorflow import set_random_seed
set_random_seed(2)
import pandas as pd
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
#-----------------------------------------------------------------------------#
# Neural network Routines
#-----------------------------------------------------------------------------#
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

def create_training_data_lstm_r(features,labels, m, n, lookback):
    ytrain = [labels[i,:] for i in range(lookback-1,m)]
    ytrain = np.array(ytrain) 
    
    xtrain = np.zeros((m-lookback+1,lookback,n))
    for i in range(m-lookback+1):
        a = features[i,:]
        for j in range(1,lookback):
            a = np.vstack((a,features[i+j,:]))
        xtrain[i,:,:] = a
    return xtrain , ytrain

def deploy_input(features, m, n, lookback):
    xtest = np.zeros((m-lookback+1,lookback,n))
    for i in range(m-lookback+1):
        a = features[i,:]
        for j in range(1,lookback):
            a = np.vstack((a,features[i+j,:]))
        xtest[i,:,:] = a
    return xtest 

def rhs(u,a,b,c): 
    x,y,z = u
    r = np.zeros(3)
    
    r[0] = -a*(x - y)
    r[1] = x*(b - z) - y
#    r[2] = x*y - c*z
    r[2] = 0.0
    
    return r

#%% Main program:
    
# weakly chaotic case parameters
a = 10.0
b = 28.0
c = 8/3
u0 = np.array([-9.42, -9.43, 28.3]) # weakly chaotic case
modelname = 'l63_lstm_weak_chaos.hd5'
data = np.load('data_weak_chaos_long_lead_time.npz')
figname = 'l63_lstm_weak_chaos.png'
ncells = 20
    
# strongly chaotic case parameters
#a = 16.0
#b = 120.1
#c = 4.0
#u0 = np.array([22.8, 35.7, 114.9]) # strongly chaotic case
#modelname = 'l63_lstm_strong_chaos_6.hd5'
#data = np.load('data_strong_chaos.npz')
#figname = 'l63_lstm_strong_chaos.png'
#ncells = 20

ne = 3
tmax = 30.0
ttrain = 20.0
dt = 0.001
nt = int(tmax/dt)
t = np.linspace(0,tmax,nt+1)

nttrain = int(ttrain/dt)

Training = True
lookback = 6

utrue = data['u']   

ut = utrue.T

features = utrue[:,:nttrain].T

labels = np.zeros((nttrain+1,1))
for k in range(3,nttrain+1):
    r1 = utrue[0,k-1]*utrue[1,k-1] - c*utrue[2,k-1]
    r2 = utrue[0,k-2]*utrue[1,k-2] - c*utrue[2,k-2]
    r3 = utrue[0,k-3]*utrue[1,k-3] - c*utrue[2,k-3]
    
    labels[k-1,0] = (23.0/12.0)*r1 - (16.0/12.0)*r2 + (5.0/12.0)*r3
        
#    labels = (utrue[2,1:nttrain+1].T - utrue[2,0:nttrain].T)/dt
    
labels = labels.reshape([-1,1])

#%%
xt, yt = create_training_data_lstm(features, labels, nttrain, ne, lookback)
#xtr, ytr = create_training_data_lstm_r(features, labels, nttrain+1, ne-1, lookback)

#%%
data = np.copy(xt) # modified GP as the input data
labels = np.copy(yt)
        
#%%
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

xtrain = data
ytrain = labels

xtrain, xvalid, ytrain, yvalid = train_test_split(data, labels, test_size=0.2 , shuffle= True)

#%%
mx,lx,nx = xtrain.shape # m is number of training samples, n is number of output features [i.e., n=nr]
my,ny = ytrain.shape
   
if Training:
    # create the LSTM architecture
    model = Sequential()
    #model.add(Dropout(0.2))
    model.add(LSTM(ncells, input_shape=(lookback, nx), return_sequences=True, activation='relu'))
    #model.add(LSTM(80, input_shape=(lookback, nx), return_sequences=True, activation='relu'))
    #model.add(LSTM(40, input_shape=(lookback, n+1), return_sequences=True, activation='relu', kernel_initializer='uniform'))
    model.add(LSTM(ncells, input_shape=(lookback, nx), activation='relu'))
    model.add(Dense(ny))
    
    # compile model
    #model.compile(loss='mean_absolute_percentage_error', optimizer='adam', metrics=['mean_absolute_percentage_error'])
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])
    
    # run the model
    history = model.fit(xtrain, ytrain, epochs=600, batch_size=32, validation_data= (xvalid,yvalid))
    #history = model.fit(xtrain, ytrain, epochs=600, batch_size=32, validation_split=0.2)
    
    # evaluate the model
    scores = model.evaluate(xtrain, ytrain, verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    plt.figure()
    epochs = range(1, len(loss) + 1)
    plt.semilogy(epochs, loss, 'b', label='Training loss')
    plt.semilogy(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    filename = 'loss.png'
    plt.savefig(filename, dpi = 400)
    plt.show()
    
    # Save the model
    model.save(modelname)
    
    np.savez('loss_history.npz',epochs=epochs,loss=loss,val_loss=val_loss)

mx,lx,nx = xtrain.shape # m is number of training samples, n is number of output features [i.e., n=nr]
my,ny = ytrain.shape

#%%    
# deployment dynamically 
 
ne = 3
tmax = 30.0
ttrain = 20.0
dt = 0.001
nt = int(tmax/dt)
t = np.linspace(0,tmax,nt+1)

    
model = load_model(modelname)
ulstm = np.zeros((ne,nt+1)) 

for k in range(lookback):
    ulstm[:,k] = utrue[:,k]

#k = 0
#ulstm[:,k] = utrue[:,k]
#k = 1
#ulstm[:,k] = utrue[:,k] 
#k = 2
#ulstm[:,k] = utrue[:,k] 
#k = 3
#ulstm[:,k] = utrue[:,k] 

xtest = np.zeros((1,lookback,ne))
for n in range(0,lookback):
    xtest[0,n,:] = ulstm[:,lookback-1-n]

for k in range(lookback,nt+1):
    print(t[k])
    r1 = rhs(ulstm[:,k-1],a,b,c)
    r2 = rhs(ulstm[:,k-2],a,b,c)
    r3 = rhs(ulstm[:,k-3],a,b,c)
    temp= (23/12) * r1 - (16/12) * r2 + (5/12) * r3
    ulstm[:2,k] = ulstm[:2,k-1] + dt*temp[:2] 
        
    xtest_sc = scalerIn.transform(xtest[0])
    xtest_sc = xtest_sc.reshape(1,lookback,ne)
    ytest_sc = model.predict(xtest_sc)
    ytest = scalerOut.inverse_transform(ytest_sc) # residual/ correction
    ulstm[2,k] = ulstm[2,k-1] + ytest*dt
    
    for q in range(lookback-1,0,-1):
        xtest[0,q,:] = xtest[0,q-1,:]
    
    xtest[0,0,:] = ulstm[:,k]
    
    
#%%
nrows = 3  ## rows of subplot
ncols = 1  ## colmns of subplot
label = [r'$X$',r'$Y$',r'$Z$']
fig, axs = plt.subplots(nrows, ncols, sharex=True,  figsize=(8,5))
fig.subplots_adjust(hspace=0.15)  
           
ntp = nt+1
for i in range(0,nrows):           
    axs[i].plot(t[:ntp],utrue[i,:ntp], color='k', linestyle='-', label='True')
    axs[i].plot(t[:ntp],ulstm[i,:ntp], color='r', linestyle='--', label='Wrong')
#    axs[i].plot(tb,uobs[i,:],'ro',fillstyle='none', markersize=4,markeredgewidth=1)
    axs[i].set_xlim([0, t[ntp-1]])
    axs[i].set_ylabel(label[i], fontsize = 14)
    axs[i].yaxis.set_major_locator(plt.MaxNLocator(3))
    axs[i].yaxis.set_label_coords(-0.05, 0.5)       
  
axs[i].set_xlabel(r'$t$', fontsize = 14)
line_labels = ['True','LSTM']
#line_labels = ['Observation','True','Wrong','EnKF']
plt.figlegend( line_labels,  loc = 'lower center', borderaxespad=-0.2, ncol=4, labelspacing=0.)
    
fig.tight_layout()    
plt.show()
fig.savefig(figname, dpi=300)        
