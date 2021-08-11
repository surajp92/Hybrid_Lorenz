#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 10:28:20 2020

@author: suraj
"""

import numpy as np
from numpy.random import seed
seed(222)
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from mpl_toolkits.mplot3d import Axes3D

font = {'family' : 'Times New Roman',
        'size'   : 12}    
plt.rc('font', **font)

import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']


#%%

# weakly chaotic case parameters
data = np.load('l63_lstm_denkf_weak_chaos_1_50.npz')
utrue_w = data['utrue']
ulstm_w = data['ulstm']
ulstm_denkf_w = data['ua']
uobs_w = data['uobs']
tb = data['tb']
    
## strongly chaotic case parameters
#data = np.load('l63_lstm_denkf_strong_chaos_5_50.npz')
#utrue_s = data['utrue']
#ulstm_s = data['ulstm']
#ulstm_denkf_s = data['ua']
#uobs_s = data['uobs']
#tb = data['tb']

ne = 3 # number of states
npe = 10 # number of ensembeles
tmax = 30
dt = 0.001
nt = int(tmax/dt)
t = np.linspace(0,tmax,nt+1)

#%%
nrows = 3  ## rows of subplot
ncols = 2  ## colmns of subplot
label = [r'$X$',r'$Y$',r'$Z$']
fig, axs = plt.subplots(nrows, ncols, sharex=True,  figsize=(10,5.25))
fig.subplots_adjust(hspace=0.15)  

ntp = nt
           
for i in range(nrows):           
    l1, = axs[i,0].plot(t[:ntp],utrue_w[i,:ntp], color='k', linestyle='-',label='True')
    l2, = axs[i,0].plot(t[:ntp],ulstm_w[i,:ntp], color='r', linestyle='-.',label='LSTM')
    
    #axs[i,0].plot(t[:ntp],ulstm_denkf_w[i,:ntp], color='b', linestyle='--')
#    axs[i,0].plot(tb,uobs_w[i,:],'go',fillstyle='none', markersize=4,markeredgewidth=1)
    axs[i,0].set_xlim([0, t[ntp-1]])
    axs[i,0].set_ylabel(label[i])
#    axs[i,0].yaxis.set_major_locator(plt.MaxNLocator(3))
    axs[i,0].yaxis.set_label_coords(-0.10, 0.5)     
    
    axs[i,1].plot(t[:ntp],utrue_w[i,:ntp], color='k', linestyle='-')
    l3, = axs[i,1].plot(t[:ntp],ulstm_denkf_w[i,:ntp], color='b', linestyle='--',label='LSTM-DEnKF')
    l4, = axs[i,1].plot(tb,uobs_w[i,:],'go',fillstyle='none', markersize=5,markeredgewidth=1,
             label='Observations')
    
    #axs[i,1].plot(t[:ntp],ulstm_s[i,:ntp], color='r', linestyle='-.')
    axs[i,1].set_xlim([0, t[ntp-1]])
    axs[i,1].set_ylabel(label[i])
    
#    axs[i,0].yaxis.set_major_locator(plt.MaxNLocator(3))
    axs[i,1].yaxis.set_label_coords(-0.10, 0.5)     
    
    axs[i,1].axvspan(0, 20, alpha=0.1, color='orange')
    axs[i,0].axvspan(0, 20, alpha=0.1, color='orange')

axs[0,0].set_ylim([-25,25])
axs[0,1].set_ylim([-25,25])
axs[1,0].set_ylim([-25,25])
axs[1,1].set_ylim([-25,25])
axs[2,0].set_ylim([0,45])
axs[2,1].set_ylim([0,45])

handles, labels = axs[0,0].get_legend_handles_labels()
handles = [l1,l2,l3,l4]
line_labels = ['True','LSTM','LSTM-DEnKF','Observations']
#plt.figlegend(handles, labels, loc='lower center', ncol=4)

leg = fig.legend(handles, line_labels, loc='lower center', borderaxespad=-0.0, ncol=4, labelspacing=0.5)
  
axs[i,0].set_xlabel(r'$t$')
axs[i,1].set_xlabel(r'$t$')


#line_labels = ['Observation','True','Wrong','EnKF']
#plt.figlegend(line_labels,  loc = 'lower center', borderaxespad=0.0, ncol=4, labelspacing=0.5)
    
plt.show()
fig.savefig('l63_w_s_lstm_denkf_w50_long.pdf', bbox_inches="tight", dpi=300)
fig.savefig('l63_w_s_lstm_denkf_w50_long.png', bbox_inches="tight", dpi=300)

#%%
# weakly chaotic case parameters
data = np.load('l63_lstm_denkf_weak_chaos_1_100.npz')
utrue_w = data['utrue']
ulstm_w = data['ulstm']
ulstm_denkf_w = data['ua']
uobs_w = data['uobs']
tb = data['tb']
    
## strongly chaotic case parameters
#data = np.load('l63_lstm_denkf_strong_chaos_5_50.npz')
#utrue_s = data['utrue']
#ulstm_s = data['ulstm']
#ulstm_denkf_s = data['ua']
#uobs_s = data['uobs']
#tb = data['tb']

ne = 3 # number of states
npe = 10 # number of ensembeles
tmax = 30
dt = 0.001
nt = int(tmax/dt)
t = np.linspace(0,tmax,nt+1)

#%%
nrows = 3  ## rows of subplot
ncols = 2  ## colmns of subplot
label = [r'$X$',r'$Y$',r'$Z$']
fig, axs = plt.subplots(nrows, ncols, sharex=True,  figsize=(10,5.25))
fig.subplots_adjust(hspace=0.15)  

ntp = nt
           
for i in range(nrows):           
    l1, = axs[i,0].plot(t[:ntp],utrue_w[i,:ntp], color='k', linestyle='-',label='True')
    l2, = axs[i,0].plot(t[:ntp],ulstm_w[i,:ntp], color='r', linestyle='-.',label='LSTM')
    
    #axs[i,0].plot(t[:ntp],ulstm_denkf_w[i,:ntp], color='b', linestyle='--')
#    axs[i,0].plot(tb,uobs_w[i,:],'go',fillstyle='none', markersize=4,markeredgewidth=1)
    axs[i,0].set_xlim([0, t[ntp-1]])
    axs[i,0].set_ylabel(label[i])
#    axs[i,0].yaxis.set_major_locator(plt.MaxNLocator(3))
    axs[i,0].yaxis.set_label_coords(-0.10, 0.5)     
    
    axs[i,1].plot(t[:ntp],utrue_w[i,:ntp], color='k', linestyle='-')
    l3, = axs[i,1].plot(t[:ntp],ulstm_denkf_w[i,:ntp], color='b', linestyle='--',label='LSTM-DEnKF')
    l4, = axs[i,1].plot(tb,uobs_w[i,:],'go',fillstyle='none', markersize=5,markeredgewidth=1,label='Observations')
    
    #axs[i,1].plot(t[:ntp],ulstm_s[i,:ntp], color='r', linestyle='-.')
    axs[i,1].set_xlim([0, t[ntp-1]])
    axs[i,1].set_ylabel(label[i])
    
#    axs[i,0].yaxis.set_major_locator(plt.MaxNLocator(3))
    axs[i,1].yaxis.set_label_coords(-0.10, 0.5)     
    
    axs[i,1].axvspan(0, 20, alpha=0.1, color='orange')
    axs[i,0].axvspan(0, 20, alpha=0.1, color='orange')

axs[0,0].set_ylim([-25,25])
axs[0,1].set_ylim([-25,25])
axs[1,0].set_ylim([-25,25])
axs[1,1].set_ylim([-25,25])
axs[2,0].set_ylim([0,45])
axs[2,1].set_ylim([0,45])

handles, labels = axs[0,0].get_legend_handles_labels()
handles = [l1,l2,l3,l4]
line_labels = ['True','LSTM','LSTM-DEnKF','Observations']
#plt.figlegend(handles, labels, loc='lower center', ncol=4)

leg = fig.legend(handles, line_labels, loc='lower center', borderaxespad=-0.0, ncol=4, labelspacing=0.5)
  
axs[i,0].set_xlabel(r'$t$')
axs[i,1].set_xlabel(r'$t$')


#line_labels = ['Observation','True','Wrong','EnKF']
#plt.figlegend(line_labels,  loc = 'lower center', borderaxespad=0.0, ncol=4, labelspacing=0.5)
    
plt.show()
fig.savefig('l63_w_s_lstm_denkf_w100_long.pdf', bbox_inches="tight", dpi=300)
fig.savefig('l63_w_s_lstm_denkf_w100_long.png', bbox_inches="tight", dpi=300)
