#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 15:31:26 2021

@author: suraj
"""

import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import ticker
import matplotlib.gridspec as gridspec

font = {'family' : 'Times New Roman',
        'weight' : 'bold',
        'size'   : 14}    
plt.rc('font', **font)

import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']


#%%
ne = 8
J = 32
fr = 20.0
c = 10.0
b = 10.0
h = 1.0

dt = 0.001
nf = 10         # frequency of observation

tstart = 10.0
tend = 20.0
nts = int(tstart/dt) 
ntest = int((tend-tstart)/dt)
nb = int(nts/nf) # number of observation time

data = np.load('data_4_1.02.npz')
X = data['X'].T
T = data['T'].T
utrue = data['utrue'][:,nts:]
uw = data['uw']
ua = data['ua']
tobs = data['tobs'][nb:]
oin = data['oin']
uobs = data['uobs']
t = data['t']

#%%
fig, axes = plt.subplots(nrows=1, ncols=5,figsize=(13,6), sharey=True)


vmin_list = [-12,-12,-12,-12,-12]
vmax_list = [12,12,12,12,12]

f = [utrue, uw, ua, utrue-uw, utrue-ua]
titles = ['True', 'LSTM', 'LSTM-DEnKF', 'Error (LSTM)', 'Error (LSTM-DEnKF)']
   
for j in range(5):
    axes[j].set_title(titles[j],fontsize=14) 
#    if j == 0:
#        axes[i,j].set_ylabel('$y$') 
    m = f[j]
    levels = np.linspace(vmin_list[j], vmax_list[j], 61)
    cs = axes[j].contourf(X,T,m.T, 40, cmap = 'RdBu', levels=levels, extend='both', zorder=-20)
    axes[j].set_rasterization_zorder(-10)
    
    axes[j].set_xlabel('$\mathbf{X}$')
    if j == 0:
        axes[j].set_ylabel('MTU')
    axes[j].set_xticks([2,4,6,8])
#    axes[j].set_yticks([0,2,4,6])

fig.subplots_adjust(bottom=0.1)
cbar_ax = fig.add_axes([0.2, -0.05, 0.6, 0.05])    
cbarlabels = np.linspace(vmin_list[j], vmax_list[j], num=5, endpoint=True)
cbar = fig.colorbar(cs, cax=cbar_ax, shrink=0.8, orientation='horizontal')
cbar.set_ticks(cbarlabels)
cbar.ax.tick_params(labelsize=14)
cbar.set_ticklabels(['{:.1f}'.format(x) for x in cbarlabels])
    
plt.show()

fig.savefig('contour_field_l2s.pdf',  bbox_inches='tight',dpi=300)
fig.savefig('contour_field_l2s.png', bbox_inches='tight',dpi=300)

#%%
error_lstm = utrue - uw
error_da = utrue - ua

l2_lstm = np.linalg.norm(error_lstm, axis=0)/np.sqrt(ne)
l2_da = np.linalg.norm(error_da, axis=0)/np.sqrt(ne)

fig, ax = plt.subplots(3,1,sharex=True,figsize=(10,6))

n = [1,5,7]
for i in range(2):
#    if i == 0:
#        ax[i].plot(tobs,uobs[n[i],:],'ro',fillstyle='none', markersize=4,markeredgewidth=1)
    ax[i].plot(t,utrue[n[i],:],'k-', lw=2)
    ax[i].plot(t,uw[n[i],:],'r-.', lw=1.5)
    ax[i].plot(t,ua[n[i],:],'b--', lw=1.5)
    
    ax[i].set_xlim([tstart,tend])
    ax[i].set_ylabel(r'$X_{'+str(n[i]+1)+'}$',fontsize='16')

i = 2
ax[i].plot(t,l2_lstm,'r-.', lw=1.5, label='LSTM')
ax[i].plot(t,l2_da,'b--', lw=1.5, label='LSTM-DEnKF')
ax[i].set_ylabel(r'RMSE',fontsize=16)
ax[i].set_xlabel(r'MTU',fontsize=16)

line_labels = ['True','LSTM','LSTM-DEnKF']
plt.figlegend( line_labels,  loc = 'lower center', fontsize=16,
              borderaxespad=0.2, ncol=4, labelspacing=0.)
#fig.tight_layout()
fig.subplots_adjust(bottom=0.15)
plt.show() 
fig.savefig('ts_l2s.pdf',  bbox_inches='tight',dpi=300)
fig.savefig('ts_l2s.png', bbox_inches='tight',dpi=300)

#%%
fig,ax = plt.subplots(figsize=(12,10),sharex=True,constrained_layout=True)

AX = gridspec.GridSpec(5,4)
AX.update(wspace = 0.5, hspace = 0.5)
axs1  = plt.subplot(AX[0,0:2])
axs2 = plt.subplot(AX[0,2:])
axs3 = plt.subplot(AX[1,0:2])
axs4 = plt.subplot(AX[1,2:])
axs5 = plt.subplot(AX[2,0:2])
axs6 = plt.subplot(AX[2,2:])
axs7 = plt.subplot(AX[3,0:2])
axs8 = plt.subplot(AX[3,2:])
axs9  = plt.subplot(AX[4,1:3])

ax = [axs1, axs2, axs3, axs4, axs5, axs6, axs7, axs8, axs9]
for i in range(8):
    ax[i].plot(t,utrue[i,:],'k-', lw=2)
    ax[i].plot(t,uw[i,:],'r-.', lw=1.5)
    ax[i].plot(t,ua[i,:],'b--', lw=1.5)
    
    ax[i].set_xlim([tstart,tend])
    ax[i].set_ylim([-15,20])
    ax[i].set_yticks([-15,0,20])
    ax[i].set_ylabel(r'$X_{'+str(i+1)+'}$',fontsize=14)
#    ax[i].set_xticks([])
    ax[i].set_xlabel(r'MTU',fontsize=14)
    
i = 8
ax[i].plot(t,l2_lstm,'r-.', lw=1.5, label='LSTM')
ax[i].plot(t,l2_da,'b--', lw=1.5, label='DEnKF')
ax[i].set_ylabel(r'RMSE',fontsize=14)
ax[i].set_xlabel(r'MTU',fontsize=14)
ax[i].set_xlim([tstart,tend])
ax[i].set_ylim([-0,10])
ax[i].set_yticks([0,5,10])
    
line_labels = ['True','LSTM','LSTM-DEnKF']
plt.figlegend( line_labels,  loc = 'lower center', fontsize=16,
              borderaxespad=0.2, ncol=4, labelspacing=0.)
#fig.tight_layout()
fig.subplots_adjust(bottom=0.1)
plt.show()     
fig.savefig('ts_l2s_all.pdf',  bbox_inches='tight',dpi=300)
fig.savefig('ts_l2s_all.png', bbox_inches='tight',dpi=300)


#%%

fig, ax = plt.subplots(1,1,sharex=True,figsize=(10,4))
ax.plot(t,l2_lstm,'b', lw=2, label='LSTM')
ax.plot(t,l2_da,'g-', lw=2, label='DEnKF')

ax.set_ylabel(r'RMSE',fontsize=16)
ax.set_xlabel(r'MTU',fontsize=16)
ax.set_xlim([tstart,tend])
#ax.set_ylim([1e-2,1e2])

ax.legend()
#fig.tight_layout()
plt.show() 
fig.savefig('rmse_l2s.pdf',  bbox_inches='tight',dpi=300)
fig.savefig('rmse_l2s.png', bbox_inches='tight',dpi=300)


