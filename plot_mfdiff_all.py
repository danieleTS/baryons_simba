"""
    Compute and plot the relative differences in the HMF and BMF given by
    different Simba models, for different redshift.
"""

import matplotlib.pyplot as plt
import numpy as np
import pygad as pg
import yt
import h5py
import caesar
import os
import sys
import math
from pygadgetreader import *

Mbins = 20
logM_max = 14
logM_min = 9
dlogM = (logM_max - logM_min)/Mbins

savedir =  '/disk04/sorini/outputs/plots/'


colours = ['green', 'red', 'blue', 'purple', 'orange']
lstyle = ['-', '-.', '--', ':']
WIND = ['s50j7k', 's50nox', 's50nojet', 's50noagn', 's50nofb']
PHASE = ['total', 'baryon']
x_lab=[r'$\log(M_{\rm h}/\rm M_{\odot})$', r'$\log(M_{\rm b}/\rm M_{\odot})$', r'$\log(M_*/\rm M_{\odot})$']
y_lab = [r'$\rm HMF/HMF_{\rm fid}-1$', r'$\rm BMF/BMF_{\rm fid}-1$']
MODEL = ['m50n512']*len(PHASE)
SNAP=['051', '078', '105', '151']
mod_lab = [r'Simba $50 \, h^{-1} \, \rm cMpc$', 'No-X-ray', 'No-jet', 'No-AGN', 'No-feedback']
y_min = [-0.6, -1]
y_max = [0.7, 5.5]


fig = plt.figure(figsize=(6*len(SNAP)+1, 6*len(PHASE)))
fig.subplots_adjust(wspace=0, hspace=0)


plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=14)


for ip in np.arange(len(PHASE)):
    phase = PHASE[ip]
    model = MODEL[ip]
    for iz in np.arange(len(SNAP)):
        snap = SNAP[iz]
        plt_ID = ip * len(SNAP) + iz + 1
        ax = plt.subplot(len(PHASE), len(SNAP), plt_ID)
        ax.set_xlim(9, 14.5)
        ax.set_ylim(y_min[ip], y_max[ip])
        ticks = [9, 10, 11, 12, 13, 14]
        ax.set_xticks(ticks)
        if (ip == len(PHASE)-1):
            ax.set_xlabel(r'$\log(M/\rm M_{\odot})$', fontsize=30)
            ax.tick_params(labelsize=22)
        else:
            ax.tick_params(labelbottom=None)
        
        if (iz == 0):
            ax.set_ylabel(y_lab[ip], fontsize=30, x=1.05)
            ax.tick_params(labelsize=22)
        else:
            ax.tick_params(labelleft=None)
        
        for i in np.arange(len(WIND)):
            wind = WIND[i]
            infile = '/disk04/rad/sim/%s/%s/Groups/%s_%03d.hdf5' %  (model,wind,model,snap)
            snap_file = '/disk04/rad/sim/%s/%s/snap_%s_%03d.hdf5' %  (model,wind,model,snap)
            s = pg.Snap(snap_file)
            ds = yt.load(snap_file)
            sim = caesar.load(infile)
            h = sim.simulation.hubble_constant
            Lbox = readheader(snap_file, 'boxsize') # kpc/h
            Lbox = Lbox/1000./h # cMpc
            print('Lbox=%s'%(Lbox))
            M = np.asarray([i.masses[phase].to('Msun') for i in sim.halos])
            z_sim = sim.simulation.redshift
            logM = np.log10(M) # M is already expressed in units of MSun
            N_haloes, M_edges = np.histogram(logM, bins=Mbins, range=(logM_min, logM_max), density=False)
            M_bins = 0.5*(M_edges[1:]+M_edges[:-1])
            binwidth = M_bins[1]-M_bins[0]
            mf = N_haloes/(binwidth*Lbox**3)
            if (i==0):
                mf_ref = mf
            # the mask mf>0 removes bins where there are no halos from the plot
            # this may happen in high-mass bins if the binning is too fine
            ax.plot(M_bins[mf>0], mf[mf>0]/mf_ref[mf>0]-1, color=colours[i], ls='-', lw=2)
                
            if (iz==0):
                ax.axvline(-1, color=colours[i], ls='-', lw=2, label=mod_lab[i])
                if (ip==1):
                    ax.legend(loc='best', fontsize=20, ncol=1, framealpha=0)
    
            if (ip==0):
                plt.title(r'$z=%g$'%(np.round(z_sim, 1)), fontsize=30, y=1.05)
fformat = 'pdf'
plt.savefig(savedir+'mf_diff.%s'%(fformat), format=fformat, bbox_inches = 'tight')
