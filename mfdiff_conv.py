"""
    Convergence test for halo mass function and baryon mass function
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
from plotting_methods import *
from pygadgetreader import *

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=14)
palette_name = 'tol'
dm = 0.25 # dex
logr_min = -2 # distance in units of Rvir
logr_max = math.log10(5)

G = 6.67e-11 # SI units
Mpc2km = 3.0856776e19
km2Mpc = 1./Mpc2km

Mbins = 20
logM_max = 14
logM_min = 9
dlogM = (logM_max - logM_min)/Mbins

savedir =  '/disk04/sorini/outputs/plots/'

colours = ['green', 'green', 'brown', 'magenta']
lstyle = ['-', '-', '--', ':']
stats = ['median', 'percentile_25_75', 'std', 'cosmic_median', 'cosmic_std']

PHASE = ['total', 'baryon']
x_lab=[r'$\log(M_{\rm h}/\rm M_{\odot})$', r'$\log(M_{\rm b}/\rm M_{\odot})$', r'$\log(M_{\rm gas}/\rm M_{\odot})$']
y_lab = [r'$\rm HMF/HMF_{\rm fid 100}-1$', r'$\rm BMF/BMF_{\rm fid 100}-1$', r'$\rm GMF/GMF_{\rm fid 100}-1$']
MODEL = ['m100n1024', 'm50n512', 'm25n256', 'm25n512']
WIND = ['s50', 's50', 's50', 's50']
SNAP=['051']
mod_lab = [r'Simba $100 \, h^{-1} \, \rm cMpc$', r'Simba $50 \, h^{-1} \, \rm cMpc$', r'Simba $25 \, h^{-1} \, \rm cMpc$', r'Simba High-res.',]
y_min = [-0.6, -1, -1]
y_max = [0.7, 5.5, 4.5]


fig = plt.figure(figsize=(6*len(SNAP), 3*len(PHASE)))
fig.subplots_adjust(wspace=0, hspace=0.4)

for ip in np.arange(len(PHASE)):
    phase = PHASE[ip]
    for iz in np.arange(len(SNAP)):
        snap = SNAP[iz]
        plt_ID = ip * len(SNAP) + iz + 1
        ax = plt.subplot(len(PHASE), len(SNAP), plt_ID)
        ax.set_xlim(9, 14.5)
        ticks = [9, 10, 11, 12, 13, 14]
        ax.set_xticks(ticks)
        ax.set_xlabel(x_lab[ip], fontsize=18)
        ax.tick_params(labelsize=18)
        
        if (iz == 0):
            ax.set_ylabel(y_lab[ip], fontsize=18, x=1.05)
            ax.tick_params(labelsize=18)
        else:
            ax.tick_params(labelleft=None)
        
        for i in np.arange(len(WIND)):
            model = MODEL[i]
            wind = WIND[i]
            infile = '/disk04/rad/sim/%s/%s/Groups/%s_%s.hdf5' % (model,wind,model,snap)
            snap_file = '/disk04/rad/sim/%s/%s/snap_%s_%s.hdf5' % (model,wind,model,snap)
            s = pg.Snap(snap_file)
            ds = yt.load(snap_file)
            sim = caesar.load(infile)
            h = sim.simulation.hubble_constant
            Lbox = readheader(snap_file, 'boxsize') # kpc/h
            Lbox = Lbox/1000./h # cMpc
            print('Lbox=%s'%(Lbox))
            M = np.asarray([i.masses[phase].to('Msun') for i in sim.halos])
            pos = np.asarray([i.pos.in_units('kpccm') for i in sim.halos])
            pos = pos/1000. # cMpc
            z_sim = sim.simulation.redshift
            logM = np.log10(M) # M is already expressed in units of MSun
            N_haloes, M_edges = np.histogram(logM, bins=Mbins, range=(logM_min, logM_max), density=False)
            M_bins = 0.5*(M_edges[1:]+M_edges[:-1])
            binwidth = M_bins[1]-M_bins[0]
            mf = N_haloes/(binwidth*Lbox**3)
            
            if (i==0):
                mf_ref = mf
                mf_err = np.zeros(len(M_bins))
                for k in np.arange(len(M_bins)):
                    mask = (logM<=M_edges[k+1]) & (logM>M_edges[k])
                if (len(pos[mask])>0):
                    _, mf_err[k] = N_scatter(np.ones(len(M[mask])), pos[mask], Lbox)
                    mf_err[k] = (8./(7.*binwidth*Lbox**3))*mf_err[k]
                else:
                    mf_err[k] = np.nan
        
            if (i==0):
                ax.axhline(0, c='black', ls=':', lw=1)
            else:
                # the mask mf>0 removes bins where there are no halos from the plot
                # this may happen in high-mass bins if the binning is too fine
                ax.plot(M_bins[mf>0], mf[mf>0]/mf_ref[mf>0]-1, color=colours[i], ls=lstyle[i], lw=2)
                
            if (iz==0):
                if (i==0):
                    continue
                else:
                    ax.axvline(-1, color=colours[i], ls=lstyle[i], lw=2, label=mod_lab[i])
                if (ip==1):
                    ax.legend(loc='upper left', fontsize=16, ncol=1, framealpha=0)

fformat = 'pdf'
plt.savefig(savedir+'mfdiff_conv_%s.%s'%(snap,fformat), format=fformat, bbox_inches = 'tight')
plt.clf()
