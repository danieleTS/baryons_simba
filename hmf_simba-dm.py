"""
   Compute and plot the halo mass function of the DMO and fiducial run at different redshift,
   and for different definitions of the halo mass.
"""

import pygad as pg
import yt
import math
from yt.units.yt_array import YTQuantity
import caesar
import h5py
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from astropy.io import ascii
from astropy.cosmology import FlatLambdaCDM
from pygadgetreader import *
from plotting_methods import *

Mbins = 20
logM_max = 14
logM_min = 9
dlogM = (logM_max - logM_min)/Mbins
Mmin = 9.7e9 # MSun  # Mass of 100 DM particles
logMmin = math.log10(Mmin)

pdfpath='/disk04/sorini/outputs/plots/'
Mpc2km = 3.0856776e19
km2Mpc = 1./Mpc2km
MODEL=['m100n1024', 'm100n1024']
WIND = [' ', 's50']
colour = ['black','teal']
lstyle=['-','--', '-.', ':']
lwlist=[2,2,2,2,2]
MVIR = ['total', 'm200c', 'm500c', 'm2500c']
DELTA = [' ', 200, 200, 500, 500]
lab = ['Simba-Dark', r'Simba $100$ $\mathrm{cMpc}/h$']
Delta_lab = ['FOF', r'$\Delta_{\rm c} = 200$', r'$\Delta_{\rm c} = 500$', r'$\Delta_{\rm c} = 2500$']

fig1 = plt.figure(figsize=(13, 18))
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
gs = gridspec.GridSpec(ncols=2, nrows=6, width_ratios=[1,1], height_ratios=[2,1,2,1,2,1],wspace=0,hspace=0)
func_ID=[0,1,4,5,8,9]
diff_ID=[2,3,6,7,10,11]
SNAP_ar = [151, 105, 78, 51]

logM_ST = np.linspace(8, 16, 1000)
M_ST = 10**logM_ST # MSun

for ii in np.arange(len(SNAP_ar)):
    M_bins = np.zeros((len(WIND), len(DELTA), Mbins), dtype=np.float64)
    hmf = np.zeros((len(WIND), len(DELTA), Mbins), dtype=np.float64)
    HMF = np.zeros((len(WIND), len(DELTA), Mbins), dtype=np.float64)
    SNAP = SNAP_ar[ii]
    fID = func_ID[ii]
    ax=fig1.add_subplot(gs[fID])

    for j in np.arange(len(WIND)):
        # index j -> different models (e.g. hydro & DMO)
        if (j==0):
            infile='/disk04/sorini/inputs/simba_DMS/Groups/Caesar_snap_%03d.hdf5' % (SNAP)
            snap_file = '/disk04/sorini/inputs/simba_DMS/snap_%s_%03d.hdf5' % (MODEL[j],SNAP)
        else:
            infile = '/disk04/rad/sim/%s/%s/Groups/%s_%03d.hdf5' % (MODEL[j],WIND[j],MODEL[j],SNAP)
            snap_file = '/disk04/rad/sim/%s/%s/snap_%s_%03d.hdf5' % (MODEL[j],WIND[j],MODEL[j],SNAP)
                
        s = pg.Snap(snap_file)
        ds = yt.load(snap_file)
        sim = caesar.load(infile)
        h = sim.simulation.hubble_constant
        z_sim = sim.simulation.redshift
        Lbox = readheader(snap_file, 'boxsize') # kpc/h
        Lbox = Lbox/1000./h # cMpc
        print('Lbox=%s'%(Lbox))
        pos = np.asarray([i.pos.in_units('kpccm') for i in sim.halos])
        pos = pos/1000. # cMpc
        for k in np.arange(len(MVIR)):
            # index k -> different definitions of halo mass
            if (k==0):
                M = np.asarray([i.masses['total'].to('Msun') for i in sim.halos])
            else:
                mvir = MVIR[k]
                M = np.asarray([i.virial_quantities[mvir].in_units('Msun') for i in sim.halos])
            logM = np.log10(M) # M is already expressed in units of MSun
            N_haloes, M_edges = np.histogram(logM, bins=Mbins, range=(logM_min, logM_max), density=False)
            binwidth = M_edges[1]-M_edges[0]
            M_bins[j,k,:] = 0.5*(M_edges[1:]+M_edges[:-1])
            hmf[j,k,:] = N_haloes/(binwidth*Lbox**3)
            hmf_err = np.zeros(len(M_bins[j,k,:]))
            for l in np.arange(len(M_bins[j,k,:])):
                # index l -> different mass bins
                mask = (logM<=M_edges[l+1]) & (logM>M_edges[l])
                if (len(pos[mask])>0):
                    # compute scatter due to cosmic variance
                    _, hmf_err[l] = N_scatter(np.ones(len(M[mask])), pos[mask], Lbox)
                    hmf_err[l] = (8./(7.*binwidth*Lbox**3))*hmf_err[l]
                else:
                    hmf_err[l] = np.nan
            if (k==0):
                # the mask hmf[...]>0 removes bins where there are no halos from the plot
                # this may happen in high-mass bins if the binning is too fine
                ax.fill_between(M_bins[j,k, :][hmf[j,k,:]>0], hmf[j,k,:][hmf[j,k,:]>0]-hmf_err[hmf[j,k,:]>0], hmf[j,k,:][hmf[j,k,:]>0]+hmf_err[hmf[j,k,:]>0], facecolor=colour[j], alpha=0.2)
            plt.plot(M_bins[j,k,:][hmf[j,k,:]>0], hmf[j,k,:][hmf[j,k,:]>0], c=colour[j], lw=lwlist[k], ls=lstyle[k])

    plt.annotate(r'$z=%g$'%(np.round(z_sim,1)), xy=(0.8, 0.9), xycoords='axes fraction',size=14,bbox=dict(boxstyle="round", fc="w"))
    plt.xlim(8.5,14)
    plt.ylim(5.e-7,0.3)
    plt.yscale('log')
    if (ii%2==0):
        plt.ylabel(r'$dn/d\mathrm{log}M_{\rm h}$ $(\rm cMpc^{-3})$', fontsize=16)
    else:
        ax.tick_params(labelleft=None)
    if (ii==0):
        plt.axvline(-1, c='black', lw=2, label='Simba-Dark')
        plt.axvline(-1, c='teal', lw=2, label=r'Simba $100 \, h^{-1} \, \rm cMpc$')
        for k in np.arange(len(MVIR)):
            plt.axvline(-1, c='grey', lw=lwlist[k], ls=lstyle[k], label=Delta_lab[k])
        plt.legend(loc='lower left', fontsize=12, ncol=2)
    ax.tick_params(labelsize=14, labelbottom=None)
    plt.axvline(logMmin, c='black', ls=':', lw=1)
    

    dID = diff_ID[ii]
    ax=fig1.add_subplot(gs[dID])
    # Plot relative difference
    for k in np.arange(len(MVIR)):
        plt.plot(M_bins[0,k,:], hmf[0,k,:]/hmf[-1,k,:]-1, c='black', ls=lstyle[k], lw=lwlist[k])
        print(hmf[0,k,:]/hmf[-1,k,:]-1)
    plt.axhline(0, c='black', lw=1, ls=':')

    plt.xlim(8.5, 14)
    plt.ylim(-1., 2.0)
    plt.axvline(logMmin, c='black', ls=':', lw=1)
    if(ii%2==0):
        plt.ylabel(r'$\rm HMF/HMF_{\rm fid} -1$', fontsize=16)
    else:
        ax.tick_params(labelleft=None)
    plt.xlabel(r'$\log(M_{\rm h}/\rm M_{\odot})$', fontsize=16)
    if (ii<2):
        ax.tick_params(labelsize=14, labelbottom=None)
    ax.tick_params(labelsize=14)



fformat='pdf'
fig1.savefig(pdfpath+"hmf_simba-dm_z.%s"%(fformat), format=fformat, bbox_inches='tight')



