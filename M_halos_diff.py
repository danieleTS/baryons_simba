"""
    Compute relative difference in total/stellar/baryonic mass for "halo copies" of different feedback variants
"""

import sys
import pygad as pg
import yt
import math
from yt.units.yt_array import YTQuantity
import caesar
import h5py
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from astropy.io import ascii
from astropy.cosmology import FlatLambdaCDM
from pygadgetreader import *
from scipy import interpolate
from scipy.interpolate import interp1d
from progen.progen import find_progens

MODEL_ref = sys.argv[1]
WIND_ref = sys.argv[2]
MODEL = MODEL_ref

Nbins = 20
# mass of 1000 DM particles
Mmin = 9.7e10 # MSun

pdfpath='/disk04/sorini/outputs/plots/'
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
WIND = ['s50nox', 's50nojet', 's50noagn', 's50nofb']
colour = ['red', 'blue', 'purple', 'orange']
mecolour = ['red', 'blue', 'purple', 'orange']
mod_lab = ['No-X-ray', 'No-jet', 'No-AGN', 'No-feedback']
PARTTYPE = ['total', 'baryon', 'stellar']
ylabel = [r'$M_{\rm h}/M_{\rm h}^{\rm fid}$-1', r'$M_{\rm b}/M_{\rm b}^{\rm fid}$', r'$M_*/M_*^{\rm fid}$']
SNAP_ar = [51, 78, 105, 151]
lstyle=['-','-.', '--', ':']
ymin=[-0.4, 0.7, 0.5]
ymax=[1, 20, 100]

fig1 = plt.figure(figsize=(6*len(SNAP_ar), 6*len(PARTTYPE)))
fig1.subplots_adjust(wspace=0, hspace=0)
for ip in np.arange(len(PARTTYPE)):
    # index ip -> type of mass function investigated (i.e., HMF, BMF, SMF)
    parttype = PARTTYPE[ip]

    for ii in np.arange(len(SNAP_ar)):
        # index ii -> snapshot number (redshift)
        plt_ID = ip * len(SNAP_ar) + ii + 1
        ax=plt.subplot(len(PARTTYPE),len(SNAP_ar),plt_ID)
        SNAP = SNAP_ar[ii]

        # open reference run (input)
        infile_ref = '/disk04/rad/sim/%s/%s/Groups/%s_%03d.hdf5' % (MODEL_ref,WIND_ref,MODEL_ref,SNAP)
        snap_file_ref = '/disk04/rad/sim/%s/%s/snap_%s_%03d.hdf5' % (MODEL_ref,WIND_ref,MODEL_ref,SNAP)
        obj_ref = caesar.load(infile_ref) # objects (in our case halos) in the reference run
        z_sim = obj_ref.simulation.redshift
        # read total mass of haloes in reference run
        M_ref = np.asarray([i.masses['total'].to('Msun') for i in obj_ref.halos])
        # read total/baryon/stellar mass of halos in refernce run
        Mp_ref = np.asarray([i.masses[parttype].to('Msun') for i in obj_ref.halos])
        # get the halo IDs in the reference run
        IDs_ref = np.asarray([i.GroupID for i in obj_ref.halos])

        for j in np.arange(len(WIND)):
            # open target run
            infile = '/disk04/rad/sim/%s/%s/Groups/%s_%03d.hdf5' % (MODEL,WIND[j],MODEL,SNAP)
            snap_file = '/disk04/rad/sim/%s/%s/snap_%s_%03d.hdf5' % (MODEL,WIND[j],MODEL,SNAP)
            obj = caesar.load(infile) # objects (in our case halos) in the target run
            
            # find the halos in the target run that share the highest and 2nd highest amount of
            # dm particles with the halos of the reference run
            prog_1, prog_2 = find_progens(snap_file_ref,snap_file,obj_ref,obj,1,0.,objtype='halos',parttype='dm')
            
            # read total mass of haloes in target run
            M = np.asarray([i.masses['total'].to('Msun') for i in obj.halos])
            # read total/baryon/stellar mass of halos in target run
            Mp = np.asarray([i.masses[parttype].to('Msun') for i in obj.halos]) # MSun
            # get the halo IDs in the target run
            IDs = np.asarray([i.GroupID for i in obj.halos]) # MSun
            # eliminate halos without particles of the type considered (e.g. stars, baryons) to avoid spurious results
            IDs_plot = IDs_ref[np.logical_and(prog_1>=0, Mp_ref>0)]
            prog_1 = prog_1[np.logical_and(prog_1>=0, Mp_ref>0)]
            DMp = np.zeros(len(IDs_plot))
            Mplt = np.zeros(len(DMp))
            # Calculate relative difference wrt reference run
            for k in np.arange(len(IDs_plot)):
                Mplt[k] = M_ref[IDs_plot[k]]
                if (ip == 0):
                    DMp[k] = Mp[prog_1[k]]/Mp_ref[IDs_plot[k]]-1
                else:
                    DMp[k] = Mp[prog_1[k]]/Mp_ref[IDs_plot[k]]
            
            DMp = DMp[Mplt>Mmin]
            Mplt = Mplt[Mplt>Mmin]
            logM = np.log10(Mplt)
            dlogM = (np.amax(logM)-np.log10(Mmin))/Nbins
            logM_edges = np.linspace(math.log10(Mmin), np.amax(logM), Nbins+1)
            logM_centre = 0.5*(logM_edges[:-1]+logM_edges[1:])
            DMp_med = np.zeros(Nbins)
            DMp_sup = np.zeros(Nbins)
            DMp_inf = np.zeros(Nbins)
            for l in np.arange(Nbins):
                mask = np.logical_and(logM >= math.log10(Mmin)+l*dlogM, logM < math.log10(Mmin)+(l+1)*dlogM)
                if (len(logM[mask])==0):
                    DMp_med[l] = np.nan
                    DMp_sup[l] = np.nan
                    DMp_inf[l] = np.nan
                else:
                    DMp_med[l] = np.median(DMp[mask])
                    DMp_sup[l] = np.percentile(DMp[mask], 84)
                    DMp_inf[l] = np.percentile(DMp[mask], 16)
        
            if (ip==0):
                plt.axhline(0, c='black', lw=1, ls=':')
            else:
                plt.axhline(1, c='black', lw=1, ls=':')
            plt.plot(logM_centre, DMp_med, c=colour[j], label=mod_lab[j], lw=3, ls='-')
            plt.plot(logM_centre, DMp_inf, c=colour[j], lw=1, ls=':')
            plt.plot(logM_centre, DMp_sup, c=colour[j], lw=1, ls=':')

        
        plt.xlim(math.log10(Mmin), 14.5)
        plt.ylim(ymin[ip], ymax[ip])
        if (ip>0):
            plt.yscale('log')
        if (ip==0):
            plt.title(r'$z=%g$'%(np.round(z_sim, 0)), fontsize=25, y=1.05)
        ax.tick_params(labelsize=22)
        if (ii == 0):
            plt.ylabel(ylabel[ip], fontsize=25)
        else:
            ax.tick_params(labelleft=None)
        if (ip < len(PARTTYPE)-1):
            ax.tick_params(labelbottom=None)
        else:
            plt.xlabel(r'$\log(M_{\rm h}/ \rm M_{\odot})$', fontsize=25)

fformat='pdf'
fig1.savefig(pdfpath+"diffM_haloes.%s"%(fformat), format=fformat, bbox_inches='tight')



