"""
    Compute and plot the halo mass function of the fiducial-100 run at different redshift.
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
import matplotlib.colors as mcolors
from astropy.io import ascii
from astropy.cosmology import FlatLambdaCDM
from pygadgetreader import *
from scipy import interpolate
from scipy.interpolate import interp1d
from plotting_methods import N_scatter

def Hubble(z):
    return H0*(Om0*(1+z)**3+OL0)**0.5

model = 'm100n1024'
wind = 's50'

Mbins = 20
logM_max = 14
logM_min = 9
dlogM = (logM_max - logM_min)/Mbins
# mass of 100 DM particles
res_DM = math.log10(9.7e9) # MSun
# mass of 100 gas elements
res_gas = math.log10(1.82e9) # MSun

pdfpath='/disk04/sorini/outputs/plots/'
lstyle=['-','-.','--', ':']
SNAP_ar = [151,105, 78, 51]

fig1 = plt.figure(figsize=(6, 4))
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

ax=plt.subplot(111)

plt.title(r'Simba $100 \, h^{-1} \, \rm cMpc$', fontsize=18)
colour = 'teal'

plt.axhline(-1, c='teal', label='BMF')
plt.axhline(-1, c='gold', label='Rescaled HMF')
for j in np.arange(len(SNAP_ar)):
    snap = SNAP_ar[j]
    infile = '/disk04/rad/sim/%s/%s/Groups/%s_%03d.hdf5' %  (model,wind,model,snap)
    snap_file = '/disk04/rad/sim/%s/%s/snap_%s_%03d.hdf5' %  (model,wind,model,snap)
    s = pg.Snap(snap_file)
    ds = yt.load(snap_file)#, index_ptype='PartType0')
    sim = caesar.load(infile)#, LoadHalo = True)
    h = sim.simulation.hubble_constant
    Lbox = readheader(snap_file, 'boxsize') # kpc/h
    Lbox = Lbox/1000./h # cMpc/h
    Lbox = 100./h # cMpc
    print('Lbox=%s'%(Lbox))
    z_sim = sim.simulation.redshift
    cosmo = FlatLambdaCDM(H0=100*sim.simulation.hubble_constant,
                              Om0=sim.simulation.omega_matter,
                              Ob0=sim.simulation.omega_baryon,
                              Tcmb0=2.73)
    Ob0=sim.simulation.omega_baryon
    Om0=sim.simulation.omega_matter
    Mtot = np.asarray([i.masses['total'].to('Msun') for i in sim.halos])
    M = np.asarray([i.masses['baryon'].to('Msun') for i in sim.halos])
    pos = np.asarray([i.pos.in_units('kpccm') for i in sim.halos])
    pos = pos/1000. # cMpc
    logM = np.log10(M) # M is already expressed in units of MSun
    N_haloes, M_edges = np.histogram(logM, bins=Mbins, range=(logM_min, logM_max), density=False)
    M_bins = 0.5*(M_edges[1:]+M_edges[:-1])
    binwidth = M_bins[1]-M_bins[0]
    bmf = N_haloes/(binwidth*Lbox**3)
    bmf_err = np.zeros(len(M_bins))
    logMtot = np.log10(Mtot)
    N_haloes_tot, M_edges_tot = np.histogram(logMtot, bins=Mbins, range=(logM_min, logM_max), density=False)
    M_bins_tot = 0.5*(M_edges_tot[1:]+M_edges_tot[:-1])
    bmf_tot = N_haloes_tot/(binwidth*Lbox**3)
    bmf_err_tot = np.zeros(len(M_bins))
    for k in np.arange(len(M_bins)):
        mask = (logM<=M_edges[k+1]) & (logM>M_edges[k])
        mask_tot = (logMtot<=M_edges_tot[k+1]) & (logMtot>M_edges_tot[k])
        if (len(pos[mask])>0):
            # compute scatter due to cosmic variance
            _, bmf_err[k] = N_scatter(np.ones(len(M[mask])), pos[mask], Lbox)
            bmf_err[k] = (8./(7.*binwidth*Lbox**3))*bmf_err[k]
        else:
            bmf_err[k] = np.nan
        if (len(pos[mask_tot])>0):
            _, bmf_err_tot[k] = N_scatter(np.ones(len(Mtot[mask_tot])), pos[mask_tot], Lbox)
            bmf_err_tot[k] = (8./(7.*binwidth*Lbox**3))*bmf_err_tot[k]
        else:
            bmf_err_tot[k] = np.nan
    # the mask bmf>0 removes bins where there are no halos from the plot
    # this may happen in high-mass bins if the binning is too fine
    plt.plot(math.log10(Ob0/Om0)+M_bins_tot[bmf_tot>0], bmf_tot[bmf_tot>0], c='gold', lw=2, ls=lstyle[j])
    plt.plot(M_bins[bmf>0], bmf[bmf>0], c=colour, lw=2, ls=lstyle[j])
    plt.axhline(-1, label='$z=%g$'%(np.round(z_sim,1)), ls=lstyle[j], c='grey')
    if (j == 0):
        ax.fill_between(M_bins[bmf>0], bmf[bmf>0]-bmf_err[bmf>0], bmf[bmf>0]+bmf_err[bmf>0], facecolor=colour, alpha=0.3)
        ax.fill_between(math.log10(Ob0/Om0)+M_bins_tot[bmf_tot>0], bmf_tot[bmf_tot>0]-bmf_err_tot[bmf_tot>0], bmf_tot[bmf_tot>0]+bmf_err_tot[bmf_tot>0], facecolor='gold', alpha=0.2)

ax.axvline(res_gas, c='black', ls=':', lw=1)
plt.xlim(8.5, 14)
plt.yscale('log')
plt.xlabel(r'$\log(M_{\rm b}/\rm M_{\odot})$', fontsize=16)
plt.ylabel(r'$dn/d\mathrm{log}M_{\rm b}$ $(\rm cMpc^{-3})$', fontsize=16)
plt.legend(loc='lower left', fontsize=12, ncol=1)
ax.tick_params(labelsize=14)
ax.tick_params(labelsize=14)



fformat='pdf'
fig1.savefig(pdfpath+"bmf_%s%s.%s"%(model,wind,fformat), format=fformat, bbox_inches='tight')



