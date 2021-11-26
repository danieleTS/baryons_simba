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
from astropy.io import ascii
from astropy.cosmology import FlatLambdaCDM
from pygadgetreader import *
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

for j in np.arange(len(SNAP_ar)):
    snap = SNAP_ar[j]
    infile = '/disk04/rad/sim/%s/%s/Groups/%s_%03d.hdf5' %  (model,wind,model,snap)
    snap_file = '/disk04/rad/sim/%s/%s/snap_%s_%03d.hdf5' %  (model,wind,model,snap)
    s = pg.Snap(snap_file)
    sim = caesar.load(infile)
    h = sim.simulation.hubble_constant
    Lbox = readheader(snap_file, 'boxsize') # kpc/h
    Lbox = Lbox/1000./h # cMpc
    z_sim = sim.simulation.redshift
    cosmo = FlatLambdaCDM(H0=100*sim.simulation.hubble_constant,
                              Om0=sim.simulation.omega_matter,
                              Ob0=sim.simulation.omega_baryon,
                              Tcmb0=2.73)
    M = np.asarray([i.masses['total'].to('Msun') for i in sim.halos])
    pos = np.asarray([i.pos.in_units('kpccm') for i in sim.halos])
    pos = pos/1000. # cMpc
    logM = np.log10(M) # M is already expressed in units of MSun
    N_haloes, M_edges = np.histogram(logM, bins=Mbins, range=(logM_min, logM_max), density=False)
    M_bins = 0.5*(M_edges[1:]+M_edges[:-1])
    binwidth = M_bins[1]-M_bins[0]
    hmf = N_haloes/(binwidth*Lbox**3)
    hmf_err = np.zeros(len(M_bins))
    for k in np.arange(len(M_bins)):
        mask = (logM<=M_edges[k+1]) & (logM>M_edges[k])
        if (len(pos[mask])>0):
            # compute scatter due to cosmic variance
            _, hmf_err[k] = N_scatter(np.ones(len(M[mask])), pos[mask], Lbox)
            hmf_err[k] = (8./(7.*binwidth*Lbox**3))*hmf_err[k]
        else:
            hmf_err[k] = np.nan
    # the mask hmf>0 removes bins where there are no halos from the plot
    # this may happen in high-mass bins if the binning is too fine
    plt.plot(M_bins[hmf>0], hmf[hmf>0], c=colour, lw=2, label='$z=%g$'%(np.round(z_sim,0)), ls=lstyle[j])
    if (j == 0):
        ax.fill_between(M_bins[hmf>0], hmf[hmf>0]-hmf_err[hmf>0], hmf[hmf>0]+hmf_err[hmf>0], facecolor=colour, alpha=0.3)

ax.axvline(res_DM, c='black', ls=':', lw=1)
plt.xlim(8.5, 14)
plt.yscale('log')
plt.xlabel(r'$\log(M_{\rm h}/\rm M_{\odot})$', fontsize=16)
plt.ylabel(r'$dn/d\mathrm{log}M_{\rm h}$ $(\rm cMpc^{-3})$', fontsize=16)
plt.legend(loc='lower left', fontsize=12, ncol=1)
ax.tick_params(labelsize=14)
ax.tick_params(labelsize=14)



fformat='pdf'
fig1.savefig(pdfpath+"hmf_%s%s.%s"%(model,wind,fformat), format=fformat, bbox_inches='tight')



