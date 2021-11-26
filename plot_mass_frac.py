"""
    Plot mass fraction vs halo mass for different Simmba runs and different redshift.
"""

import matplotlib.pyplot as plt
import numpy as np
import h5py
import caesar
import os
import sys
from plotting_methods import *

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=14)

alpha = .8
palette_name = 'tol'
min_mass = 11.
max_mass = 14.5
dm = 0.25 # dex

savedir = '/disk01/sorini/outputs/plots/'
all_phases = ['Cool CGM (T < Tphoto)', 'Warm CGM (Tphoto < T < 0.5Tvir)', 'Hot CGM (T > 0.5Tvir)',
              'Cool CGM (T < 10^5)', 'Warm CGM (10^5 < T < 10^6)', 'Hot CGM (T > 10^6)',
              'ISM', 'Wind', 'Dust', 'Stars', 'Cosmic baryon mass']
plot_phases = ['Hot CGM (T > 0.5Tvir)', 'Warm CGM (Tphoto < T < 0.5Tvir)', 'Cool CGM (T < Tphoto)',
               'Wind', 'Dust', 'ISM', 'Stars']
plot_phases_labels = [r'Hot CGM $(T > 0.5T_{\rm vir})$', r'Warm CGM $(T_{\rm photo} < T < 0.5T_{\rm vir})$',
                      r'Cool CGM $(T < T_{\rm photo})$', 'Wind', 'Dust', 'ISM', 'Stars']
colours = ['m', 'b', 'c', 'g', 'tab:orange', 'tab:pink', 'r']
colours = get_cb_colours(palette_name)[::-1]
stats = ['median', 'percentile_25_75', 'std', 'cosmic_median', 'cosmic_std']


WIND = ['s50j7k', 's50nox', 's50nojet', 's50noagn', 's50nofb']
MODEL = ['m50n512']*len(WIND)
SNAP = ['051', '078', '105', '151']
mod_lab = [r'Simba $50 \, h^{-1} \, \rm cMpc$', 'SFB + AGN Winds + Jets', 'SFB + AGN Winds', 'Stellar Feedback', 'No Feedback']

fig = plt.figure(figsize=(6*len(SNAP)+1, 6*len(WIND)))
fig.subplots_adjust(wspace=0, hspace=0)

for iw in np.arange(len(WIND)):
    wind = WIND[iw]
    model = MODEL[iw]
    if model == 'm100n1024':
        boxsize = 100000.
    elif model == 'm50n512':
        boxsize = 50000.
    elif model == 'm25n512':
        boxsize = 25000.
    for iz in np.arange(len(SNAP)):
        snap = SNAP[iz]
        plt_ID = iw * len(SNAP) + iz + 1
        print('plt_ID = ', plt_ID)
        ax = plt.subplot(len(WIND), len(SNAP), plt_ID)

        fracdata_dir = '/disk04/sorini/outputs/budgets/'+model+'/'+wind+'/'
        caesarfile = '/disk04/rad/sim/'+model+'/'+wind+'/Groups/'+model+'_'+snap+'.hdf5'
        sim = caesar.quick_load(caesarfile)
        z_sim = sim.simulation.redshift
        print('z = %s'%(z_sim))
        central = np.array([i.central for i in sim.galaxies])
        halo_m = np.array([i.halo.masses['total'].in_units('Msun') for i in sim.galaxies])[central]

        gal_pos = np.array([i.pos.in_units('kpc/h') for i in sim.galaxies])[central]

        fractions = read_phases(fracdata_dir+'omega_mass_fraction_%s.h5'%(snap), all_phases)

        frac_stats = {}
        mass_bins = get_bin_edges(min_mass, max_mass, dm)
        frac_stats['hmass_bins'] = get_bin_middle(np.append(mass_bins, mass_bins[-1] + dm))
        mask = np.array([True] * len(halo_m))
        frac_stats['all'] = get_phase_stats(halo_m, gal_pos, fractions, mask, all_phases, mass_bins, boxsize, logresults=False)

        running_total = np.zeros(len(frac_stats['hmass_bins']))
        for i, phase in enumerate(plot_phases):
            if phase == 'Dust':
                continue
            ax.fill_between(frac_stats['hmass_bins'], running_total, running_total + frac_stats['all'][phase]['median'],
                               color=colours[i], label=plot_phases_labels[i], alpha=alpha)
            running_total += frac_stats['all'][phase]['median']

        plt.axhline(1, c='black', lw=1, ls=':')
        ax.set_xlim(min_mass+dm, max_mass-dm)

        if (iw == 0):
            plt.title(r'$z=%s$'%(np.round(z_sim, 0)), fontsize=30, y=1.05)
            if (iz == len(SNAP)-1):
                ax.legend(loc='upper left', fontsize=18, framealpha=0.)

        if (iw == len(WIND)-1):
            ax.set_xlabel(r'$\log (M_{\rm halo} / \rm M_{\odot})$', fontsize=30, y=-1.05)
            ax.tick_params(labelsize=22)
            ax.set_ylim(0, 1.1)
        else:
            ax.set_ylim(1.e-2, 1.1)
            ax.tick_params(labelbottom=None)

        if (iz == 0):
            ax.text(-0.3,0.5, mod_lab[iw],horizontalalignment='center', verticalalignment='center',rotation='vertical',fontsize=30, transform=ax.transAxes)
            ax.set_ylabel(r'$M/ f_{\rm b} \: M_{\rm halo}$', fontsize=30, x=1.05)
            ax.tick_params(labelsize=22)
        else:
            ax.tick_params(labelleft=None)




fformat='pdf'
plt.savefig(savedir+'omega_fracs.%s'%(fformat), format=fformat, bbox_inches = 'tight')

plt.clf()
