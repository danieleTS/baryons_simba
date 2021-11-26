"""
    Plot densoty profiles of various components of halos in a certain Simba run
"""

import matplotlib.pyplot as plt
import numpy as np
import math
import h5py
import caesar
import os
import sys
import matplotlib.gridspec as gridspec
from plotting_methods import *

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=14)
palette_name = 'tol'
minM_ar = [11.0, 11.6, 12.2, 12.8]
maxM_ar = [11.6, 12.2, 12.8, 15]
dm = 0.25 # dex

logr_min = -2 # distance in units of Rvir
logr_max = math.log10(5)
#rbins = 20 # nr of logr-bins
#dlogr = (logr_max-logr_min)/rbins

model = sys.argv[1]
wind = sys.argv[2]
snap = sys.argv[3]

ymin=1.e-2 # min of the y-axis

if model == 'm100n1024':
    boxsize = 100000.
elif model == 'm50n512':
    boxsize = 50000.
elif model == 'm25n512':
    boxsize = 25000.

data_dir = '/disk04/sorini/outputs/budgets/'+model+'/'+wind+'/'
savedir =  '/disk04/sorini/outputs/plots/'

all_phases = ['Cool CGM (T < Tphoto)', 'Warm CGM (Tphoto < T < 0.5Tvir)', 'Hot CGM (T > 0.5Tvir)',
              'Cool CGM (T < 10^5)', 'Warm CGM (10^5 < T < 10^6)', 'Hot CGM (T > 10^6)',
              'ISM', 'Wind', 'Stars', 'Dark matter', 'Total Gas']
plot_phases_down = ['Hot CGM (T > 0.5Tvir)', 'Warm CGM (Tphoto < T < 0.5Tvir)', 'Cool CGM (T < Tphoto)', 'Wind', 'ISM']
plot_phases_up = ['Dark matter', 'Total Gas', 'Stars']
plot_phases_labels_down = ['Hot CGM', 'Warm CGM',
                      'Cool CGM', 'Wind', 'ISM']
plot_phases_labels_up = ['Dark matter', 'Gas', 'Stars']
colours = ['m', 'b', 'c', 'g', 'tab:orange', 'tab:pink', 'r', 'black']
colours_down = get_cb_colours(palette_name)[::-1]
colours_up = ['grey', 'blue', '#332288']
stats = ['median', 'percentile_25_75', 'std', 'cosmic_median', 'cosmic_std']

caesarfile = '/disk04/rad/sim/'+model+'/'+wind+'/Groups/'+model+'_'+snap+'.hdf5'
sim = caesar.quick_load(caesarfile)
h = sim.simulation.hubble_constant
boxsize=boxsize/h
central = np.array([i.central for i in sim.galaxies])
halo_m = np.array([i.halo.masses['total'].in_units('Msun') for i in sim.galaxies])[central]

# get the mass budget data:
density_profile = read_phases(data_dir+'density_profiles_%s.h5'%(snap), all_phases)

fig=plt.figure(figsize=(6*len(minM_ar)+1, 12))
gs = gridspec.GridSpec(ncols=4, nrows=2, width_ratios=[1,1,1,1], height_ratios=[1,1],wspace=0,hspace=0)

for im in np.arange(len(minM_ar)):
    
    #ax = plt.subplot(2, len(minM_ar), im+2)
    ax=fig.add_subplot(gs[im])
    min_mass = 10**minM_ar[im]
    max_mass = 10**maxM_ar[im]
    mask = (halo_m > min_mass) & (halo_m < max_mass)
    if (im == len(minM_ar)-1):
        ax.set_title(r'$\log(M_{\rm h}/\mathrm{M}_{\odot})>%s$'%(minM_ar[im]), fontsize=25, y=1.05)
    else:
        ax.set_title(r'$%s<\log(M_{\rm h}/\mathrm{M}_{\odot})<%s$'%(minM_ar[im], maxM_ar[im]), fontsize=25, y=1.05)
    for i in np.arange(len(plot_phases_up)):
        phase = plot_phases_up[i]
        print('Shape of %s = %s'%(phase, np.shape(density_profile[phase])))
        rbins = np.shape(density_profile[phase])[1]
        all_bins = np.linspace(logr_min, logr_max, rbins+1)
        logr_med = 0.5*(all_bins[:-1]+all_bins[1:])
        med = np.zeros(rbins)
        stdev = np.zeros(rbins)
        stdev_inf = np.zeros(rbins)
        inf_per = np.zeros(rbins)
        sup_per = np.zeros(rbins)
        cosmic_std = np.zeros(rbins)
        print('rbins = %s len(med) = %s'%(rbins, len(med)))
        dens = density_profile[phase]
        dens = dens[mask]
        gal_pos_cut = gal_pos[mask]
        print('After masking: shape of %s = %s'%(phase, np.shape(dens)))
        print('Shape of gal_pos = ',np.shape(gal_pos))
        
        
        for j in np.arange(rbins):
            if (len(dens[:,j])>0):
                med[j] = np.mean(dens[:,j], dtype=np.float64)
                stdev[j] = np.std(dens[:, j], dtype=np.float64, ddof=1)
            else:
                med[j] = np.nan
                sup_per[j] = np.nan
                inf_per[j] = np.nan

        # plot only the upper part of the error bar
        #if the lower bound of the bar falls below the y-min lower limit
        stdev_inf[med-stdev>ymin] = stdev[med-stdev>ymin]
        ax.errorbar(10**logr_med[med>0], med[med>0], yerr=[stdev_inf[med>0], stdev[med>0]], capsize=0, color=colours_up[i], label=plot_phases_labels_up[i])

    ax.set_xlim(0.009, 6)
    ax.set_ylim(ymin, 5.e8)
    plt.yscale('log')
    plt.xscale('log')
    ax.set_xlabel(r'$r / r_{200}$', fontsize=25)
    if (im == 0):
        ax.set_ylabel(r'$\rho$  $\rm (M_{\odot} \, ckpc^{-3})$', fontsize=25)
        ax.legend(loc='lower left', framealpha=0, fontsize=18)
    else:
        ax.tick_params(labelleft=False)
    ax.tick_params(labelsize=20, labelbottom=False)


    ax=fig.add_subplot(gs[im+len(minM_ar)])
    for i in np.arange(len(plot_phases_down)):
        phase = plot_phases_down[i]
        print('Shape of %s = %s'%(phase, np.shape(density_profile[phase])))
        rbins = np.shape(density_profile[phase])[1]
        all_bins = np.linspace(logr_min, logr_max, rbins)
        logr_med = 0.5*(all_bins[:-1]+all_bins[1:])
        med = np.zeros(rbins-1)
        stdev = np.zeros(rbins-1)
        stdev_inf = np.zeros(rbins-1)
        inf_per = np.zeros(rbins-1)
        sup_per = np.zeros(rbins-1)
        cosmic_std = np.zeros(rbins-1)
        print('rbins = %s'%(rbins))
        dens = density_profile[phase]
        dens = dens[mask]
       
       
        for j in np.arange(rbins-1):
            if (len(dens[:,j])>0):
                med[j] = np.mean(dens[:,j], dtype=np.float64)
                stdev[j] = np.std(dens[:, j], dtype=np.float64, ddof=1)
            else:
                med[j] = np.nan
                sup_per[j] = np.nan
                inf_per[j] = np.nan

        stdev_inf[med-stdev>ymin] = stdev[med-stdev>ymin]
        ax.errorbar(10**logr_med[med>0], med[med>0], yerr=[stdev_inf[med>0],stdev[med>0]], capsize=0, color=colours_down[i], label=plot_phases_labels_down[i])

    ax.set_xlim(0.009, 6)
    ax.set_ylim(ymin, 5.e8)
    plt.yscale('log')
    plt.xscale('log')
    ax.set_xlabel(r'$r / r_{200}$', fontsize=25)
    if (im == 0):
        ax.set_ylabel(r'$\rho$  $\rm (M_{\odot} \, ckpc^{-3})$', fontsize=25)
        ax.legend(loc='upper right', framealpha=0, fontsize=18)
    else:
        ax.tick_params(labelleft=False)
    ax.tick_params(labelsize=20)

fformat = 'pdf'
plt.savefig(savedir+'dens_profiles_%s%s_%s.%s'%(model, wind, snap, fformat), format=fformat, bbox_inches = 'tight')
plt.clf()
