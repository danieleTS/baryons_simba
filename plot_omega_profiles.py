"""
    Plot profiles of enclosed mass M(<r) vs distance from halos, normalised by f_b \times M(<r),
    for different baryonic phases, across different Simba runs and different redshift
"""
import matplotlib.pyplot as plt
import numpy as np
import h5py
import caesar
import os
import sys
from plotting_methods import *
import math

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=14)

alpha=.8
palette_name = 'tol'
dm = 0.25 # dex
logr_min = -2 # distance in units of Rvir

model = sys.argv[1]
wind = sys.argv[2]

mod_lab = [r'Simba $50 \, h^{-1} \, \rm cMpc$', 'SFB + AGN Winds + Jets', 'SFB + AGN Winds', 'Stellar Feedback', 'No Feedback', r'Simba $100 \, h^{-1} \, \rm cMpc$']

if (wind=='s50j7k'):
    if (model=='m100n1024'):
        title=mod_lab[-1]
    else:
        title = mod_lab[0]
if (wind=='s50nox'):
    title = mod_lab[1]
if (wind=='s50nojet'):
    title = mod_lab[2]
if (wind=='s50noagn'):
    title = mod_lab[3]
if (wind=='s50nofb'):
    title = mod_lab[4]

data_dir = '/disk04/sorini/outputs/budgets/'+model+'/'+wind+'/'
savedir =  '/disk04/sorini/outputs/plots/'

all_phases = ['Cool CGM (T < 10^5)', 'Warm CGM (10^5 < T < 10^6)', 'Hot CGM (T > 10^6)',
              'Cool CGM (T < Tphoto)', 'Warm CGM (Tphoto < T < 0.5Tvir)', 'Hot CGM (T > 0.5Tvir)',
              'ISM', 'Wind', 'Stars', 'Total gas', 'Total baryons','Total']
plot_phases = ['Hot CGM (T > 0.5Tvir)', 'Warm CGM (Tphoto < T < 0.5Tvir)', 'Cool CGM (T < Tphoto)',
               'Wind', 'ISM', 'Stars']
plot_phases_labels = [r'Hot CGM', r'Warm CGM',
                      r'Cool CGM', 'Wind', 'ISM', 'Stars']

colours = ['m', 'b', 'c', 'g', 'tab:orange', 'tab:pink', 'r']
colours = get_cb_colours(palette_name)[::-1]
stats = ['median', 'percentile_25_75', 'std', 'cosmic_median', 'cosmic_std']

minM_ar = [11., 11.6, 12.2, 12.8]
maxM_ar = [11.6, 12.2, 12.8, 15]
SNAP = ['051', '078', '105', '151']
y_max=[2.2, 2.5,2.9,4,4.8]


fig = plt.figure(figsize=(6*len(SNAP)+1, 6*len(minM_ar)))
fig.subplots_adjust(wspace=0, hspace=0)
fig.suptitle(title, fontsize=40, y=0.93)

for im in np.arange(len(minM_ar)):
    min_mass = 10**minM_ar[im]
    max_mass = 10**maxM_ar[im]
    if model == 'm100n1024':
        boxsize = 100000.
    elif model == 'm50n512':
        boxsize = 50000.
    elif model == 'm25n512':
        boxsize = 25000.
    for iz in np.arange(len(SNAP)):
        if (iz==len(SNAP)-1):
            logr_max=1
        else:
            logr_max = 1
        snap = SNAP[iz]
        plt_ID = im * len(SNAP) + iz + 1
        print('plt_ID = ', plt_ID)
        ax = plt.subplot(len(minM_ar), len(SNAP), plt_ID)
        ax.set_xlim(0.009, 11)
        ax.set_ylim(0,y_max[im])
        ax.set_xscale('log')

        caesarfile = '/disk04/rad/sim/'+model+'/'+wind+'/Groups/'+model+'_'+snap+'.hdf5'
        sim = caesar.quick_load(caesarfile)
        h = sim.simulation.hubble_constant
        z_sim = sim.simulation.redshift
        boxsize=boxsize/h
        central = np.array([i.central for i in sim.galaxies])
        halo_m = np.array([i.halo.masses['total'].in_units('Msun') for i in sim.galaxies])[central]
        gal_pos = np.array([i.pos.in_units('kpccm') for i in sim.galaxies])[central]
        print('z = %s'%(z_sim))
        if (im == len(minM_ar)-1):
            mask = halo_m > min_mass
        else:
            mask = (halo_m > min_mass) & (halo_m < max_mass)

        # get the mass budget data:
        omega_profile = read_phases(data_dir+'omega_frac_profiles_%s.h5'%(snap), all_phases)

        for i in np.arange(len(plot_phases)):
            phase = plot_phases[i]
            print('Shape of %s = %s'%(phase, np.shape(omega_profile[phase])))
            rbins = np.shape(omega_profile[phase])[1]
            if (i==0):
                running_total = np.zeros(rbins)
            logr_med = np.linspace(logr_min, logr_max, rbins)
            med = np.zeros(rbins)
            print('rbins = %s'%(rbins))
            omega = omega_profile[phase]
            omega = omega[mask]
            gal_pos_cut = gal_pos[mask]
            print('After masking: shape of %s = %s'%(phase, np.shape(omega)))
            print('Shape of gal_pos = ',np.shape(gal_pos))
            
            
            for j in np.arange(rbins):
                if (len(omega[:,j])>0):
                    med[j] = np.nanmedian(omega[:,j])
                else:
                    med[j] = np.nan

            print(med)
            ax.fill_between(10**logr_med, running_total, running_total + med,
                color=colours[i], label=plot_phases_labels[i], alpha=alpha)
            running_total += med

        plt.axhline(1, c='black', lw=1, ls=':')
        if (im == 0):
            plt.title(r'$z=%g$'%(np.round(z_sim, 0)), fontsize=30, y=1.05)
            if (iz == len(SNAP)-1):
                ax.legend(loc='upper right', fontsize=18, framealpha=0., ncol=2)
                
        if (im == len(minM_ar)-1):
            ax.set_xlabel(r'$r / r_{200}$', fontsize=30)
            ax.tick_params(labelsize=22)
        else:
            ax.tick_params(labelbottom=None)

        if (iz == 0):
            if (im == len(minM_ar)-1):
                ax.annotate(r'$\log(M_{\rm halo}/\mathrm{M}_{\odot})>%s$'%(minM_ar[im]), xy=(0.05, 0.9), xycoords='axes fraction',size=18,bbox=dict(boxstyle="round",fc="w"))
            else:
                ax.annotate(r'$%s<\log(M_{\rm halo}/\mathrm{M}_{\odot})<%s$'%(minM_ar[im], maxM_ar[im]), xy=(0.35, 0.9), xycoords='axes fraction',size=18,bbox=dict(boxstyle="round",fc="w"))
            ax.set_ylabel(r'$M(<r)/ f_{\rm b} \: M_{\rm tot}(<r)$', fontsize=30, x=1.05)
            ax.tick_params(labelsize=22)
        else:
            ax.tick_params(labelleft=None)

fformat = 'pdf'
plt.savefig(savedir+'omega_profiles_%s%s.%s'%(model, wind, fformat), format=fformat, bbox_inches = 'tight')
plt.clf()
