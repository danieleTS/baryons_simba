"""
    Plot relative difference of density profiles of various components in different
    Simba runs
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
palette_name = 'tol'
dm = 0.25 # dex
logr_min = -2 # distance in units of r_200
logr_max = math.log10(5)

savedir =  '/disk04/sorini/outputs/plots/'

all_phases = ['Cool CGM (T < Tphoto)', 'Warm CGM (Tphoto < T < 0.5Tvir)', 'Hot CGM (T > 0.5Tvir)',
              'Cool CGM (T < 10^5)', 'Warm CGM (10^5 < T < 10^6)', 'Hot CGM (T > 10^6)',
              'ISM', 'Wind', 'Stars', 'Dark matter']
plot_phases = ['Dark matter', 'Hot CGM (T > 0.5Tvir)', 'Warm CGM (Tphoto < T < 0.5Tvir)', 'Cool CGM (T < Tphoto)', 'Wind', 'ISM', 'Stars']
plot_phases_labels = ['Dark matter', r'Hot CGM', r'Warm CGM',
                      r'Cool CGM', 'Wind', 'ISM', 'Stars']


colours = ['green', 'red', 'blue', 'purple', 'orange']
lstyle = ['-', '--']
WIND = ['s50', 's50nox', 's50nojet', 's50noagn', 's62nofb']
minM_ar = [11., 11.6, 12.2, 12.8]
maxM_ar = [11.6, 12.2, 12.8, 15]


MODEL = ['m50n512']*len(minM_ar)
SNAP = ['151', '078']
mod_lab = [r'Simba $50 \, h^{-1} \, \rm cMpc$', 'No-X-ray', 'No-jet', 'No-AGN', 'No-feedback']
# Limits of y-axis
y_min = [0.02, 0.005, 0.09, 0.1, 0.03, 0.2, 0.03]
y_max = [3, 30, 40, 200, 800, 200, 60]

fig = plt.figure(figsize=(6*len(minM_ar)+1, 4*len(plot_phases)))
fig.subplots_adjust(wspace=0, hspace=0)

for ip in np.arange(len(plot_phases)):
    phase = plot_phases[ip]
    for im in np.arange(len(minM_ar)):
        model = MODEL[im]
        min_mass = 10**minM_ar[im]
        max_mass = 10**maxM_ar[im]
        if model == 'm100n1024':
            boxsize = 100000.
        elif model == 'm50n512':
            boxsize = 50000.
        elif model == 'm25n512':
            boxsize = 25000.

        plt_ID = ip * len(minM_ar) + im + 1
        print('plt_ID = ', plt_ID)
        ax = plt.subplot(len(plot_phases),len(minM_ar), plt_ID)
        ax.set_xlim(0.04, 6)
        ax.set_ylim(y_min[ip], y_max[ip])
        ax.set_xscale('log')
        ax.set_yscale('log')
        
        for iz in np.arange(len(SNAP)):
            snap = SNAP[iz]
            for i in np.arange(len(WIND)):
                wind = WIND[i]
                if (wind=='s62nofb' and phase=='Wind'):
                    continue
                data_dir = '/disk04/sorini/outputs/budgets/'+model+'/'+wind+'/'
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
            
                # get the density profile data:
                density_profile = read_phases(data_dir+'density_profiles_%s.h5'%(snap), all_phases)
                print('Shape of %s = %s'%(phase, np.shape(density_profile[phase])))
                rbins = np.shape(density_profile[phase])[1]
                all_bins = np.linspace(logr_min, logr_max, rbins)
                logr_med = 0.5*(all_bins[:-1]+all_bins[1:])
                med = np.zeros(rbins-1)
                if (i==0 and iz==0):
                    ref_med = np.zeros(rbins-1)
                inf_per = np.zeros(rbins-1)
                sup_per = np.zeros(rbins-1)
                cosmic_std = np.zeros(rbins-1)
                dens = density_profile[phase]
                dens = dens[mask]
                gal_pos_cut = gal_pos[mask]

                
                for j in np.arange(rbins-1):
                    if (len(dens[:,j]>0)>0):
                        med[j] = np.mean(dens[:,j], dtype=np.float64)
                    else:
                        med[j] = np.nan
                    if (i==0 and iz==0):
                        ref_med[j] = med[j]

                # cut plot at 0.05 \times r_200 (poor convergence for smaller r)
                med = med[logr_med>=math.log10(0.05)]
                if (i==0 and iz==0):
                    ref_med = ref_med[logr_med>=math.log10(0.05)]
                logr_med = logr_med[logr_med>=math.log10(0.05)]
                ax.plot(10**logr_med, med/ref_med, color=colours[i], ls=lstyle[iz], lw=2)
                if (iz==0):
                    ax.axvline(-1, color=colours[i], ls='-', lw=2, label=mod_lab[i])
                if (i==len(WIND)-1):
                    ax.axvline(-1, color='grey', ls=lstyle[iz], lw=2, label=r'$z=%g$'%(round(z_sim,0)))

        
        if (im == 0):
            ax.set_ylabel(r'$\rho/\rho^{\rm fid}_{z=0}$', fontsize=30, x=1.05)
            ax.tick_params(labelsize=22)
            ax.text(-0.3,0.5, plot_phases_labels[ip],horizontalalignment='center', verticalalignment='center',rotation='vertical',fontsize=30, transform=ax.transAxes)
        else:
            ax.tick_params(labelleft=None)
        
                
        if (ip == len(plot_phases)-1):
            ax.set_xlabel(r'$r  / r_{200}$', fontsize=30)
            ax.tick_params(labelsize=22)
        else:
            ax.tick_params(labelbottom=None)

        if (ip == 0):
            if (im == len(minM_ar)-1):
                ax.set_title(r'$\log(M_{\rm h}/\mathrm{M}_{\odot})>%s$'%(minM_ar[im]), fontsize=28, y=1.05)
                ax.legend(loc='lower left', fontsize=16, ncol=2, framealpha=0)
            else:
                ax.set_title(r'$%s<\log(M_{\rm h}/\mathrm{M}_{\odot})<%s$'%(minM_ar[im], maxM_ar[im]), fontsize=28, y=1.05)

fformat = 'pdf'
plt.savefig(savedir+'dens_prof_diff.%s'%(fformat), format=fformat, bbox_inches = 'tight')
plt.clf()
