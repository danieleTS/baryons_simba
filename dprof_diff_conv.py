"""
    Convergence test for density profiles
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
logr_min = -2 # distance in units of Rvir
logr_max = math.log10(5)

savedir =  '/disk04/sorini/outputs/plots/'

all_phases = ['Cool CGM (T < Tphoto)', 'Warm CGM (Tphoto < T < 0.5Tvir)', 'Hot CGM (T > 0.5Tvir)',
              'Cool CGM (T < 10^5)', 'Warm CGM (10^5 < T < 10^6)', 'Hot CGM (T > 10^6)',
              'ISM', 'Wind', 'Dust', 'Stars', 'Dark matter']
plot_phases = ['Dark matter', 'Hot CGM (T > 0.5Tvir)', 'Warm CGM (Tphoto < T < 0.5Tvir)', 'Cool CGM (T < Tphoto)', 'Wind', 'ISM', 'Stars']
plot_phases_labels = ['Dark matter', r'Hot CGM', r'Warm CGM',
                      r'Cool CGM', 'Wind', 'ISM', 'Stars']

colours = ['green', 'green', 'brown', 'magenta']
lstyle = ['-', '-', '--', ':']
stats = ['median', 'percentile_25_75', 'std', 'cosmic_median', 'cosmic_std']


minM_ar = [11., 11.6, 12.3, 13]
maxM_ar = [11.6, 12.3, 13, 13.5]


MODEL = ['m100n1024', 'm50n512', 'm25n256', 'm25n512']
WIND = ['s50', 's50', 's50', 's50']
SNAP = ['151']
mod_lab = [r'Simba $100 \, h^{-1} \, \rm cMpc$', r'Simba $50 \, h^{-1} \, \rm cMpc$', r'Simba $25 \, h^{-1} \, \rm cMpc$', r'Simba High-res.',]

fig = plt.figure(figsize=(6*len(minM_ar)+1, 3*len(plot_phases)))
fig.subplots_adjust(wspace=0, hspace=0)

for ip in np.arange(len(plot_phases)):
    phase = plot_phases[ip]
    for im in np.arange(len(minM_ar)):
        min_mass = 10**minM_ar[im]
        max_mass = 10**maxM_ar[im]
        plt_ID = ip * len(minM_ar) + im + 1
        print('plt_ID = ', plt_ID)
        ax = plt.subplot(len(plot_phases),len(minM_ar), plt_ID)
        ax.set_xlim(0.009, 6)
        if (ip==0):
            ax.set_ylim(-0.7, 1.3)
        else:
            ax.set_yscale('log')
            ax.set_ylim(5.e-2, 2.e2)
        ax.set_xscale('log')
        
        for iz in np.arange(len(SNAP)):
            snap = SNAP[iz]
            for i in np.arange(len(WIND)):
                model = MODEL[i]
                wind = WIND[i]
                data_dir = './budgets/'+model+'/'+wind+'/'
                caesarfile = '/disk04/rad/sim/'+model+'/'+wind+'/Groups/'+model+'_'+snap+'.hdf5'
                sim = caesar.quick_load(caesarfile)
                h = sim.simulation.hubble_constant
                z_sim = sim.simulation.redshift
                central = np.array([i.central for i in sim.galaxies])
                halo_m = np.array([i.halo.masses['total'].in_units('Msun') for i in sim.galaxies])[central]
                gal_pos = np.array([i.pos.in_units('kpccm') for i in sim.galaxies])[central]
                print('z = %s'%(z_sim))
                if (im == len(minM_ar)-1):
                    mask = halo_m > min_mass
                else:
                    mask = (halo_m > min_mass) & (halo_m < max_mass)
            
                # get the dennsity profile data:
                density_profile = read_phases(data_dir+'density_profiles_%s_minpotpos.h5'%(snap), all_phases)
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
                print('rbins = %s'%(rbins))
                dens = density_profile[phase]
                dens = dens[mask]
                gal_pos_cut = gal_pos[mask]
                print('After masking: shape of %s = %s'%(phase, np.shape(dens)))
                print('Shape of gal_pos = ',np.shape(gal_pos))
                
                
                for j in np.arange(rbins-1):
                    if (len(dens[:,j])>0):
                        med[j] = np.mean(dens[:,j], dtype=np.float64)
                        sup_per[j] = np.percentile(dens[:,j], 84)
                        inf_per[j] = np.percentile(dens[:,j], 16)
                    else:
                        med[j] = np.nan
                        sup_per[j] = np.nan
                        inf_per[j] = np.nan
                    if (i==0 and iz==0):
                        ref_med[j] = med[j]

                if (i==0):
                    if (ip == 0):
                        ax.axhline(0, c='black', ls=':', lw=1)
                    else:
                        ax.axhline(1, c='black', ls=':', lw=1)
                else:
                    if (ip == 0):
                        ax.plot(10**logr_med[med>0], med[med>0]/ref_med[med>0]-1, color=colours[i], ls=lstyle[i], lw=2)
                    else:
                        ax.plot(10**logr_med[med>0], med[med>0]/ref_med[med>0], color=colours[i], ls=lstyle[i], lw=2)
                if (iz==0):
                    if (i==0):
                        continue
                    ax.axvline(-1, color=colours[i], ls=lstyle[i], lw=2, label=mod_lab[i])
                

        
        if (im == 0):
            if (ip==0):
                ax.set_ylabel(r'$\rho/\rho_{\rm fid \: 100}-1$', fontsize=30, x=1.05)
            else:
                ax.set_ylabel(r'$\rho/\rho_{\rm fid \: 100}$', fontsize=30, x=1.05)
            ax.tick_params(labelsize=22)
            ax.text(-0.3,0.5, plot_phases_labels[ip],horizontalalignment='center', verticalalignment='center',rotation='vertical',fontsize=30, transform=ax.transAxes)
        else:
            ax.tick_params(labelleft=None)
        
                
        if (ip == len(plot_phases)-1):
            ax.set_xlabel(r'$r_{\rm halo} / r_{200}$', fontsize=30)
            ax.tick_params(labelsize=22)
        else:
            ax.tick_params(labelbottom=None)

        if (ip == 0):
            if (im == len(minM_ar)-1):
                ax.set_title(r'$\log(M_{\rm h}/\mathrm{M}_{\odot})>%s$'%(minM_ar[im]), fontsize=28, y=1.05)
                ax.legend(loc='upper left', fontsize=20, ncol=1, framealpha=0)
            else:
                ax.set_title(r'$%s<\log(M_{\rm h}/\mathrm{M}_{\odot})<%s$'%(minM_ar[im], maxM_ar[im]), fontsize=28, y=1.05)

fformat = 'pdf'
plt.savefig(savedir+'dens_prof_diff_conv_noshade.%s'%(fformat), format=fformat, bbox_inches = 'tight')
plt.clf()
