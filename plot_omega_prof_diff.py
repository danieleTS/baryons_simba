"""
    Find the radius r_0.9rbb at which the enclosed baryonic mass M_b(<r_0.9fb) equals
    0.9 \times f_b \times M(<r). Plot the results as a function of halo mass bin,
    for different Simba runs and redshift.
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

Delta=200
alpha=.8
palette_name = 'tol'
dm = 0.25 # dex
logr_min = -2 # distance in units of Rvir
logr_max = math.log10(50)
#logr_max = 1

model = sys.argv[1]

mod_lab = [r'Simba $50 \, h^{-1} \, \rm cMpc$', 'No-X-ray', 'No-jet', 'No-AGN', 'No-feedback']


savedir =  '/disk01/sorini/outputs/plots/'

all_phases = ['Cool CGM (T < 10^5)', 'Warm CGM (10^5 < T < 10^6)', 'Hot CGM (T > 10^6)',
              'Cool CGM (T < Tphoto)', 'Warm CGM (Tphoto < T < 0.5Tvir)', 'Hot CGM (T > 0.5Tvir)',
              'ISM', 'Wind', 'Dust', 'Stars', 'Total gas', 'Total baryons','Total','Black holes']
plot_phases = ['Hot CGM (T > 0.5Tvir)', 'Warm CGM (Tphoto < T < 0.5Tvir)', 'Cool CGM (T < Tphoto)',
               'Wind', 'Dust', 'ISM', 'Stars','Black holes']


plot_phases_labels = [r'Hot CGM', r'Warm CGM',
                      r'Cool CGM', 'Wind', 'Dust', 'ISM', 'Stars', 'Black holes']

colours = ['green', 'red', 'blue', 'purple', 'orange']
stats = ['median', 'percentile_25_75', 'std', 'cosmic_median', 'cosmic_std']


WIND = ['s50j7k', 's50nox', 's50nojet', 's50noagn', 's50nofb']
lstyle=['-', '-.', '--', ':']
maxM_ar = np.array([11.6,12.2, 12.8, 15])
minM_ar = np.array([11.,11.6,12.2, 12.8])
M_med = 0.5*(minM_ar+maxM_ar)
fb_th = 0.9
r_th = np.zeros(len(M_med)) # radius where Mb(<r)/Mtot(<r)=fb_th*fb
r_sc1 = [0.98, 0.99, 1, 1.01, 1.02]
# artificial % scatter along y-axis to avoid superposition
# of too many lines and make plot more readable


SNAP = ['151', '105', '078', '051']

fig = plt.figure(figsize=(6,6))
fig.subplots_adjust(wspace=0, hspace=0)
ax = plt.subplot(1,1,1)
for i in np.arange(len(WIND)):
    wind = WIND[i]
    for iz in np.arange(len(SNAP)):
        snap = SNAP[iz]
        
        data_dir = '/disk04/sorini/outputs/budgets/'+model+'/'+wind+'/'
        caesarfile = '/disk04/rad/sim/'+model+'/'+wind+'/Groups/'+model+'_'+snap+'.hdf5'
        sim = caesar.quick_load(caesarfile)
        h = sim.simulation.hubble_constant
        z_sim = sim.simulation.redshift
        
        # Read galaxies data
        central = np.array([i.central for i in sim.galaxies])
        halo_m = np.array([i.halo.masses['total'].in_units('Msun') for i in sim.galaxies])[central]
        gal_pos = np.array([i.pos.in_units('kpccm') for i in sim.galaxies])[central]


        # Get the omega profiles
        phase = 'Total baryons'
        omega_profile = read_phases(data_dir+'omega_frac_profiles_%s_bh.h5'%(snap), all_phases)
        print('Shape of %s = %s'%(phase, np.shape(omega_profile[phase])))
        rbins = np.shape(omega_profile[phase])[1]
        all_bins = np.linspace(logr_min, logr_max, rbins)
        logr_med = 0.5*(all_bins[:-1]+all_bins[1:])
        med = np.zeros(rbins-1)
        print('rbins = %s'%(rbins))
        omega = omega_profile[phase]
        
        # Filter by halo mass
        for im in np.arange(len(minM_ar)):
            min_mass = 10**minM_ar[im]
            max_mass = 10**maxM_ar[im]
            if (im == len(minM_ar)-1):
                mask = halo_m > min_mass
                # define bin centre for the overflow
                if (len(halo_m[mask])>0):
                    M_med[im] = 0.5*(minM_ar[-1]+math.log10(np.amax(halo_m[mask])))
                else:
                    M_med[im] = minM_ar[im]
            else:
                mask = (halo_m > min_mass) & (halo_m < max_mass)
            omega_cut = omega[mask]

        
            # Calculate median omega profile within halo mass binn
            for j in np.arange(rbins-1):
                if (len(omega_cut[:,j])>0):
                    med[j] = np.nanmedian(omega_cut[:,j])
                else:
                    med[j] = np.nan

            # Get r_th
            def_th = np.logical_and(med>=fb_th, logr_med>0)
            if (len(logr_med[def_th])>0):
                r_th[im] = logr_med[def_th][0]
                if ((r_th[im] == logr_med[logr_med>0][0]) and (med[np.where(logr_med==logr_med[logr_med<0][-1])[0]]>fb_th)):
                    r_th[im] = 0
            else:
                r_th[im] = np.nan

        plt.plot(M_med, r_sc1[i]*10**r_th, c=colours[i], ls=lstyle[iz])
        if (iz==0):
            plt.axvline(-1, c=colours[i], label=mod_lab[i])

        if (i==len(WIND)-1):
            plt.axhline(-1, c='grey', ls=lstyle[iz], label=r'$z=%g$'%(round(z_sim,1)))

ax.set_xlim(11,14)
ax.set_ylim(0.8, 30)
ax.set_yscale('log')
ax.legend(loc='lower left', fontsize=18, framealpha=1, ncol=2, bbox_to_anchor=(-0.015, 1.02))
ax.set_xlabel(r'$\log(M / \rm M_{\odot})$', fontsize=20)
ax.tick_params(labelsize=16)
ax.set_ylabel(r'$r_{ %s f_{\rm b}}/r_{200}$'%(fb_th), fontsize=20, x=1.05)
ax.tick_params(labelsize=16)



fformat = 'pdf'
plt.savefig(savedir+'omega_prof_diff_%s.%s'%(fb_th, fformat), format=fformat, bbox_inches = 'tight')
