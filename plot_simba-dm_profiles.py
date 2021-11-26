"""
    Plot dark density profiles of Simba fiducial-100 vs Simba-Dark
"""
import matplotlib.pyplot as plt
import numpy as np
import math
import h5py
import caesar
import os
import matplotlib.gridspec as gridspec
from pygadgetreader import readsnap, readheader
from plotting_methods import *
from scipy import optimize

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=14)
palette_name = 'tol'
minM_ar = [11.0, 11.6, 12.2, 12.8]
maxM_ar = [11.6, 12.2, 12.8, 15]
dm = 0.25 # dex
SNAP = ['151']

logr_min = -2 # distance in units of Rvir
logr_max = math.log10(5)
#rbins = 20 # nr of logr-bins
#dlogr = (logr_max-logr_min)/rbins

boxsize = 100000. # ckpc/h
ymin=1.e2 # min of the y-axis

savedir =  '/disk04/sorini/outputs/plots/'

all_phases = ['Cool CGM (T < Tphoto)', 'Warm CGM (Tphoto < T < 0.5Tvir)', 'Hot CGM (T > 0.5Tvir)',
              'Cool CGM (T < 10^5)', 'Warm CGM (10^5 < T < 10^6)', 'Hot CGM (T > 10^6)',
              'ISM', 'Wind', 'Dust', 'Stars', 'Dark matter', 'Total Gas']
phases_comp = ['Dark matter']#, 'Total Gas', 'Stars']
colours = ['m', 'b', 'c', 'g', 'tab:orange', 'tab:pink', 'r', 'black']
colours_down = get_cb_colours(palette_name)[::-1]
colours_up = ['grey', 'blue', '#332288']
stats = ['median', 'percentile_25_75', 'std', 'cosmic_median', 'cosmic_std']


fig=plt.figure(figsize=(5*len(minM_ar)+1, 8))
gs = gridspec.GridSpec(ncols=4, nrows=2, width_ratios=[1,1,1,1], height_ratios=[2,1],wspace=0,hspace=0)
func_ID = [0,1,2,3]

for ii in np.arange(len(SNAP)):
    snap = SNAP[ii]
    
    ### Open full physics run

    data_dir = '/disk04/sorini/outputs/budgets/m100n1024/s50/'
    caesarfile = '/disk04/rad/sim/m100n1024/s50/Groups/m100n1024_'+snap+'.hdf5'
    snapfile = '/disk04/rad/sim/m100n1024/s50/snap_m100n1024_'+snap+'.hdf5'
    sim = caesar.load(caesarfile)
    Om0 = sim.simulation.omega_matter
    Ob0 = sim.simulation.omega_baryon
    fb = Ob0/Om0
    h = sim.simulation.hubble_constant
    boxsize=boxsize/h #ckpc
    central = np.array([i.central for i in sim.galaxies])
    halo_m = np.array([i.halo.masses['total'].in_units('Msun') for i in sim.galaxies])[central]
    print(halo_m)
    gal_pos = np.array([i.pos.in_units('kpccm') for i in sim.galaxies])[central]
    print(gal_pos)
    print(np.shape(gal_pos))

    # get the mass budget data:
    density_profile = read_phases(data_dir+'density_profiles_%s_minpotpos.h5'%(snap), all_phases)
    print('Shape of %s = %s'%('Dark matter', np.shape(density_profile['Dark matter'])))
    dens_hydro = np.zeros(np.shape(density_profile['Dark matter']))
    for i in np.arange(len(halo_m)):
        #    for phase in phases_comp:
        #        dens_hydro[i,:] += density_profile[phase][i,:]
        dens_hydro[i, :] = density_profile['Dark matter'][i, :]

    ### Open DM-only run

    data_dir = '/disk04/sorini/outputs/budgets/m100n1024/DMS/'
    snapfile = '/disk04/sorini/inputs/simba_DMS/snap_m100n1024_%s.hdf5'%(snap)
    caesarfile = '/disk04/sorini/inputs/simba_DMS/Groups/Caesar_snap_%s.hdf5'%(snap)
    sim = caesar.quick_load(caesarfile)
    h = sim.simulation.hubble_constant
    print('fb=%s'%(fb))
    z_sim = sim.simulation.redshift
    boxsize=readheader(snapfile,'boxsize')
    boxsize=boxsize/h
    central = np.array([i.central for i in sim.galaxies])
    dm_halo_m = np.array([i.masses['total'].in_units('Msun') for i in sim.halos])
    halo_pos = np.array([i.minpotpos.in_units('kpccm') for i in sim.halos])
    print(halo_pos)
    print(np.shape(halo_pos))

    # get the mass budget data:
    dm_profile = read_phases(data_dir+'density_profiles_%s_minpotpos_DMS.h5'%(snap), ['Dark matter'])
    print('Shape of dm_profile = ', np.shape(dm_profile['Dark matter']))

    rbins = np.shape(dm_profile['Dark matter'])[1]
    all_bins = np.linspace(logr_min, logr_max, rbins+1)
    #logr_med = 0.5*(all_bins[:-1]+all_bins[1:])
    #print('rbins = %s len(med) = %s'%(rbins, len(med)))

    for im in np.arange(len(minM_ar)):
        logr_med = 0.5*(all_bins[:-1]+all_bins[1:])
        med = np.zeros(rbins)
        stdev = np.zeros(rbins)
        stdev_inf = np.zeros(rbins)
        med_dark = np.zeros(rbins)
        stdev_dark = np.zeros(rbins)
        stdev_inf_dark = np.zeros(rbins)
        print('rbins = %s len(med) = %s'%(rbins, len(med)))
        fID = func_ID[len(SNAP)*ii+im]
        #ax = plt.subplot(2, len(minM_ar), im+2)
        ax=fig.add_subplot(gs[fID])
        min_mass = 10**minM_ar[im]
        max_mass = 10**maxM_ar[im]
        mask = (halo_m > min_mass) & (halo_m < max_mass)
        mask_dark = (dm_halo_m > min_mass) & (dm_halo_m < max_mass)
        if (ii == 0):
            if (im == len(minM_ar)-1):
                ax.set_title(r'$\log(M_{\rm h}/\mathrm{M}_{\odot})>%s$'%(minM_ar[im]), fontsize=25, y=1.05)
            else:
                ax.set_title(r'$%s<\log(M_{\rm h}/\mathrm{M}_{\odot})<%s$'%(minM_ar[im], maxM_ar[im]), fontsize=25, y=1.05)

        dens = dens_hydro[mask]
        dens_dark = dm_profile['Dark matter']
        dens_dark = dens_dark[mask_dark]
        gal_pos_cut = gal_pos[mask]
        halo_pos_cut = halo_pos[mask_dark]

        for j in np.arange(rbins):
            if (len(dens[:,j])>0):
                med[j] = np.mean(dens[:,j], dtype=np.float64)
                stdev[j] = np.std(dens[:, j], dtype=np.float64, ddof=1)
            else:
                med[j] = np.nan
                stdev[j] = np.nan
            if (len(dens_dark[:,j])>0):
                med_dark[j] = np.mean(dens_dark[:,j], dtype=np.float64)/(1+fb)
                stdev_dark[j] = np.std(dens_dark[:, j], dtype=np.float64, ddof=1)/(1+fb)
            else:
                med_dark[j] = np.nan
                stdev_dark[j] = np.nan


        ax.errorbar(10**logr_med[med>0], med[med>0], yerr=[stdev_inf[med>0], stdev[med>0]], capsize=0, color='teal', label=r'Simba $100 \, h^{-1} \, \rm cMpc$')
        ax.errorbar(10**logr_med, med_dark, yerr=[stdev_inf_dark, stdev_dark], capsize=0, color='black', label='Simba-Dark')

        ax.set_xlim(0.04, 6)
        ax.set_ylim(ymin, 1.e7)
        plt.yscale('log')
        plt.xscale('log')
        ax.set_xlabel(r'$r / r_{200}$', fontsize=25)
        if (im == 0):
            ax.set_ylabel(r'$\rho$  $\rm (M_{\odot} \, kpc^{-3})$', fontsize=25)
            if (ii==0):
                ax.legend(loc='lower left', framealpha=0, fontsize=18)
        else:
            ax.tick_params(labelleft=False)
        ax.tick_params(labelsize=20, labelbottom=False)


        ax=fig.add_subplot(gs[fID+len(minM_ar)])

        ax.plot(10**logr_med[med>0], med[med>0]/med_dark[med>0]-1, lw=2, c='black')
        ax.axhline(0, c='black', lw=1, ls=':')
        
        ax.set_xlim(0.04, 6)
        ax.set_ylim(-0.25, 0.4)
        plt.yscale('linear')
        plt.xscale('log')
        ax.set_xlabel(r'$r / r_{200}$', fontsize=25)
        if (im == 0):
            ax.set_ylabel(r'$\rho_{\rm fid \, 100}/\rho_{\rm Dark}-1$', fontsize=25)


        ax.tick_params(labelsize=20)
        if (im > 0):
            ax.tick_params(labelsize=20, labelleft=False)
        if (ii<len(SNAP)-1):
            ax.tick_params(labelsize=20, labelbottom=False)

fformat = 'pdf'
plt.savefig(savedir+'dprof_compare_dark-full.%s'%(fformat), format=fformat, bbox_inches = 'tight')
plt.clf()
