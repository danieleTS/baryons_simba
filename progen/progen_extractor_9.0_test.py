#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 24 December 2018

@author: currorodriguez
"""

# Import required libraries
import caesar
import numpy as np
import pylab as plt
import os
from functools import reduce
from astropy.cosmology import FlatLambdaCDM

caesarfile_0 = '/home/rad/data/m50n512/s49/Groups/m50n512_151.hdf5' #input('Final group file: ')
progenref_file = '/home/rad/data/m50n512/s49/Groups/progen_m50n512_151.dat'

sim_0 = caesar.load(caesarfile_0,LoadHalo=False)

progenref = open(progenref_file, 'r').readlines()
lengal = int(progenref[0].split(' ')[0])
progenref_data = []
lines = int((len(progenref)-1)/(2*lengal))
for galaxy in range(0, lengal):

    start = galaxy*lines + 1
    end = start + lines

    super_line = reduce(lambda x,y:x+y,[progenref[line] for line in range(start, end)])
    super_line = super_line.replace('[', ' ')
    super_line = super_line.replace(']', ' ')

    super_line = super_line.split()


    progenref_data.append([int(x) for x in super_line])


d = {}
for j in range(0, lengal):
    d['m_gal'+str(j)] = np.array([])
    d['sfr_gal'+str(j)] = np.array([])
    d['fgas_gal'+str(j)] = np.array([])
    d['t_gal'+str(j)] = np.array([])
    d['z_gal'+str(j)] = np.array([])
    d['sfe_gal'+str(j)] = np.array([])
    d['gal_type'+str(j)] = np.array([])

n_snap = int(caesarfile_0[-8:-5])

count = 0
for snap in range(0, n_snap-25):
    loc = n_snap - snap
    if loc<100:
        caesarfile = caesarfile_0[:-8]+'0'+str(loc)+'.hdf5'
    else:
        caesarfile = caesarfile_0[:-8]+str(loc)+'.hdf5'
    if not os.path.isfile(caesarfile): continue

    sim = caesar.load(caesarfile,LoadHalo=False) # load caesar file

    # initialize simulation parameters
    redshift = sim.simulation.redshift  # this is the redshift of the simulation output
    h = sim.simulation.hubble_constant  # this is the hubble parameter = H0/100
    cosmo = FlatLambdaCDM(H0=100*sim.simulation.hubble_constant, Om0=sim.simulation.omega_matter, Ob0=sim.simulation.omega_baryon,Tcmb0=2.73)  # set our cosmological parameters
    thubble = cosmo.age(redshift).value  # age of universe at this redshift

    # Get galaxy info
    gals = np.asarray([i.central for i in sim.galaxies])   # read in galaxies from caesar file
    ms = np.asarray([i.masses['stellar'] for i in sim.galaxies])   # read in stellar masses of galaxies
    mHI = np.asarray([i.masses['HI'] for i in sim.galaxies])   # read in neutral hydrogen masses
    mH2 = np.asarray([i.masses['H2'] for i in sim.galaxies])   # read in molecular hydrogen
    sfr = np.asarray([i.sfr for i in sim.galaxies])   # read in instantaneous star formation rates
    try:
        pg_snap = np.asarray([i.progen_index for i in sim.galaxies]) # read in the progen index for all galaxies
    except:
        print 'No progen_index in caesarfile; using progenref_file'

    print snap,count,lengal,progenref_data[0][count]
    for k in range(0, lengal):
        if snap==0:
            d['m_gal'+str(k)] = np.concatenate((d['m_gal'+str(k)], ms[k]), axis=None)
            d['sfr_gal'+str(k)] = np.concatenate((d['sfr_gal'+str(k)],sfr[k]), axis=None)
            frac = (mHI[k] + mH2[k])/ms[k]
            d['fgas_gal'+str(k)] = np.concatenate((d['fgas_gal'+str(k)],frac), axis=None)
            sfe = sfr[k]/(mHI[k] + mH2[k])
            d['sfe_gal'+str(k)] = np.concatenate((d['sfe_gal'+str(k)],sfe),axis=None)
            d['t_gal'+str(k)] = np.concatenate((d['t_gal'+str(k)], thubble), axis=None)
            d['z_gal'+str(k)] = np.concatenate((d['z_gal'+str(k)], redshift), axis=None)
            d['gal_type'+str(k)] = np.concatenate((d['gal_type'+str(k)],gals[k]), axis=None)

        else:

            if progenref_data[k][count]==-1 or progenref_data[k][count]>=len(ms):
                continue
            else:
                index = progenref_data[k][count]
                d['m_gal'+str(k)] = np.concatenate((d['m_gal'+str(k)], ms[index]), axis=None)
                d['sfr_gal'+str(k)] = np.concatenate((d['sfr_gal'+str(k)],sfr[index]), axis=None)
                frac = (mHI[index] + mH2[index])/ms[index]
                d['fgas_gal'+str(k)] = np.concatenate((d['fgas_gal'+str(k)],frac), axis=None)
                sfe = sfr[index]/(mHI[index] + mH2[index])
                d['sfe_gal'+str(k)] = np.concatenate((d['sfe_gal'+str(k)],sfe),axis=None)
                d['t_gal'+str(k)] = np.concatenate((d['t_gal'+str(k)], thubble), axis=None)
                d['z_gal'+str(k)] = np.concatenate((d['z_gal'+str(k)], redshift), axis=None)
                d['gal_type'+str(k)] = np.concatenate((d['gal_type'+str(k)],gals[index]), axis=None)
    if snap>0: count += 1



print('Progen data extracted from simulation.')
########################################################################################################
# Galaxy that you want to have a look at
galaxy = 780

fig,ax = plt.subplots(2)
for galaxy in [0,10,100,200,780]:
    ms = d['m_gal' + str(galaxy)]
    ms_gal = np.log10(ms[::-1])
    ssfr = d['sfr_gal' + str(galaxy)]/d['m_gal'+str(galaxy)]
    ssfr_gal = np.log10(ssfr[::-1]+1e-14)
    galaxy_t = (d['t_gal' + str(galaxy)][::-1])
    ax[0].plot(galaxy_t, ms_gal, label='gal %d'%galaxy)
    ax[1].plot(galaxy_t, ssfr_gal, label='gal %d'%galaxy)
    ax[0].set_ylabel(r'$\log{M_*}$')
    ax[1].set_ylabel(r'$\log{sSFR}$')

plt.xlabel(r'${t}$')
plt.legend()
plt.show()
