import matplotlib.pyplot as plt
import numpy as np
import h5py
import caesar
import sys
import os
import math
from pygadgetreader import readsnap, readheader
from plotting_methods import *
from scipy import spatial
from astropy.io import ascii

def time(z, Om0, OL0, H0, F):
    Z = 1+z
    if (F==0):
        t = (2./(3*H0)) * Z**-1.5
    else:
        ### Convert from z to cosmic time
        # Here Z is 1+z
        t = (2./(3.*math.sqrt(OL0)*H0)) * math.asinh( math.sqrt(OL0/Om0) * Z**-1.5 )
    return t/Gyr2sec

def time_np(z, Om0, OL0, H0, F):
    Z = 1+z
    if (F==0):
        t = (2./(3*H0)) * Z**-1.5
    else:
        ### Convert from z t cosmic time if z is an array
        # Here Z is 1+z
        t = (2./(3.*math.sqrt(OL0)*H0)) * np.arcsinh( math.sqrt(OL0/Om0) * Z**-1.5 )
    return t/Gyr2sec # time in Gyr

def inv_time(t, Om0, OL0, H0, F):
    if (F==0):
        Z = (3*H0*t*Gyr2sec/2.)**(-2./3.)
    else:
        ### Convert from cosmic time to 1+z (here t is an array)
        a = (Om0/OL0)**(1./3.) * (np.sinh(1.5*math.sqrt(OL0)*H0*t*Gyr2sec))**(2./3.)
        Z = 1./a
    return Z-1


plt.rcParams['hatch.color']='white'
plt.rcParams['hatch.linewidth']=1
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=14)

Mpc2km = 3.0856776e19
km2Mpc = 1./Mpc2km
yr2sec = 365.25*24*3600.
Myr2sec=1.e6*yr2sec
Gyr2sec = 1.e9*yr2sec

OL = 0.7
Om = 1 - OL
Ob = 0.047
h = 0.685
H0 = 100*h*km2Mpc
fb = Ob/Om
print('Reference cosmology:')
print('Om0=%s\nOL0=%s\nOb0=%s\nfb=%s\n'%(Om, OL, Ob, fb))


model = 'm50n512'
WIND=['s50nox', 's50nojet', 's50noagn', 's50nofb']
TITLE = ['No-X-ray', 'No-jet', 'No-AGN', 'No-feedback']
pdfpath='/disk04/sorini/outputs/plots/'
alpha = .8
palette_name = 'okabe'
plot_phases_labels = [r'$T>10^7 \, \rm K$', r'$10^6 \, \rm K < T<10^7 \, \rm K$', r'$10^5 \, \rm K < T<10^6 \, \rm K$', r'$T<10^5 \, \rm K$']
hatches = [None, '/', '|', None, '.']
plot_phases = ['>10^7 K', '10^6-10^7 K', '10^5-10^6 K', '<10^5 K']
colours = get_cb_colours(palette_name)[::-1]
phases = ['>10^7 K', '10^6-10^7 K', '10^5-10^6 K', '<10^5 K']


fig = plt.figure(figsize=(6*len(WIND), 4))
fig.subplots_adjust(wspace=0)

for j in np.arange(len(WIND)):
    ax = plt.subplot(1,4,j+1)
    wind = WIND[j]
    plot_title = TITLE[j]
    data = ascii.read('fIGM-T_%s%s.dat'%(model, wind))
    z = data['z']
    omega_frac = {phase: np.zeros(len(z)) for phase in phases}
    for phase in phases:
        omega_frac[phase] = data[phase]
    plt.title(plot_title, fontsize = 25, y=1.2)
    running_total = np.zeros(len(z))
    for i, phase in enumerate(plot_phases):
        ax.fill_between(1+z, running_total, running_total + omega_frac[phase],
                        color=colours[i], label=plot_phases_labels[i], alpha=alpha, hatch=hatches[i])
        running_total += omega_frac[phase]

    #print(z)
    #print(running_total)
    ax.set_xscale('log')
    ax.set_xlim(7, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel(r'$z$', fontsize=20)
    if (j==0):
        ax.set_ylabel(r'$f_{\rm IGM}$', fontsize=20, x=0.95)
        ax.legend(loc='center left', fontsize=16, framealpha=1.)
        ax.tick_params(labelsize=18)
        ax.set_xticks([7,6,5,4,3,2,1])
        ax.set_xticklabels([6,5,4,3,2,1,0])
    else:
        ax.tick_params(labelsize=18, labelleft=False)
        ax.set_xticks([6,5,4,3,2,1])
        ax.set_xticklabels([5,4,3,2,1,0])

    axbis=ax.twiny()
    xbounds=np.zeros(2)
    xbounds=ax.get_xlim()
    axbis.set_xlim(np.log10(xbounds))
    tticks = [1,2,3,4,5,6,7,8,9,10,11,12,13]
    tlabels=['']*len(tticks)

    tlabels[0]='1'
    tlabels[2]='3'
    tlabels[4]='5'
    tlabels[6]='7'
    tlabels[8]='9'
    tlabels[10]='11'
    tlabels[12]='13'

    axbis.set_xticks(np.log10(1+inv_time(np.asarray(tticks), Om, OL, 100*h*km2Mpc, 1)))
    axbis.set_xticklabels(tlabels)
    axbis.set_xlabel(r'$t$($\rm Gyr$)',fontsize=20)
    axbis.tick_params(pad=6, labelsize=18, length=8)

fformat='pdf'
plt.savefig(pdfpath+'fIGM-T_all.%s'%(fformat), format=fformat, bbox_inches = 'tight')
