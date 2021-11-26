"""
    Get mass and metal budgets in halos.
"""

import numpy as np
import h5py
import caesar
import sys
from pygadgetreader import readsnap

def ism_phase_line(nH):
    # ISM particles have:
    # log T  = 4 + 1/3 log nH  (+ 1 dex)
    return 5. + 0.33*nH

def get_ism_mask(temp, nH, ism_density):
    nH_mask = nH > ism_density
    ism_line = ism_phase_line(np.log10(nH))
    temp_mask = (np.log10(temp) - ism_line < 0.)
    return temp_mask * nH_mask


photo_temp = 10.**4.5 # in K
cold_temp = 1.e5
hot_temp = 1.e6
ism_density = 0.13 # hydrogen number density, cm**-3
ism_sfr = 0.
dust_mass_factor = 1.e10
omega_b = 0.048
omega_m = 0.3
f_baryon = omega_b / omega_m

model = sys.argv[1]
wind = sys.argv[2]
snap = sys.argv[3]

datadir = '/disk04/rad/sim/'+model+'/'+wind+'/'
snapfile = datadir + 'snap_'+model+'_'+snap+ '.hdf5'
caesarfile = datadir + 'Groups/'+model+'_'+snap+'.hdf5'
savedir = '/disk04/sorini/outputs/budgets/'+model+'/'+wind+'/'
sim = caesar.quick_load(caesarfile)


h = sim.simulation.hubble_constant

central = np.array([i.central for i in sim.galaxies])
gal_sm = np.array([i.masses['stellar'].in_units('Msun') for i in sim.galaxies])
gal_sfr = np.array([i.sfr.in_units('Msun/yr') for i in sim.galaxies])
gal_tvir = np.array([i.halo.virial_quantities['temperature'].in_units('K') for i in sim.galaxies])
gal_ssfr = gal_sfr / gal_sm
gal_sm = np.log10(gal_sm)
gal_ssfr = np.log10(gal_ssfr)

dm_mass = readsnap(snapfile, 'mass', 'dm', suppress=1, units=1) / h # in Mo)
gas_mass = readsnap(snapfile, 'mass', 'gas', suppress=1, units=1) / h # in Mo
gas_sfr = readsnap(snapfile, 'sfr', 'gas', suppress=1, units=1) # in Mo/yr
gas_z = readsnap(snapfile, 'z', 'gas', suppress=1, units=1)
gas_nh = readsnap(snapfile, 'nh', 'gas', suppress=1, units=1) # in g/cm^3
gas_delaytime = readsnap(snapfile, 'DelayTime', 'gas', suppress=1)
gas_temp = readsnap(snapfile, 'u', 'gas', suppress=1, units=1) # in K
dust_mass = readsnap(snapfile, 'Dust_Masses', 'gas', suppress=1, units=1) * dust_mass_factor/ h # in Mo
star_mass = readsnap(snapfile, 'mass', 'star', suppress=1, units=1) / h # in Mo
star_z = readsnap(snapfile, 'z', 'star', suppress=1, units=1)

phases = ['Cool CGM (T < 10^5)', 'Warm CGM (10^5 < T < 10^6)', 'Hot CGM (T > 10^6)',
          'Cool CGM (T < Tphoto)', 'Warm CGM (Tphoto < T < 0.5Tvir)', 'Hot CGM (T > 0.5Tvir)',
          'ISM', 'Wind', 'Dust', 'Stars', 'Dark matter', 'Total baryons']
mass_budget = {phase: np.zeros(len(sim.galaxies)) for phase in phases}
metal_budget = {phase: np.zeros(len(sim.galaxies)) for phase in phases}
del metal_budget['Dark matter']

print('Finished loading in particle data arrays')

for i in range(len(sim.galaxies)):
    
    if not sim.galaxies[i].central:
        continue
    else:
        glist = sim.galaxies[i].halo.glist
        slist = sim.galaxies[i].halo.slist
        dmlist = sim.galaxies[i].halo.dmlist
        
        ism_gas_mask = get_ism_mask(gas_temp[glist], gas_nh[glist], ism_density)
        cgm_gas_mask = np.invert(ism_gas_mask)
        wind_mask = gas_delaytime[glist] > 0.
        
        cool_gas_mask = cgm_gas_mask & np.invert(wind_mask) & (gas_temp[glist] < cold_temp)
        warm_gas_mask = cgm_gas_mask & np.invert(wind_mask) & (gas_temp[glist] > cold_temp) & (gas_temp[glist] < hot_temp)
        hot_gas_mask = cgm_gas_mask & np.invert(wind_mask) & (gas_temp[glist] > hot_temp)
        mass_budget['Cool CGM (T < 10^5)'][i] = np.sum(gas_mass[glist][cool_gas_mask])
        mass_budget['Warm CGM (10^5 < T < 10^6)'][i] = np.sum(gas_mass[glist][warm_gas_mask])
        mass_budget['Hot CGM (T > 10^6)'][i] = np.sum(gas_mass[glist][hot_gas_mask])
        metal_budget['Cool CGM (T < 10^5)'][i] = np.sum(gas_mass[glist][cool_gas_mask] * gas_z[glist][cool_gas_mask])
        metal_budget['Warm CGM (10^5 < T < 10^6)'][i] = np.sum(gas_mass[glist][warm_gas_mask] * gas_z[glist][warm_gas_mask])
        metal_budget['Hot CGM (T > 10^6)'][i] = np.sum(gas_mass[glist][hot_gas_mask] * gas_z[glist][hot_gas_mask])
        
        cool_gas_mask = cgm_gas_mask & np.invert(wind_mask) & (gas_temp[glist] < photo_temp)
        warm_gas_mask = cgm_gas_mask & np.invert(wind_mask) & (gas_temp[glist] > photo_temp) & (gas_temp[glist] < 0.5*gal_tvir[i])
        hot_gas_mask = cgm_gas_mask & np.invert(wind_mask) & (gas_temp[glist] > 0.5*gal_tvir[i])
        mass_budget['Cool CGM (T < Tphoto)'][i] = np.sum(gas_mass[glist][cool_gas_mask])
        mass_budget['Warm CGM (Tphoto < T < 0.5Tvir)'][i] = np.sum(gas_mass[glist][warm_gas_mask])
        mass_budget['Hot CGM (T > 0.5Tvir)'][i] = np.sum(gas_mass[glist][hot_gas_mask])
        metal_budget['Cool CGM (T < Tphoto)'][i] = np.sum(gas_mass[glist][cool_gas_mask] * gas_z[glist][cool_gas_mask])
        metal_budget['Warm CGM (Tphoto < T < 0.5Tvir)'][i] = np.sum(gas_mass[glist][warm_gas_mask] * gas_z[glist][warm_gas_mask])
        metal_budget['Hot CGM (T > 0.5Tvir)'][i] = np.sum(gas_mass[glist][hot_gas_mask] * gas_z[glist][hot_gas_mask])
        
        mass_budget['ISM'][i] = np.sum(gas_mass[glist][np.invert(cgm_gas_mask) & np.invert(wind_mask)])
        mass_budget['Wind'][i] = np.sum(gas_mass[glist][wind_mask])
        mass_budget['Dust'][i] = np.sum(dust_mass[glist][np.invert(wind_mask)])
        mass_budget['Stars'][i] = np.sum(star_mass[slist])
        mass_budget['Dark matter'][i] = np.sum(dm_mass[dmlist])
        metal_budget['ISM'][i] = np.sum(gas_mass[glist][np.invert(cgm_gas_mask) & np.invert(wind_mask)] * gas_z[glist][np.invert(cgm_gas_mask) & np.invert(wind_mask)])
        metal_budget['Wind'][i] = np.sum(gas_mass[glist][wind_mask] * gas_z[glist][wind_mask])
        metal_budget['Dust'][i] = np.sum(dust_mass[glist][np.invert(wind_mask)])
        metal_budget['Stars'][i] = np.sum(star_mass[slist] * star_z[slist])
        
        mass_budget['Total baryons'][i] = np.sum(gas_mass[glist]) + np.sum(dust_mass[glist]) + np.sum(star_mass[slist])
        metal_budget['Total baryons'][i] = np.sum(gas_mass[glist] * gas_z[glist]) + np.sum(dust_mass[glist]) + np.sum(star_mass[slist] * star_z[slist])

mass_budget = {k: p[central] for k, p in mass_budget.items()}
metal_budget = {k: p[central] for k, p in metal_budget.items()}

available_mass_fractions = {k: p/mass_budget['Total baryons'] for k, p in mass_budget.items()}
available_mass_fractions['Total baryons'] = mass_budget['Total baryons'].copy()
del available_mass_fractions['Dark matter']

cosmic_baryons = (mass_budget['Total baryons'] + mass_budget['Dark matter']) * f_baryon
omega_mass_fractions = {k: p/cosmic_baryons for k, p in mass_budget.items()}
omega_mass_fractions['Cosmic baryon mass'] = cosmic_baryons.copy()
del omega_mass_fractions['Dark matter']

available_metal_fractions = {k: p/metal_budget['Total baryons'] for k, p in metal_budget.items()}
available_metal_fractions['Total baryons'] = metal_budget['Total baryons'].copy()

metallicities = {k: p/mass_budget[k] for k, p in metal_budget.items()}

with h5py.File(savedir+'mass_budget_%s.h5'%(snap), 'w') as hf:
    for k, p in mass_budget.items():
        hf.create_dataset(k, data=np.array(p))

with h5py.File(savedir+'available_mass_fraction_%s.h5'%(snap), 'w') as hf:
    for k, p in available_mass_fractions.items():
        hf.create_dataset(k, data=np.array(p))

with h5py.File(savedir+'omega_mass_fraction_%s.h5'%(snap), 'w') as hf:
    for k, p in omega_mass_fractions.items():
        hf.create_dataset(k, data=np.array(p))

with h5py.File(savedir+'metal_budget_%s.h5'%(snap), 'w') as hf:
    for k, p in metal_budget.items():
        hf.create_dataset(k, data=np.array(p))

with h5py.File(savedir+'available_metal_fraction_%s.h5'%(snap), 'w') as hf:
    for k, p in available_metal_fractions.items():
        hf.create_dataset(k, data=np.array(p))

with h5py.File(savedir+'metallicities_%s.h5'%(snap), 'w') as hf:
    for k, p in metallicities.items():
        hf.create_dataset(k, data=np.array(p))
