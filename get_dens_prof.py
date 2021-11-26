"""
    Extract radial density profile of various components of halos in Simba
"""
import numpy as np
import h5py
import caesar
import sys
import math
from pygadgetreader import readsnap, readheader
from scipy import spatial

def ism_phase_line(nH):
    # ISM particles have:
    # log T  = 4 + 1/3 log nH  (+ 1 dex)
    return 5. + 0.33*nH

def get_ism_mask(temp, nH, ism_density):
    nH_mask = nH > ism_density
    ism_line = ism_phase_line(np.log10(nH))
    temp_mask = (np.log10(temp) - ism_line < 0.)
    return temp_mask * nH_mask

def sphere(r):
    return (4./3.)*np.pi*r**3

logr_min = -2 # log-distance in units of Rvir
logr_max = math.log10(5)
rbins = 20 # nr of logr-bins
dlogr = (logr_max-logr_min)/(rbins-1)
print('rbins = %s logr_min=%s logr_max=%s dlogr=%s'%(rbins, logr_min, logr_max, dlogr))

photo_temp = 10.**4.5 # in K
cold_temp = 1.e5
hot_temp = 1.e6
ism_density = 0.13 # hydrogen number density, cm**-3


model = sys.argv[1]
wind = sys.argv[2]
snap = sys.argv[3]

datadir = '/disk04/rad/sim/'+model+'/'+wind+'/'
snapfile = datadir + 'snap_'+model+'_'+snap+ '.hdf5'
caesarfile = datadir + 'Groups/'+model+'_'+snap+'.hdf5'
savedir = '/disk04/sorini/outputs/budgets/'+model+'/'+wind+'/'

sim = caesar.quick_load(caesarfile)
redshift = sim.simulation.redshift
h = sim.simulation.hubble_constant
print('h=%s'%(h))
#Lbox = 100.
#Lbox = readheader(snapfile,'boxsize')
if (model=='m100n1024'):
    Lbox = 100000.
if (model=='m50n512'):
    Lbox=50000.
if (model=='m25n256' or model=='m25n512'):
    Lbox=25000.
Lbox = Lbox/h # ckpc
print('Lbox = %s ckpc z=%s'%(Lbox,redshift))

central = np.array([i.central for i in sim.galaxies])
gal_sfr = np.array([i.sfr.in_units('Msun/yr') for i in sim.galaxies])
gal_tvir = np.array([i.halo.virial_quantities['temperature'].in_units('K') for i in sim.galaxies])
gal_rvir = np.array([i.halo.virial_quantities['r200c'].in_units('kpccm') for i in sim.galaxies])
Ngal = len(central)
Ngal_c = len(gal_rvir[central])

dm_mass = readsnap(snapfile, 'mass', 'dm', suppress=1, units=1) / h # in Mo
dm_pos = readsnap(snapfile, 'pos', 'dm', suppress=1, units=1) /h

# Gas info
gas_mass = readsnap(snapfile, 'mass', 'gas', suppress=1, units=1) / h # in Mo
gas_nh = readsnap(snapfile, 'nh', 'gas', suppress=1, units=1) # in g/cm^3
gas_delaytime = readsnap(snapfile, 'DelayTime', 'gas', suppress=1)
gas_temp = readsnap(snapfile, 'u', 'gas', suppress=1, units=1) # in K
gas_pos = readsnap(snapfile, 'pos', 'gas', suppress=1, units=1) /h # in ckpc

# Stars info
star_mass = readsnap(snapfile, 'mass', 'star', suppress=1, units=1) / h # in Mo
star_pos = readsnap(snapfile, 'pos', 'star', suppress=1, units=1)/h # in ckpc

# Prepare density profile arrays
phases = ['Cool CGM (T < 10^5)', 'Warm CGM (10^5 < T < 10^6)', 'Hot CGM (T > 10^6)',
          'Cool CGM (T < Tphoto)', 'Warm CGM (Tphoto < T < 0.5Tvir)', 'Hot CGM (T > 0.5Tvir)',
          'ISM', 'Wind', 'Stars', 'Dark matter', 'Total Gas']
density_profile = {phase: np.zeros((len(sim.galaxies),rbins-1)) for phase in phases}

print('Finished loading in particle data arrays')

# Generate particle trees
gtree = spatial.cKDTree(gas_pos, boxsize=Lbox)
print('Gas tree constructed')
stree = spatial.cKDTree(star_pos, boxsize=Lbox)
print('Star tree constructed')
dmtree = spatial.cKDTree(dm_pos, boxsize=Lbox)
print('DM tree constructed')

print('There are %s galaxies, of which %s are central'%(Ngal, Ngal_c))
for i in range(len(sim.galaxies)):
    if not sim.galaxies[i].central:
        continue
    else:
        sys.stdout.flush()
        print('\rWorking on galaxy %s of %s'%(i+1, Ngal), end='')
        sys.stdout.flush()
        # Get positions of galaxies
        gal_pos = sim.galaxies[i].halo.minpotpos[:].to('kpccm').value
        
        # Get gas particles around the halo
        glist = gtree.query_ball_point(gal_pos, (10**logr_max)*gal_rvir[i])
        gpos_in = gas_pos[glist]
        gas_r = np.sqrt((gpos_in[:,0]-gal_pos[0])**2+(gpos_in[:,1]-gal_pos[1])**2+(gpos_in[:, 2]-gal_pos[2])**2) # ckpc
        gas_r = gas_r/gal_rvir[i]
        gas_logr = np.log10(gas_r)
        
        # Define phases
        ism_gas_mask = get_ism_mask(gas_temp[glist], gas_nh[glist], ism_density)
        cgm_gas_mask = np.invert(ism_gas_mask)
        wind_mask = gas_delaytime[glist] > 0.
        
        cool_gas_mask = cgm_gas_mask & np.invert(wind_mask) & (gas_temp[glist] < cold_temp)
        warm_gas_mask = cgm_gas_mask & np.invert(wind_mask) & (gas_temp[glist] > cold_temp) & (gas_temp[glist] < hot_temp)
        hot_gas_mask = cgm_gas_mask & np.invert(wind_mask) & (gas_temp[glist] > hot_temp)
        
        for j in np.arange(rbins-1):
            logr_mask = (gas_logr >= logr_min+j*dlogr) & (gas_logr < logr_min+(j+1)*dlogr)
            r_up = gal_rvir[i]*10**(logr_min + (j+1)*dlogr) # ckpc
            r_down = gal_rvir[i]*10**(logr_min + j*dlogr) # ckpc
            if (len(gas_mass[glist][cool_gas_mask & logr_mask])>0):
                density_profile['Cool CGM (T < 10^5)'][i, j] = np.sum(gas_mass[glist][cool_gas_mask & logr_mask])/(sphere(r_up)-sphere(r_down))
            if (len(gas_mass[glist][warm_gas_mask & logr_mask])>0):
                density_profile['Warm CGM (10^5 < T < 10^6)'][i, j] = np.sum(gas_mass[glist][warm_gas_mask & logr_mask])/(sphere(r_up)-sphere(r_down))
            if(len(gas_mass[glist][hot_gas_mask & logr_mask])>0):
                density_profile['Hot CGM (T > 10^6)'][i, j] = np.sum(gas_mass[glist][hot_gas_mask & logr_mask])/(sphere(r_up)-sphere(r_down))

        cool_gas_mask = cgm_gas_mask & np.invert(wind_mask) & (gas_temp[glist] < photo_temp)
        warm_gas_mask = cgm_gas_mask & np.invert(wind_mask) & (gas_temp[glist] > photo_temp) & (gas_temp[glist] < 0.5*gal_tvir[i])
        hot_gas_mask = cgm_gas_mask & np.invert(wind_mask) & (gas_temp[glist] > 0.5*gal_tvir[i])

        for j in np.arange(rbins-1):
            logr_mask = (gas_logr >= logr_min+j*dlogr) & (gas_logr < logr_min+(j+1)*dlogr)
            r_up = gal_rvir[i]*10**(logr_min + (j+1)*dlogr) # ckpc
            r_down = gal_rvir[i]*10**(logr_min + j*dlogr) # ckpc
            if (len(gas_mass[glist][logr_mask])>0):
                density_profile['Total Gas'][i, j] = np.sum(gas_mass[glist][logr_mask])/(sphere(r_up)-sphere(r_down))
            if (len(gas_mass[glist][cool_gas_mask & logr_mask])>0):
                density_profile['Cool CGM (T < Tphoto)'][i, j] = np.sum(gas_mass[glist][cool_gas_mask & logr_mask])/(sphere(r_up)-sphere(r_down))
            if (len(gas_mass[glist][warm_gas_mask & logr_mask])>0):
                density_profile['Warm CGM (Tphoto < T < 0.5Tvir)'][i, j] = np.sum(gas_mass[glist][warm_gas_mask & logr_mask])/(sphere(r_up)-sphere(r_down))
            if (len(gas_mass[glist][hot_gas_mask & logr_mask])>0):
                density_profile['Hot CGM (T > 0.5Tvir)'][i, j] = np.sum(gas_mass[glist][hot_gas_mask & logr_mask])/(sphere(r_up)-sphere(r_down))
            if (len(gas_mass[glist][(np.invert(cgm_gas_mask) & np.invert(wind_mask)) & logr_mask])>0):
                density_profile['ISM'][i, j] = np.sum(gas_mass[glist][(np.invert(cgm_gas_mask) & np.invert(wind_mask)) & logr_mask])/(sphere(r_up)-sphere(r_down))
            if (len(gas_mass[glist][wind_mask & logr_mask])>0):
                density_profile['Wind'][i, j] = np.sum(gas_mass[glist][wind_mask & logr_mask])/(sphere(r_up)-sphere(r_down))

        # Get gas particles around the halo
        slist = stree.query_ball_point(gal_pos, (10**logr_max)*gal_rvir[i])
        spos_in = star_pos[slist]
        star_r = np.sqrt((spos_in[:,0]-gal_pos[0])**2+(spos_in[:,1]-gal_pos[1])**2+(spos_in[:, 2]-gal_pos[2])**2) # ckpc
        star_r = star_r/gal_rvir[i]
        star_logr = np.log10(star_r)

        for j in np.arange(rbins-1):
            logr_mask = (star_logr >= logr_min+j*dlogr) & (star_logr < logr_min+(j+1)*dlogr)
            r_up = gal_rvir[i]*10**(logr_min + (j+1)*dlogr) # ckpc
            r_down = gal_rvir[i]*10**(logr_min + j*dlogr) # ckpc
            if (len(star_mass[slist][logr_mask])>0):
                density_profile['Stars'][i, j] = np.sum(star_mass[slist][logr_mask])/(sphere(r_up)-sphere(r_down))

        # Get gas particles around the halo
        dmlist = dmtree.query_ball_point(gal_pos, (10**logr_max)*gal_rvir[i])
        dmpos_in = dm_pos[dmlist]
        dm_r = np.sqrt((dmpos_in[:,0]-gal_pos[0])**2+(dmpos_in[:,1]-gal_pos[1])**2+(dmpos_in[:, 2]-gal_pos[2])**2) # ckpc
        dm_r = dm_r/gal_rvir[i]
        dm_logr = np.log10(dm_r)

        for j in np.arange(rbins-1):
            logr_mask = (dm_logr >= logr_min+j*dlogr) & (dm_logr < logr_min+(j+1)*dlogr)
            r_up = gal_rvir[i]*10**(logr_min + (j+1)*dlogr) # ckpc
            r_down = gal_rvir[i]*10**(logr_min + j*dlogr) # ckpc
            if (len(dm_mass[dmlist][logr_mask])>0):
                density_profile['Dark matter'][i, j] = np.sum(dm_mass[dmlist][logr_mask])/(sphere(r_up)-sphere(r_down))


density_profile = {k: p[central] for k, p in density_profile.items()}

with h5py.File(savedir+'density_profiles_%s.h5'%(snap), 'w') as hf:
    for k, p in density_profile.items():
        hf.create_dataset(k, data=np.array(p))

