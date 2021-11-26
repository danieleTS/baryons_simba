import numpy as np
import h5py
import caesar
import sys
import os
from pygadgetreader import readsnap, readheader
from plotting_methods import *
from scipy import spatial

T1 = 1.e5 # K
T2=1.e6 # K
T3=1.e7 # K
G = 6.67e-11 # SI units
Mpc2km = 3.0856776e19
km2Mpc = 1./Mpc2km
Msun = 1.98847e30 # kg

model = sys.argv[1]
wind = sys.argv[2]

# Select snapshots
SNAP = np.linspace(35,151,(151-35)+1)
z = np.zeros(len(SNAP))

# Define phases
phases = ['<10^5 K', '10^5-10^6 K', '10^6-10^7 K', '>10^7 K']
omega_frac = {phase: np.zeros(len(SNAP)) for phase in phases}

# Open output file
fout = open('fIGM-T_%s%s.dat'%(model, wind),'w')
fout.write('z\t<10^5 K\t10^5-10^6 K\t10^6-10^7 K\t>10^7 K\n')

for iz in np.arange(len(SNAP)):
    # Read snapshot
    snap = SNAP[iz]
    datadir = '/disk04/rad/sim/'+model+'/'+wind+'/'
    snapfile = datadir + 'snap_'+model+'_%03d.hdf5'%(snap)
    # Load halo catalogue
    caesarfile = datadir + 'Groups/'+model+'_%03d.hdf5'%(snap)
    sim = caesar.quick_load(caesarfile)

    # Read cosmology
    h = sim.simulation.hubble_constant
    Lbox = readheader(snapfile,'boxsize')
    Lbox = Lbox/h # ckpc
    z_sim = sim.simulation.redshift
    z[iz] = z_sim
    Om0=sim.simulation.omega_matter
    Ob0=sim.simulation.omega_baryon
    f_baryon = Ob0/Om0
    OL0 = 1-Om0

    # Define threshold for the classification of IGM (see Christiansen et al. 2020)
    f_Om = Om0*(1+z_sim)**3 / (Om0*(1+z_sim)**3 + (1-Om0-OL0)*(1+z_sim)**2 + OL0)
    delta_th = 6*np.pi**2 * (1+0.4093*(1./f_Om-1)**0.9052)-1 # Dave et al. (2010)
    print('z=%s delta_th=%s'%(z_sim, delta_th))

    H0 = 100*sim.simulation.hubble_constant*km2Mpc
    Hz = H0 * (Om0*(1+z_sim)**3+OL0)**0.5
    rho_crit0 = 3*H0**2/(8*np.pi*G) # kg/m^3
    rho_bar = Ob0*rho_crit0*(1+z_sim)**3 # kg/m^3

    
    central = np.array([i.central for i in sim.galaxies])
    gal_rvir = np.array([i.halo.virial_quantities['r200c'].in_units('kpccm') for i in sim.galaxies])
    Ngal = len(central)
    Ngal_c = len(gal_rvir[central])

    gas_mass = readsnap(snapfile, 'mass', 'gas', suppress=1, units=1) / h # in Mo
    gas_temp = readsnap(snapfile, 'u', 'gas', suppress=1, units=1) # in K
    gas_pos = readsnap(snapfile, 'pos', 'gas', suppress=1, units=1) /h # in ckpc ( verified it)
    print('Finished loading in particle data arrays')

    # Generate gas elements tree
    gtree = spatial.cKDTree(gas_pos, boxsize=Lbox)
    print('Gas tree constructed')

    # List of gas particles in halos
    halo_g = np.array([], dtype=np.int)

    for i in range(len(sim.galaxies)):
        
        if not sim.galaxies[i].central:
            continue
        else:
            sys.stdout.flush()
            print('\rWorking on galaxy %s of %s'%(i+1, Ngal), end='')
            sys.stdout.flush()
            # Get positions of galaxies
            gal_pos = sim.galaxies[i].pos[:].to('kpccm').value
            R = gal_rvir[i] # ckpc
            # Select gas particles within distance R from central galaxy
            glist = gtree.query_ball_point(gal_pos, R)
            # Add particles to the list
            halo_g = np.concatenate((halo_g, glist))

    # Define IGM as gas particles outside halos
    igm_g = np.arange(len(gas_mass), dtype=np.int)
    igm_g = np.delete(igm_g, halo_g).astype('int')

    # Define IGM phases
    cold_mask = gas_temp[igm_g]<T1
    cool_mask = (gas_temp[igm_g]<T2) & (gas_temp[igm_g]>=T1)
    warm_mask = (gas_temp[igm_g]<T3) & (gas_temp[igm_g]>=T2)
    hot_mask = gas_temp[igm_g]>=T3

    baryon_mass = rho_bar*(Mpc2km*Lbox/(1+z_sim))**3/Msun # Msun
    # The (1+z) in the above expression serves to convert Lbox from co-moving to physical units
    omega_frac['<10^5 K'][iz] = np.sum(gas_mass[igm_g][cold_mask])/baryon_mass
    omega_frac['10^5-10^6 K'][iz] = np.sum(gas_mass[igm_g][cool_mask])/baryon_mass
    omega_frac['10^6-10^7 K'][iz] = np.sum(gas_mass[igm_g][warm_mask])/baryon_mass
    omega_frac['>10^7 K'][iz] = np.sum(gas_mass[igm_g][hot_mask])/baryon_mass
    
    fout.write('%s\t%s\t%s\t%s\t%s\n'%(z_sim, omega_frac['<10^5 K'][iz], omega_frac['10^5-10^6 K'][iz],
                                       omega_frac['10^6-10^7 K'][iz], omega_frac['>10^7 K'][iz]))
fout.close()
