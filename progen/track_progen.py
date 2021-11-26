
import pylab as plt
import numpy as np
from scipy import stats
import sys
import os
import caesar

MODEL = sys.argv[1]
WIND = sys.argv[2]
BASEDIR = '/home/rad/data/%s/%s'%(MODEL,WIND)
igal = sys.argv[3:]

SNAPS = range(151,26,-1)

colors = ('crimson', 'c', 'navy', 'g', 'm', 'y', 'k')

ms = []
sfr = []
redshift = []
ig = igal[:]
for isnap in range(len(SNAPS)):
    infile = '%s/Groups/%s_%03d.hdf5' % (BASEDIR,MODEL,int(SNAPS[isnap]))
    if not os.path.isfile(infile):
        print 'infile ',infile,'not found, skipping',k
        continue
    sim = caesar.load(infile,LoadHalo=False)
    ms0 = np.asarray([i.masses['stellar'] for i in sim.galaxies if i.central <= 1])
    sfr0 = np.asarray([i.sfr for i in sim.galaxies if i.central <= 1])
    redshift.append(sim.simulation.redshift)
    ms1 = np.array([ms0[int(i)] for i in ig])
    sfr1 = np.array([sfr0[int(i)] for i in ig])
    #print ms1,sfr1
    ms.append(ms1)
    sfr.append(sfr1)
    prog = np.asarray([i.progen_index for i in sim.galaxies if i.central <= 1])
    ig = [prog[int(i)] for i in ig]

for i in range(len(igal)):
    plt.plot(np.log10(1.+redshift),np.log10(ms[i]),color=colors[i],label='%d'%(igal))

plt.rc('text', usetex=True)
plt.xlabel(r'$\log\ 1+z$' ,fontsize=16)
plt.ylabel(r'$\log\ M_*$',fontsize=16)
plt.legend(loc='lower left',fontsize=16)
plt.show()

