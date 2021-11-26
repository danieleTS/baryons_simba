
import pylab as plt
import numpy as np
from scipy import stats
import sys
import os

igal = sys.argv[1:]

model = 'm50n512'
wind = 'fh_qr'
zorig = 0

colors = ('crimson', 'c', 'navy', 'g', 'm', 'y', 'k')

for k in range(0,len(igal)):
  j = 0
  mhist = []
  zhist = [zorig]
  ig = int(igal[k])
  for z in np.arange(0,6,0.25):
    infile = 'progen.%s.z%g-z%g.out' % (model,zorig,z)
    if os.path.isfile(infile):
        origgrp,proggrp,origm,progm = np.loadtxt(infile,usecols=(0,1,2,3),unpack=True)
	if j == 0:
	    mhist.append(origm[origgrp==ig][0])
	mhist.append(progm[origgrp==ig][0])
	zhist.append(z)
        j = j+1
  print zhist,mhist
  plt.plot(zhist,mhist,color=colors[k],label='%d'%(ig))

plt.rc('text', usetex=True)
plt.ylabel(r'$\log M_*$',fontsize=16)
plt.xlabel(r'$z$' ,fontsize=16)
plt.legend(loc='lower left',fontsize=16)
plt.show()

