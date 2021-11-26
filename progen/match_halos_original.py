
# Matches halos across different wind models

import caesar
from pygadgetreader import *
import numpy as np
from scipy import stats
import time
import sys
import os
from progen import find_progens

MODEL = sys.argv[1]
WIND1 = sys.argv[2]
WIND2 = sys.argv[3]
SNAP = int(sys.argv[4])

GDIR = 'Groups'
BASEDIR1 = '/home/rad/data/%s/%s'%(MODEL,WIND1)
BASEDIR2 = '/home/rad/data/%s/%s'%(MODEL,WIND2)

#=========================================================
# MAIN DRIVER ROUTINE
#=========================================================

if __name__ == '__main__':

    snapfile1 = '%s/snap_%s_%03d.hdf5' % (BASEDIR1,MODEL,SNAP)   # current snapshot
    caesarfile1 = '%s/%s/%s_%03d.hdf5' % (BASEDIR1,GDIR,MODEL,SNAP)
    obj1 = caesar.load(caesarfile1)
    snapfile2 = '%s/snap_%s_%03d.hdf5' % (BASEDIR2,MODEL,SNAP)   # progenitor snapshot
    caesarfile2 = '%s/%s/%s_%03d.hdf5' % (BASEDIR2,GDIR,MODEL,SNAP)
    obj2 = caesar.load(caesarfile2)
    prog_index,prog_index2 = find_progens(snapfile1,snapfile2,obj1,obj2,1,0.,objtype='halos',parttype='dm')  # find galaxies with most particles in common in progenitor snapshot

    print(prog_index,prog_index2)

