
# Loops over all snapshots (in reverse) from last_snap to first_snap, setting up snapshot pairs [last_snap,snapfile2] to find progenitors.
# For each galaxy in last_snap, finds progenitors in snapfile2.  progenitors are defined as having the most star particles in common.
# Writes out an ascii file giving the top two most massive progenitors at all previous snapshots.  Works with multi-processor via OpenMP.
# Last modified: Romeel Dave 19 Feb 2019

import caesar
from pygadgetreader import *
import numpy as np
from scipy import stats
import time
import sys
import os
from progen import *

MODEL = sys.argv[1]
WIND = sys.argv[2]
last_snap = int(sys.argv[3])
if len(sys.argv)==5: nproc = int(sys.argv[4])
else: nproc = 1

BASEDIR = '/home/rad/data/%s/%s'%(MODEL,WIND)
GDIR = 'Groups'
parttype = 'star'  # 'gas', 'star, 'dm', or 'bndry' (for BHs)
first_snap = 26

#=========================================================
# MAIN DRIVER ROUTINE
#=========================================================

if __name__ == '__main__':

    # Set up pairs;
    if not os.path.isfile('%s/snap_%s_%03d.hdf5' % (BASEDIR,MODEL,last_snap)):
        sys.exit('Reference_snap %d does not exist'%last_snap)
    allsnaps = np.arange(last_snap,first_snap-1,-1)
    # find snapshots with caesar files that exist in the directory, in reverse order
    snapnums = []
    for i in allsnaps:
        if os.path.isfile('%s/snap_%s_%03d.hdf5' % (BASEDIR,MODEL,i)) and os.path.isfile('%s/%s/%s_%03d.hdf5' % (BASEDIR,GDIR,MODEL,i)):
            snapnums.append(i)
    #print('Found these snapshots with caesar files: %s'%snapnums)
    # set up pairs to progen
    pairs = []
    prevsnap = snapnums[0]  # find progenitors based on a single reference (final) snapshot
    for i in range(len(snapnums)-1):
        pairs.append([prevsnap,snapnums[i+1]])

    print('progen : Doing snapshot pairs: %s'%pairs)

# Set up multiprocessing; note: this doesn't help much, because this only parallelizes over galaxy ID searches, not the sorting.
    if nproc != 1:   # if multi-core, set up Parallel processing
        import multiprocessing
        from joblib import Parallel, delayed
        from functools import partial
        num_cores = multiprocessing.cpu_count()
        if nproc <= 0: print('progen : Using %d cores (all but %d)'%(num_cores+nproc+1,-nproc-1) )
        if nproc > 1: print('progen : Using %d of %d cores'%(nproc,num_cores))
        else: print('progen : Using single core')

# loop over pairs to find progenitors
    output_file = '%s/%s/progen_%s_%03d.dat' % (BASEDIR,GDIR,MODEL,last_snap)
    if parttype == 'dm': output_file = '%s/%s/progendm_%s_%03d.dat' % (BASEDIR,GDIR,MODEL,last_snap)
    if parttype == 'bndry': output_file = '%s/%s/progenbh_%s_%03d.dat' % (BASEDIR,GDIR,MODEL,last_snap)
    if parttype == 'gas': output_file = '%s/%s/progengas_%s_%03d.dat' % (BASEDIR,GDIR,MODEL,last_snap)
    t0 = time.time()
    snapfile1 = '%s/snap_%s_%03d.hdf5' % (BASEDIR,MODEL,last_snap)   # current snapshot; never changes
    caesarfile1 = '%s/%s/%s_%03d.hdf5' % (BASEDIR,GDIR,MODEL,last_snap)
    obj1 = caesar.load(caesarfile1,LoadHalo=False)
    if nproc == 1:
        all_index = []
        all_index2 = []
        for pair in pairs:
            print('progen : Doing pair %s [t=%g s]'%(pair,np.round(time.time()-t0,3)))
            snapfile2 = '%s/snap_%s_%03d.hdf5' % (BASEDIR,MODEL,pair[1])   # progenitor snapshot
            caesarfile2 = '%s/%s/%s_%03d.hdf5' % (BASEDIR,GDIR,MODEL,pair[1])
            obj2 = caesar.load(caesarfile2,LoadHalo=False)
        
            prog_index,prog_index2 = find_progens(snapfile1,snapfile2,obj1,obj2,nproc,t0,objtype='galaxies',parttype=parttype)  # find galaxies with most stars in common in prog snapshot
            all_index.append(prog_index)
            all_index2.append(prog_index2)

    if nproc > 1:
        snaps = []
        objs = []
        for pair in pairs:
            print('progen : Doing pair %s [t=%g s]'%(pair,np.round(time.time()-t0,3)))
            snaps.append('%s/snap_%s_%03d.hdf5' % (BASEDIR,MODEL,pair[1]))
            caesarfile2 = '%s/%s/%s_%03d.hdf5' % (BASEDIR,GDIR,MODEL,pair[1])
            objs.append(caesar.load(caesarfile2,LoadHalo=False))
        all_index,all_index2 = Parallel(n_jobs=nproc)(delayed(find_progens)(snapfile1,snaps[i],obj1,objs[i],1,t0,objtype='galaxies',parttype=parttype) for i in range(len(pairs)))  # find galaxies with most stars in common in prog snapshot

    all_index = np.asarray(all_index)
    all_index2 = np.asarray(all_index2)
    np.set_printoptions(threshold=np.inf)
    outfile = open(output_file,'w')
    print('%d %s'%(obj1.ngalaxies,pairs),file=outfile)
    print('%s'%(all_index.T),file=outfile)
    print('%s'%(all_index2.T),file=outfile)
    outfile.close()
    print('progen : Wrote info to file %s [t=%g s]'%(output_file,time.time()-t0))


