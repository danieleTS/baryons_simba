import pylab as plt
import numpy as np
import os,sys

wind = 'fh_qr'
model = 'm50n512'

zref = float(sys.argv[1])	# reference redshift at which to start tracking back
igal = sys.argv[2:]		# galaxies to plot
for i in range(0,len(igal)):
    igal[i] = int(igal[i])
igal = np.asarray(igal)

BANDNUM = [2,4]
if len(BANDNUM) != 2:
    print 'You must provide 2 bands'
    exit
for i in range(0,len(BANDNUM)):
    BANDNUM[i] = int(BANDNUM[i])
BANDNUM = np.asarray(BANDNUM)


mlim = 1.16e9

colors = ('r', 'g', 'b', 'c', 'm', 'y', 'k')
plt.rc('text', usetex=True)

def cmdData(): 
    ax = plt.axis()
    mag = np.arange(ax[0],ax[1],0.1)
    redblue = -0.0585*(mag+16)+0.78
    plt.plot(mag,redblue,'--',lw=2,c='k')

def readheader(infile,index):
    f = open(infile)
    line = f.readline()
    line = line.split()
    redshift =  float(line[2])
    h        =  float(line[12]) / 100.
    boxsize  =  float(line[10])         # already has 1/h
    line = f.readline()
    line = f.readline()

    ncolors = 0
    for i in range(0,200):
        line  = f.readline()
        sline = line.split()

        if sline[0] != '#':
            break

        if int(sline[2]) == BANDNUM[j]:
            print 'plotting: ',line,

        if sline[1] == 'Color':
            ncolors += 1
    f.close()
    #print 'found %d colors' % ncolors
    return ncolors,boxsize,h,redshift


if __name__ == '__main__':

    Nproj = 1
    progid = []
    m1 = []
    m2 = []
    for redshift in np.arange(0,6,0.25):
	INFILE = '/home/rad/closer/gal_z%.3f.'%(redshift)+model+'.'+wind+'.abs'
	PROGENFILE = '/home/rad/gizmo-analysis/romeeld/progen/progen.%s.z%g-z%g.out' % (model,zref,redshift)
	if os.path.isfile(INFILE)  == 0:
	    continue
        for j in range(0,len(BANDNUM)):
	    ncolors,boxsize,h,redz = readheader(INFILE,j)
	    galid,sfr,ms,Lfir,mag,magfree = np.loadtxt(INFILE,usecols=(0,1,4,5,BANDNUM[j]+10,BANDNUM[j]+11+ncolors),unpack=True)
	    if j==0:
		mag1 = np.asarray(mag[ms>mlim])
	    if j==1:
		mag2 = np.asarray(mag[ms>mlim])
	    Lfir = np.asarray(Lfir[ms>mlim])
	    sfr = np.asarray(sfr[ms>mlim])
	    galid = np.asarray(galid[ms>mlim])
	    ms = np.asarray(ms[ms>mlim])

	if np.abs(redshift-zref) < 0.01:	# plot full CMD at z=zref
	    for j in range(0,len(igal)):
	        progid.append(igal[j])
		for k in range(0,len(galid)):
		    if int(galid[k]) == igal[j]:
	        	m1.append(mag1[k])
	        	m2.append(mag2[k])
			break
	    print progid,m1,m2
            ssfr = 1.e9*sfr/ms
            ssfr = np.log10(ssfr+10**(-2.5+0.5*redshift))
            pixcolor = ssfr
            pixsize = 5*(np.log10(ms/min(ms))+1)
            fig,ax = plt.subplots()
            im = ax.scatter(mag2,mag1-mag2, c=pixcolor, s=pixsize, lw=0, cmap=plt.cm.jet_r)
            im.set_clim(-3+0.5*redshift,0)  
            #plt.plot(mag2,mag1-mag2,'o',ms=3,color=pixcolor,label='Color')
            #plt.hexbin(mag2,mag1-mag2,C=sfr,gridsize=50)
            fig.colorbar(im,ax=ax,label=r'sSFR$_{100}$ (Gyr$^{-1}$)')
            #cb.set_label('Log FIR')

	if os.path.isfile(PROGENFILE)  == 0:
	    continue
	print INFILE,PROGENFILE

	if np.abs(redshift-zref) > 0.01:	# read in progenitor file and compile magnitudes of progenitors
	    origid,progenid = np.loadtxt(PROGENFILE,usecols=(0,1),unpack=True)
	    for j in range(0,len(igal)):
		pid = -1
		for k in range(0,len(origid)):
		    if int(origid[k]) == igal[j]:
			pid = int(progenid[k])
			break
		if pid == -1:
		    print 'Cannot find progen id'
		    sys.exit()
		found = 0
		for k in range(0,len(galid)):
		    if int(galid[k]) == pid:
	        	progid.append(pid)
	        	m1.append(mag1[k])
	        	m2.append(mag2[k])
			Nproj = Nproj+1
			found = 1
			break
		if found == 0:
	            progid.append(-1)
	            m1.append(-100)
	            m2.append(-100)

    m1 = np.asarray(m1)
    m2 = np.asarray(m2)
    for j in range(0,len(igal)):
	mg1 = m1[j::len(igal)]	# every len(igal)'th element, starting with element j
	mg2 = m2[j::len(igal)]
	mg1 = mg1[mg1>-99]
	mg2 = mg2[mg2>-99]
	print igal[j],mg1,mg2
	plt.plot(mg2,mg1-mg2,'o',ms=10,color=colors[j])
	plt.plot(mg2,mg1-mg2,'-',lw=4,color=colors[j],label='Gal %d'%(igal[j]))

    cmdData()

    plt.xlim(-16-2*zref,-24.5-2*zref)
    plt.ylim(0,1.5)
    #plt.axis((ax[1],ax[0],ax[2],ax[3]))	# reverse x axis

    plt.xlabel(r'$i$',fontsize=20)
    plt.ylabel(r'$g-i$',fontsize=20)
    plt.legend(loc='lower left')

    figname = 'cmdtracks.pdf'
    plt.savefig(figname,bbox_inches='tight')
    plt.show()
