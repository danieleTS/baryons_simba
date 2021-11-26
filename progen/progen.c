/* progen -- Finds largest progenitors of galaxies in binfile within
	     oldbinfile.
  progen binfile grpfile idfile oldbinfile oldgrpfile oldidfile statfile oldstatfile munit mlim(Mo)

*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <malloc.h>
#include "tipsydefs_n2.h"

typedef struct idstruct {
  int id, index;
} idstruct;
idstruct *idold;

int CmpId(const void *p1,const void *p2)
{
        struct idstruct *a = (struct idstruct *)p1;
        struct idstruct *b = (struct idstruct *)p2;
        if(a->id > b->id) return 1;
        if(a->id < b->id) return -1;
        return 0;
}

int main(int argc,char **argv)
{
	int i,j,np,groupnum,imax,nmax,jlo,jhi;
	int ngroups,ngrpold,ngindex,gindex;
	int *idall;
	int *grpid,*grpold,*grpindex;
	int *proggrp;
	float *ms,*msold;
	float munit,mlim,junk;
	char line[256];
	struct dump oldhead;
	FILE *binfile,*grpfile,*idfile,*statfile,*oldbinf,*oldgrpf,*oldidf,*oldstatf,*tmpfile;

    if (argc != 11 ) {
	fprintf(stderr,"Usage: progen binfile grpfile idfile oldbinfile oldgrpfile oldidfile statfile oldstatfile munit mlim\n") ;
	return -1;
    }

	if( (binfile = fopen(argv[1],"r")) == NULL ) {
		fprintf(stderr,"Could not open snapshot file %s\n",argv[1]);
		return -1;
    	}
	if( (grpfile = fopen(argv[2],"r")) == NULL ) {
		fprintf(stderr,"Could not open grp/tag file %s\n",argv[2]);
		return -1;
    	}
	if( (idfile = fopen(argv[3],"r")) == NULL ) {
		fprintf(stderr,"Could not open idnum file %s\n",argv[3]);
		return -1;
    	}
	if( (oldbinf = fopen(argv[4],"r")) == NULL ) {
		fprintf(stderr,"Could not open tipsy binary file %s\n",argv[4]);
		return -1;
    	}
	if( (oldgrpf = fopen(argv[5],"r")) == NULL ) {
		fprintf(stderr,"Could not open grp/tag file %s\n",argv[5]);
		return -1;
    	}
	if( (oldidf = fopen(argv[6],"r")) == NULL ) {
		fprintf(stderr,"Could not open idnum file %s\n",argv[6]);
		return -1;
    	}
	if( (statfile = fopen(argv[7],"r")) == NULL ) {
		fprintf(stderr,"Could not open stat file %s\n",argv[7]);
		return -1;
    	}
	if( (oldstatf = fopen(argv[8],"r")) == NULL ) {
		fprintf(stderr,"Could not open old stat file %s\n",argv[8]);
		return -1;
    	}
	munit = atof(argv[9]);
	mlim = atof(argv[10]);	// in Mo


/* Read header to get number of stars */
	fprintf(stderr,"Loading HDF5 file %s\n",argv[1]);
	hid_t hdf5_file, hdf5_headergrp, hdf5_attribute, hdf5_grp, hdf5_dataset;
	herr_t status;

	hdf5_file      = H5Fopen(binfile, H5F_ACC_RDONLY, H5P_DEFAULT);
	hdf5_headergrp = H5Gopen1(hdf5_file, "/Header");
	hdf5_attribute = H5Aopen_name(hdf5_headergrp,"NumPart_Total");
	H5Aread(hdf5_attribute, H5T_NATIVE_INT, h.npartTotal);
	H5Aclose(hdf5_attribute);
/* Read in IDs */
#ifdef DEBUG
	printf("reading IDs for %d particles...\n",h.npartTotal[4]);
#endif
	idall = (int *) malloc(header.npartTotal[4]*sizeof(int));
	hdf5_grp  = H5Gopen1(hdf5_file, "/PartType4");
	hdf5_dataset = H5Dopen1(hdf5_grp, "ID");
	H5Dread(hdf5_dataset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, idall);
	H5Dclose(hdf5_dataset);
	for(i=0; i<h.npartTotal[4]; i++) fprintf(stderr,"%d ",idall[i]);
	H5Fclose(hdf5_grp);
	H5Fclose(hdf5_file);

/* Read progenitor file header */
	fprintf(stderr,"Loading HDF5 file %s\n",argv[1]);

	hdf5_file      = H5Fopen(binfile, H5F_ACC_RDONLY, H5P_DEFAULT);
	hdf5_headergrp = H5Gopen1(hdf5_file, "/Header");
	hdf5_attribute = H5Aopen_name(hdf5_headergrp,"NumPart_Total");
	H5Aread(hdf5_attribute, H5T_NATIVE_INT, h.npartTotal);
	H5Aclose(hdf5_attribute);

	idall = (int *) malloc(header.npartTotal[4]*sizeof(int));
	hdf5_grp  = H5Gopen1(hdf5_file, "/PartType4");
	hdf5_dataset = H5Dopen1(hdf5_grp, "ID");
	H5Dread(hdf5_dataset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, idall);
	H5Dclose(hdf5_dataset);
	for(i=0; i<h.npartTotal[4]; i++) idold[i].index = i;
	H5Fclose(hdf5_grp);
	H5Fclose(hdf5_file);

/* Read in group id numbers of star particles */
	grpid = (int *) malloc(header.nstar*sizeof(int));
	fscanf(grpfile,"%d",&i);
	if( header.nbodies != i ) {
		fprintf(stderr,"Something wrong in grp file: %d != %d\n",header.nbodies,i);
		return -1;
	}
	for( i=0; i<header.nsph+header.ndark; i++ ) fscanf(grpfile,"%d",&j);
	for( ngroups=0, i=0; i<header.nstar; i++ ) {
		fscanf(grpfile,"%d",&grpid[i]);
		if( grpid[i] > ngroups ) ngroups = grpid[i];
	}
	fclose(grpfile);

/* Read in group id numbers of star particles of progenitor file */
	grpold = (int *) malloc(oldhead.nbodies*sizeof(int));
	fscanf(oldgrpf,"%d",&i);
	if( oldhead.nbodies != i ) {
		fprintf(stderr,"Something wrong in oldgrp file: %d != %d\n",oldhead.nbodies,i);
		return -1;
	}
	for( ngrpold=0, i=0; i<oldhead.nbodies; i++ ) {
		fscanf(oldgrpf,"%d",&grpold[i]);
		if( grpold[i] > ngrpold ) ngrpold = grpold[i];
	}
	fclose(oldgrpf);
	proggrp = (int *) malloc((ngrpold+1)*sizeof(int));

/* Read in stellar masses of galaxies from stat files */
	ms = (float *)malloc((ngroups+1)*sizeof(float));
	msold = (float *)malloc((ngrpold+1)*sizeof(float));
	for( i=1; i<=ngroups; i++ ) {
		fgets(line,256,statfile);
		sscanf(line,"%d %d %g %g %g",&j,&j,&junk,&junk,&ms[i]);
		ms[i] *= munit;
	}
	for( i=1; i<=ngrpold; i++ ) {
		fgets(line,256,oldstatf);
		sscanf(line,"%d %d %g %g %g",&j,&j,&junk,&junk,&msold[i]);
		msold[i] *= munit;
	}
	ms[0] = msold[0] = 0.;
	//for( i=1; i<=5; i++ ) fprintf(stderr,"mass %d: %g %g\n",i,ms[i],msold[i]);
	fclose(statfile);
	fclose(oldstatf);

/* Read in/open tmp file for index list */
	grpindex = (int *) malloc((ngroups+1)*sizeof(int));
	if( (tmpfile = fopen("progen.tmp","r")) == NULL ) {
		ngindex = ngroups;
		for( i=0; i<=ngindex; i++ ) grpindex[i] = i;
	}
	else {
		i = 1;
		while( fgets(line,256,tmpfile) != NULL ) {
			sscanf(line,"%d",&grpindex[i]);
			i++;
		}
		ngindex = i-1;
		grpindex[0] = 0;
		fclose(tmpfile);
	}
	tmpfile = fopen("progen.tmp","w");
	fprintf(stderr,"%d stars in %d groups, searching thru %d old particles in %d groups (%d used)\n",header.nstar,ngroups,oldhead.nbodies,ngrpold,ngindex);
	
/* Sort id list */
	 qsort(idold,oldhead.nbodies,sizeof(idstruct),CmpId);

/* Begin loop over groups */
    for( gindex = 1; gindex <= ngindex; gindex ++ ) {
	groupnum = grpindex[gindex];
	if( ms[groupnum] < mlim ) continue;
	for( j=0; j<=ngrpold; j++ ) proggrp[j] = 0;  // init progen counter
	for( i=0, np=0; i<header.nstar; i++ ) {
	  if( grpid[i] == groupnum ) {	// for each group member...
	    jlo=0; jhi=oldhead.nbodies;
	    while( jhi-jlo > 1 ) {
	      j = (jlo+jhi)/2;
	      if( idall[i] <= idold[j].id ) jhi = j;
	      else jlo = j;
	    }
	    j = jhi;
	    //fprintf(stderr,"%d %d %d %d %d %d %d\n",jlo,jhi,j,idall[i],idold[jlo].id,idold[jhi].id,idold[jhi+1].id);
	    //for( j=0; j<oldhead.nbodies; j++ ) {
	    while( idall[i] == idold[j].id ) {
	      //fprintf(stderr,"%d %d %d\n",idall[i],j,grpold[idold[j].index]);
	      proggrp[grpold[idold[j].index]] ++; // bump
	      j++;
	    }
	    np ++;
	  }
	}
	for( j=1, imax=0, nmax=0; j<=ngrpold; j++ ) {	// find max count
	  if( proggrp[j] > nmax ) {
	    nmax = proggrp[j];
	    imax = j;
	  }
	}
	fprintf(stdout,"%d %g %d %g\n",groupnum,ms[groupnum],imax,msold[imax]);
	fprintf(tmpfile,"%d\n",imax);
	//if( ms[groupnum] > 1.e11 ) fprintf(stderr,"progen of %d is %d (%d of %d particles): %g %g\n",groupnum,imax,nmax,np,ms[groupnum],msold[imax]);
    }
    fclose(tmpfile);

    return 0;
}


