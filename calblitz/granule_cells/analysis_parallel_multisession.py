# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 18:02:10 2016

@author: agiovann
"""

#analysis parallel
%load_ext autoreload
%autoreload 2
from time import time
from scipy.sparse import coo_matrix
import tifffile
import subprocess
import time as tm
from time import time
import pylab as pl
import psutil
import glob
import os
import scipy
from ipyparallel import Client
import ca_source_extraction as cse
import calblitz as cb
import sys
import numpy as np
#%%
backend='SLURM'
if backend == 'SLURM':
    n_processes = np.int(os.environ.get('SLURM_NPROCS'))
else:
    n_processes = np.maximum(np.int(psutil.cpu_count()),1) # roughly number of cores on your machine minus 1
print 'using ' + str(n_processes) + ' processes'

#%% start cluster for efficient computation
single_thread=False

if single_thread:
    dview=None
else:    
    try:
        c.close()
    except:
        print 'C was not existing, creating one'
    print "Stopping  cluster to avoid unnencessary use of memory...."
    sys.stdout.flush()  
    if backend == 'SLURM':
        try:
            cse.utilities.stop_server(is_slurm=True)
        except:
            print 'Nothing to stop'
        slurm_script='/mnt/xfs1/home/agiovann/SOFTWARE/Constrained_NMF/SLURM/slurmStart.sh'
        cse.utilities.start_server(slurm_script=slurm_script)
        pdir, profile = os.environ['IPPPDIR'], os.environ['IPPPROFILE']
        c = Client(ipython_dir=pdir, profile=profile)        
    else:
        cse.utilities.stop_server()
        cse.utilities.start_server()        
        c=Client()

    
    dview=c[::4]
    print 'Using '+ str(len(dview)) + ' processes'
    
#%%
base_folders=[
              '/mnt/ceph/users/agiovann/ImagingData/eyeblink/b37/20160627154015/',            
              '/mnt/ceph/users/agiovann/ImagingData/eyeblink/b37/20160624105838/',
              '/mnt/ceph/users/agiovann/ImagingData/eyeblink/b37/20160625132042/',
              '/mnt/ceph/users/agiovann/ImagingData/eyeblink/b37/20160626175708/',
              '/mnt/ceph/users/agiovann/ImagingData/eyeblink/b37/20160627110747/',
              '/mnt/ceph/users/agiovann/ImagingData/eyeblink/b37/20160628100247/',
              '/mnt/ceph/users/agiovann/ImagingData/eyeblink/b37/20160705103903/',

              '/mnt/ceph/users/agiovann/ImagingData/eyeblink/b37/20160628162522/',
              '/mnt/ceph/users/agiovann/ImagingData/eyeblink/b37/20160629123648/',
              '/mnt/ceph/users/agiovann/ImagingData/eyeblink/b37/20160630120544/',
              '/mnt/ceph/users/agiovann/ImagingData/eyeblink/b37/20160701113525/',
              '/mnt/ceph/users/agiovann/ImagingData/eyeblink/b37/20160702152950/',
              '/mnt/ceph/users/agiovann/ImagingData/eyeblink/b37/20160703173620/',
              '/mnt/ceph/users/agiovann/ImagingData/eyeblink/b37/20160704130454/',
#              ]
#error:               '/mnt/ceph/users/agiovann/ImagingData/eyeblink/b35/20160711104450/', 
#              '/mnt/ceph/users/agiovann/ImagingData/eyeblink/b35/20160712105933/',             
#base_folders=[
              '/mnt/ceph/users/agiovann/ImagingData/eyeblink/b35/20160710134627/',
              '/mnt/ceph/users/agiovann/ImagingData/eyeblink/b35/20160710193544/',
              '/mnt/ceph/users/agiovann/ImagingData/eyeblink/b35/20160711164154/',
              '/mnt/ceph/users/agiovann/ImagingData/eyeblink/b35/20160711212316/',
              '/mnt/ceph/users/agiovann/ImagingData/eyeblink/b35/20160712101950/',
              '/mnt/ceph/users/agiovann/ImagingData/eyeblink/b35/20160712173043/',
              '/mnt/ceph/users/agiovann/ImagingData/eyeblink/b35/20160713100916/',
              '/mnt/ceph/users/agiovann/ImagingData/eyeblink/b35/20160713171246/',
              '/mnt/ceph/users/agiovann/ImagingData/eyeblink/b35/20160714094320/',
              '/mnt/ceph/users/agiovann/ImagingData/eyeblink/b35/20160714143248/'
              ] 
#%%
results=dview.map_sync(cb.granule_cells.utils_granule.fast_process_day,base_folders)              