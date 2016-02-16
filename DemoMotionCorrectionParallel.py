# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 17:56:14 2016

@author: agiovann
"""

#%%
try:
    %load_ext autoreload
    %autoreload 2
    print 1
except:
    print 'NOT IPYTHON'

import matplotlib as mpl
mpl.use('TKAgg')
from matplotlib import pyplot as plt
#plt.ion()

import sys
import numpy as np
import ca_source_extraction as cse

#sys.path.append('../SPGL1_python_port')
#%
from time import time
from scipy.sparse import coo_matrix
import tifffile
import subprocess
import time as tm
from time import time
import pylab as pl
import psutil
import calblitz as cb

#%%
import os
fnames=[]
for file in os.listdir("./"):
    if file.endswith(".tif"):
        fnames.append(file)
fnames.sort()
print fnames  
#%%
n_processes = np.maximum(psutil.cpu_count() - 2,1) # roughly number of cores on your machine minus 1
#print 'using ' + str(n_processes) + ' processes'
p=2 # order of the AR model (in general 1 or 2)
print "Stopping  cluster to avoid unnencessary use of memory...."
sys.stdout.flush()  
cse.utilities.stop_server() 
cse.utilities.start_server(n_processes)
#%%
t1 = time()
file_res=cb.motion_correct_parallel(fnames,30,template=None,margins_out=10,max_shift_w=5, max_shift_h=5)
#%%   
all_movs=[]
for f in  file_res:
    with np.load(f+'npz') as fl:
        pl.subplot(1,2,1)
        pl.imshow(fl['template'],cmap=pl.cm.gray)
        pl.subplot(1,2,2)
        pl.plot(fl['shifts'])       
        all_movs.append(fl['template'][np.newaxis,:,:])
        pl.pause(1)
        pl.cla()
        
all_movs=cb.movie(np.concatenate(all_movs,axis=0),fr=10)
all_movs,shifts,_,_=all_movs.motion_correct(template=np.median(all_movs,axis=0))
template=np.median(all_movs,axis=0)
np.save('template_total',template)
pl.imshow(template,cmap=pl.cm.gray,vmax=100)
#%%
file_res=cb.motion_correct_parallel(fnames,30,template=template,margins_out=10,max_shift_w=10, max_shift_h=10,remove_blanks=False)
#%%
for f in  file_res:
    with np.load(f+'npz') as fl:
        pl.subplot(1,2,1)
        pl.imshow(fl['template'],cmap=pl.cm.gray)
        pl.subplot(1,2,2)
        pl.plot(fl['shifts'])       
        pl.pause(0.1)
        pl.cla()
        
print time() - t1 - 200
#%%
big_mov=[];
big_shifts=[]
for f in  fnames:
    with np.load(f[:-3]+'npz') as fl:
        big_shifts.append(fl['shifts'])
        
    print f
    Yr=cb.load(f[:-3]+'hdf5')
    Yr=Yr.resize(fx=1,fy=1,fz=.5)
    Yr = np.transpose(Yr,(1,2,0)) 
    d1,d2,T=Yr.shape
    Yr=np.reshape(Yr,(d1*d2,T),order='F')
    print Yr.shape
#    np.save(fname[:-3]+'npy',np.asarray(Yr))
    big_mov.append(np.asarray(Yr))
#%%
big_mov=np.concatenate(big_mov,axis=-1)
big_shifts=np.concatenate(big_shifts,axis=0)
#%%
np.save('Yr.npy',big_mov)
np.save('big_shifts.npy',big_shifts)

#%%
_,d1,d2=np.shape(cb.load(fnames[0][:-3]+'hdf5',subindices=range(3),fr=10))
Yr=np.load('Yr.npy',mmap_mode='r')  
d,T=Yr.shape      
Y=np.reshape(Yr,(d1,d2,T),order='F')