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
from ipyparallel import Client
#%%
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
    cse.utilities.stop_server()
    cse.utilities.start_server()
    c=Client()
    dview=c[::2]

print 'using '+ str(len(dview))+ ' processors'    
#%%
import os
fnames=[]
for file in os.listdir("./"):
    if file.startswith("") and file.endswith(".tif"):
        fnames.append(file)
fnames.sort()
print fnames  
#%%
#low_SNR=False
#if low_SNR:
#    N=1000     
#    mn1=m.copy().bilateral_blur_2D(diameter=5,sigmaColor=10000,sigmaSpace=0)     
#    
#    mn1,shifts,xcorrs, template=mn1.motion_correct()
#    mn2=mn1.apply_shifts(shifts)     
#    #mn1=cb.movie(np.transpose(np.array(Y_n),[2,0,1]),fr=30)
#    mn=cb.concatenate([mn1,mn2],axis=1)
#    mn.play(gain=5.,magnification=4,backend='opencv',fr=30)
#%%
t1 = time()
file_res=cb.motion_correct_parallel(fnames,fr=30,template=None,margins_out=0,max_shift_w=45, max_shift_h=45,backend='ipyparallel',apply_smooth=True)
t2=time()-t1
print t2
#%%   
all_movs=[]
for f in  fnames:
    idx=f.find('.')
    with np.load(f[:idx+1]+'npz') as fl:
        print f
#        pl.subplot(1,2,1)
#        pl.imshow(fl['template'],cmap=pl.cm.gray)
#        pl.subplot(1,2,2)
        pl.plot(fl['shifts'])       
        all_movs.append(fl['template'][np.newaxis,:,:])
        pl.pause(.001)
#        pl.cla()
#%%
num_movies_per_chunk=20        
chunks=range(0,len(fnames),20)
chunks[-1]=len(fnames)
#%%
template_each=[];
all_movs_each=[];
movie_names=[]
for idx in range(len(chunks)-1):
        print chunks[idx], chunks[idx+1]
        all_mov=all_movs[chunks[idx]:chunks[idx+1]]
        all_mov=cb.movie(np.concatenate(all_mov,axis=0),fr=30)
        all_mov,shifts,_,_=all_mov.motion_correct(template=np.median(all_mov,axis=0))
        template=np.median(all_mov,axis=0)
        all_movs_each.append(all_mov)
        template_each.append(template)
        movie_names.append(fnames[chunks[idx]:chunks[idx+1]])
        pl.imshow(template,cmap=pl.cm.gray,vmax=100)
        
np.savez('template_total.npz',template_each=template_each, all_movs_each=all_movs_each,movie_names=movie_names)        
#%%
for mov in all_movs_each:
    mov.play(backend='opencv',gain=10.,fr=100)
#%%

#%%
file_res=[]
for template,fn in zip(template_each,movie_names):
    print fn
    file_res.append(cb.motion_correct_parallel(fn,30,dview=None,template=template,margins_out=0,max_shift_w=35, max_shift_h=35,remove_blanks=True))
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
fr_remove_init=30
for f in  fnames:
    with np.load(f[:-3]+'npz') as fl:
        big_shifts.append(fl['shifts'])
        
    print f
    Yr=cb.load(f[:-3]+'hdf5')[fr_remove_init:]
    Yr=Yr.resize(fx=1,fy=1,fz=.2)
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
np.save('Yr_DS.npy',big_mov)
np.save('big_shifts.npy',big_shifts)

#%%
_,d1,d2=np.shape(cb.load(fnames[0][:-3]+'hdf5',subindices=range(3),fr=10))
Yr=np.load('Yr_DS.npy',mmap_mode='r')  
d,T=Yr.shape      
Y=np.reshape(Yr,(d1,d2,T),order='F')
Y=cb.movie(np.array(np.transpose(Y,(2,0,1))),fr=30)
#%%
Y.play(backend='opencv',fr=300,gain=10,magnification=1)
