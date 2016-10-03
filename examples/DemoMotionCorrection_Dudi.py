# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 20:56:07 2015
@author: agiovann
Updated on Tue Apr 26 12:13:18 2016
@author: agiovann
Updated on Wed Aug 17 13:51:41 2016
@author: deep-introspection
"""

# init
import calblitz as cb
import pylab as pl
import numpy as np

# set basic ipython functionalities
from IPython import get_ipython
ipython = get_ipython()

if '__IPYTHON__' in globals():
    ipython.magic('load_ext autoreload')
    ipython.magic('autoreload 2')
#%%
# define movie
filename = 'your_movie.tif'
filename_hdf5 = filename[:-4]+'.hdf5'
filename_mc = filename[:-4]+'_mc.npz'
frameRate = 30
start_time = 0

# load movie
# for loading only a portion of the movie or only some channels
# you can use the option: subindices=range(0,1500,10)
m = cb.load(filename, fr=frameRate, start_time=start_time)
print m.shape
# red and green channels
m_r=m[:,:,:,0] # red
m=m[:,:,:,1] # green

# this syntax depends on how you organized red and green channels it could also be
#m_r=m[1::2,:,:]
#m=m[0::2,:,:,:]
# or other. Look at m.shape


# backend='opencv' is much faster
cb.concatenate([m,m_r],axis=1).resize(.5,.5,.1).play(fr=100, gain=1.0, magnification=1,backend='opencv')


# automatic parameters motion correction
max_shift_h = 20  # maximum allowed shifts in y
max_shift_w = 20  # maximum allowed shifts in x
m_r_mc, shifts, xcorrs, template = m_r.motion_correct(max_shift_w=max_shift_w,
                                               max_shift_h=max_shift_h,
                                               num_frames_template=None,
                                               template=None, remove_blanks=False,
                                               method='opencv')
#%%
pl.figure()
pl.imshow(template,cmap='gray')
pl.figure()
pl.plot(shifts)

#%% apply the shifts to the green channel
m_mc=m.apply_shifts(shifts, interpolation='linear', method='opencv', remove_blanks=True)
#%%
m_mc.resize(.5,.5,.1).play(fr=30, gain=1.0, magnification=1)

