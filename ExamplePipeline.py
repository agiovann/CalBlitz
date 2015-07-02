# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 20:56:07 2015

@author: agiovann
"""
#%%
import sys
sys.path
# add 
sys.path.append('/Users/agiovann/Dropbox/Python/princeton-ecs/princeton-ecs')
from XMovie import XMovie
import time
plt.ion()
%load_ext autoreload
%autoreload 2
#%% define movie
#m=XMovie('M_FLUO.tif', frameRate=.064);
#m=XMovie('M_FLUO_1.tif', frameRate=.064);
m=XMovie('k23_20150424_002_001.tif', frameRate=.033);

#%% motion correct run x times
# WHEN YOU RUN motion_correct YOUR ARE MODIFYING THE OBJECT!!!!
templates=[];
shifts=[];
max_shift=5;
num_iter=3;
for j in range(0,num_iter):
    template,shift=m.motion_correct(max_shift=max_shift,template=None,show_movie=False);
    templates.append(template)
    shift=np.asarray(shift)
    shifts.append(shift)
    plt.plot(np.asarray(shifts).reshape((j+1)*shift.shape[0],shift.shape[1]))
#%% plot final template    
#plt.imshow(templates[-1],cmap=plt.cm.Greys_r,vmin=50,vmax=300)
plt.imshow(templates[-1],cmap=plt.cm.Greys_r,vmin=-200,vmax=500)

#%% example plot movie
m.playMovie(frate=.03,gain=6.0)
#%% example plot a frame
plt.imshow(m.mov[1000],cmap=plt.cm.Greys_r)
#%% crop movie after motion correction
m=m.crop(max_shift,max_shift,max_shift,max_shift)
#%% compute delta f over f DF/F
initTime=time.time()
m=m.computeDFF(secsWindow=5,quantilMin=8,subtract_minimum=True)
m=m[0]
print 'elapsed time:' + str(time.time()-initTime) 
#%%
m.playMovie(frate=.001,gain=10.0,magnification=1)
#%% compute spatial components via ICA PCA
initTime=time.time()
spcomps=m.IPCA_stICA(components=50);
print 'elapsed time:' + str(time.time()-initTime) 
#%%
masks=m.extractROIsFromPCAICA(spcomps, numSTD=5, gaussiansigmax=2 , gaussiansigmay=2)
#%%
plt.imshow(np.sum( np.asarray(masks)>0,axis=0))
#%% create masks:here you should divide each of the masks (neurons found in one component) according to your taste
allMasks=np.asarray(masks)>0;
nA,d,d=allMasks.shape
#%%  create fluorescence traces
T,h,w=m.mov.shape
Y=np.reshape(m.mov,(T,h*w))
A=np.reshape(allMasks,(nA,h*w))
Ftraces=np.dot(A,np.transpose(Y))
#%% plot one trace
plt.plot(Ftraces[0])
#%% plot all traces
plt.plot(Ftraces.transpose())

#%% example take part of the frames
partmov=m.makeSubMov(range(0,100,10))
partmov.playMovie(frate=.1,gain=1.0)