# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 20:56:07 2015
@author: agiovann
"""
#%%
import sys
sys.path
# add 
sys.path.append('/Users/agiovann/Documents/Software/CalBlitz')
from XMovie import XMovie
import time
from pylab import plt
plt.ion()
%load_ext autoreload
%autoreload 2
import numpy as np
#%% define movie
filename='k26_v1_176um_target_pursuit_001_005.tif'
filename_mc=filename[:-4]+'_mc.npy'
filename_analysis=filename[:-4]+'_analysis.npz'
#%%
#m=XMovie('M_FLUO.tif', frameRate=.064);
#m=XMovie('M_FLUO_1.tif', frameRate=.064);
m=XMovie(filename, frameRate=.033);
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
plt.imshow(templates[-1],cmap=plt.cm.Greys_r,vmin=-100,vmax=00)
#%% save motion corrected movie inpython format. This takes some time now but will save  a lot later...
np.save(filename_mc,m.mov)    
#%% crop movie after motion correction. 
m.crop(max_shift,max_shift,max_shift,max_shift)    
#%% resize to increase SNR and have better convergence of PCA-ICA
fx=.5;
fy=.5;
fz=.2;
m.resize(fx=fx,fy=fy,fz=fx)
#%% example plot movie
m.playMovie(frate=.03,gain=6.0,offset=100,magnification=1)
#%% example plot a frame
plt.imshow(m.mov[100],cmap=plt.cm.Greys_r)

#%% compute delta f over f DF/F
initTime=time.time()
m.computeDFF(secsWindow=5,quantilMin=20,subtract_minimum=True)
print 'elapsed time:' + str(time.time()-initTime) 
#%%
m.playMovie(frate=.1,gain=6.0,magnification=1)


#%% compute spatial components via ICA PCA
initTime=time.time()
spcomps=m.IPCA_stICA(components=50);
print 'elapsed time:' + str(time.time()-initTime) 
#%% extract ROIs from spatial components 
masks=m.extractROIsFromPCAICA(spcomps, numSTD=8, gaussiansigmax=2 , gaussiansigmay=2)
#%%  extract single ROIs from each mask
allMasks=[np.array(mm==ll) for mm in masks  for ll in xrange(np.max(mm)) if 2000>np.sum(np.array(mm==ll))>10   ]
allMasks=np.asarray(allMasks,dtype=np.float16)
print allMasks.shape
#%%
allMasksForPlot=[kk*ii*1.0 for ii,kk in enumerate(allMasks)]
plt.imshow(np.max(np.asarray(allMasksForPlot,dtype=np.float16),axis=0))
#%%
np.savez(filename_analysis,allMasks=allMasks,shifts=shifts,templates=templates,spcomps=spcomps,max_shift=max_shift,fx=fx,fy=fy,fz=fz)
#%% create masks:here you should divide each of the masks (neurons found in one component) according to your taste



#%% load and extract traces
initTime=time.time()
minPercentileRemove=1;
m=XMovie(mat=np.load(filename_mc), frameRate=.033);
m.mov=m.mov-np.percentile(m.mov,minPercentileRemove);
vars_=np.load(filename_analysis)
max_shift=vars_['max_shift']
allMasks=vars_['allMasks']
m.crop(max_shift,max_shift,max_shift,max_shift)
#%% reshape dendrites 
mdend=XMovie(mat=np.asarray(allMasks,dtype=np.float32), frameRate=1);
mdend.resize(fx=2,fy=2)
allMasks=mdend.mov;
#%% reshape movie and mask to conveniently compute DFF
T,h,w=m.mov.shape
Y=np.reshape(m.mov,(T,h*w))
print allMasks.shape
nA,_,_=allMasks.shape
A=np.reshape(allMasks,(nA,h*w))
pixelsA=np.sum(A,axis=1)
A=A/pixelsA[:,None] # obtain average over ROI
Ftraces=np.dot(A,np.transpose(Y))
print 'elapsed time:' + str(time.time()-initTime) 
#%% compute DFF
tracesDFF=[];
window=int(10/m.frameRate);
minQuantile=20;
for trace in Ftraces:
    #print trace.shape
    traceBL=[np.percentile(trace[i:i+window],minQuantile) for i in xrange(1,len(trace)-window)]
    missing=np.percentile(trace[-window:],minQuantile);
    missing=np.repeat(missing,window+1)
    traceBL=np.concatenate((traceBL,missing))
    tracesDFF.append((trace-traceBL)/traceBL)

tracesDFF=np.asarray(tracesDFF)
#%% visualize traces
plt.plot(tracesDFF[1:20].T)
#%% visualize traces imagesc style
plt.imshow(tracesDFF,aspect='auto',vmax=1.5) 




#%% example take part of the frames
partmov=m.makeSubMov(range(0,100,10))
partmov.playMovie(frate=.1,gain=1.0)