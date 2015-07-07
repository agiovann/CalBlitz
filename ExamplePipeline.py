# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 20:56:07 2015
@author: agiovann
"""
#%% add CalBlitz folder to python directory
path_to_CalBlitz_folder='/Users/agiovann/Documents/Software/CalBlitz'

import sys
sys.path
sys.path.append(path_to_CalBlitz_folder)
#% add required packages
from XMovie import XMovie
import time
from pylab import plt
import numpy as np
from utils import matrixMontage
#% set basic ipython functionalities
try: 
    plt.ion()
    %load_ext autoreload
    %autoreload 2
except:
    print "Probably not a Ipython interactive environment" 


#%% define movie

#filename='k26_v1_176um_target_pursuit_001_005.tif'
#frameRate=.033;

#filename='20150522_1_1_001.tif'
       
filename='M_FLUO.tif'
frameRate=.064;

#data_type='fly'
#filename='20150522_1_1_001.tif'
#frameRate=.128;

#%%
filename_mc=filename[:-4]+'_mc.npz'
filename_analysis=filename[:-4]+'_analysis.npz'    
filename_traces=filename[:-4]+'_traces.npz'    

#%%
m=XMovie(filename, frameRate=frameRate);


#%% example plot a frame
plt.imshow(m.mov[100],cmap=plt.cm.Greys_r)

#%% example play movie
m.playMovie(frate=.06,gain=6.0,magnification=3)

#%% example take first channel of two
channelId=1;
totalChannels=2;
m.makeSubMov(range(channelId-1,m.mov.shape[0],totalChannels))


#%% concatenate movies (it will add to the original movie)
m.append(new_mov)

#%% motion correct run 3 times
# WHEN YOU RUN motion_correct YOUR ARE MODIFYING THE OBJECT!!!!
templates=[];
shifts=[];

max_shift=5;
num_iter=3; # numer of times motion correction is executed
template=None # here you can use your own template (best representation of the FOV)

for j in range(0,num_iter):
    template,shift=m.motion_correct(max_shift=max_shift,template=template,show_movie=False);
    templates.append(template)
    shift=np.asarray(shift)
    shifts.append(shift)

plt.plot(np.asarray(shifts).reshape((j+1)*shift.shape[0],shift.shape[1]))

#%% apply shifts to the original movie, you need to reload the original movie 
if False:    
    # here reload the original imaging channel from movie
    totalShifts=np.sum(np.asarray(shifts),axis=0)[:,0:2].tolist()
    m.applyShifstToMovie(totalShifts)

#%% plot final template    
minBrightness=50;
maxBrightness=300;
plt.imshow(templates[-1],cmap=plt.cm.Greys_r,vmin=minBrightness,vmax=maxBrightness)

#%% save motion corrected movie inpython format along with the results. This takes some time now but will save  a lot later...
np.savez(filename_mc,mov=m.mov,templates=templates,shifts=shifts,max_shift=max_shift)    

#%% crop movie after motion correction. 
m.crop(max_shift,max_shift,max_shift,max_shift)    

#%% resize to increase SNR and have better convergence of PCA-ICA
resizeMovie=True
if resizeMovie:
    fx=.25; # downsample a factor of four along x axis
    fy=.25;
    fz=.2; # downsample  a factor of 5 across time dimension
    m.resize(fx=fx,fy=fy,fz=fx)
else:
    fx,fy,fz=1,1,1

#%% compute delta f over f (DF/F)
initTime=time.time()
m.computeDFF(secsWindow=5,quantilMin=40,subtract_minimum=True)
print 'elapsed time:' + str(time.time()-initTime) 


#%%
m1=m.copy()
m1.IPCA_denoise(components = 50, batch = 1000)
    
#%%

#%% compute spatial components via NMF
initTime=time.time()
nmf_spcomps=m.NonnegativeMatrixFactorization(n_components=10,beta=1,tol=5e-7);
print 'elapsed time:' + str(time.time()-initTime) 
matrixMontage(np.asarray(nmf_spcomps),cmap=plt.cm.gray) # visualize components

#%% compute spatial components via ICA PCA
initTime=time.time()
spcomps=m.IPCA_stICA(components=5,mu=1);
print 'elapsed time:' + str(time.time()-initTime) 
matrixMontage(spcomps,cmap=plt.cm.gray) # visualize components
 
#%% extract ROIs from spatial components 
masks=m.extractROIsFromPCAICA(spcomps, numSTD=10, gaussiansigmax=1 , gaussiansigmay=1)
#masks=m.extractROIsFromPCAICA(spcomps, numSTD=10, gaussiansigmax=1 , gaussiansigmay=2)
matrixMontage(np.asarray(masks))

#%%  extract single ROIs from each mask
minPixels=30;
maxPixels=1000;
allMasks=[];
for mask in masks:
    #print np.max(mask)
    for ll in xrange(1,np.max(mask)+1):        
        numPixels=np.sum(np.array(mask==ll));        
        if (numPixels>minPixels and numPixels<maxPixels):
            print numPixels
            numPixels=np.sum(np.array(mask==ll));
            allMasks.append(mask>0)           

allMasks=np.asarray(allMasks,dtype=np.float16)
print allMasks.shape
allMasksForPlot=[kk*ii*1.0 for ii,kk in enumerate(allMasks)]
plt.imshow(np.max(np.asarray(allMasksForPlot,dtype=np.float16),axis=0))

#%% save the results of the analysis
np.savez(filename_analysis,allMasks=allMasks,shifts=shifts,templates=templates,spcomps=spcomps,max_shift=max_shift,fx=fx,fy=fy,fz=fz)




#%%
## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# ERASING AND RELOADING HERE #

#%% if reload results
# warning, this clears all variables!!
%reset 
#%%
reload_files=True
if reload_files:  
#    filename='20150522_1_1_001.tif'
    filename='k26_v1_176um_target_pursuit_001_005.tif'
#    filename='M_FLUO.tif'
    filename_mc=filename[:-4]+'_mc.npz'
    filename_analysis=filename[:-4]+'_analysis.npz'    
    m=XMovie(mat=np.load(filename_mc)['mov'], frameRate=.033);
    vars_=np.load(filename_analysis)
    max_shift=vars_['max_shift']
    allMasks=vars_['allMasks']
    spcomps=vars_['spcomps']
    shifts=vars_['shifts']
    templates=vars_['templates']
    fx=vars_['fx']
    fy=vars_['fy']
    fz=vars_['fz']

#%% load and extract traces
initTime=time.time()
minPercentileRemove=1;
m.mov=m.mov-np.percentile(m.mov,minPercentileRemove);
m.crop(max_shift,max_shift,max_shift,max_shift)
#%% reshape dendrites 
mdend=XMovie(mat=np.asarray(allMasks,dtype=np.float32), frameRate=1);
mdend.resize(fx=1/fx,fy=1/fy)
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
window=int(3/m.frameRate);
minQuantile=20;
for trace in Ftraces:
    #print trace.shape
    traceBL=[np.percentile(trace[i:i+window],minQuantile) for i in xrange(1,len(trace)-window)]
    missing=np.percentile(trace[-window:],minQuantile);
    missing=np.repeat(missing,window+1)
    traceBL=np.concatenate((traceBL,missing))
    tracesDFF.append((trace-traceBL)/traceBL)

tracesDFF=np.asarray(tracesDFF)
#%% save original and DFF traces along with mask
np.savez(filename_traces,allMasks=allMasks,traces=Ftraces,tracesDFF=tracesDFF)



#%% visualize traces
plt.plot(tracesDFF[1:100].T)
#%% visualize traces imagesc style
plt.imshow(tracesDFF,vmax=1.5) 
