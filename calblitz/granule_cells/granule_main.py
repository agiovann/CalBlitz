# -*- coding: utf-8 -*-
"""
Created on Sun Jul 24 17:06:17 2016

@author: agiovann
"""
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
from glob import glob
import os
import scipy
from ipyparallel import Client
import calblitz as cb
from calblitz.granule_cells import utils_granule as gc
#%%
fls=glob('2016*.tif')     

fls.sort()     
triggers,trigger_names=extract_triggers(fls,read_dictionaries=False)    
 
np.savez('all_triggers.npz',triggers=triggers,trigger_names=trigger_names)   
#%% get information from eyelid traces
t_start=time()     
res_bt=gc.get_behavior_traces('20160705103903_cam2.h5',t0=0,t1=8.0,freq=60,ISI=.25,draw_rois=False,plot_traces=False,mov_filt_1d=True,window_lp=5)   
t_end=time()-t_start
print t_end

#%%   
with np.load('all_triggers.npz') as at:
    triggers_img=at['triggers']
    trigger_names_img=at['trigger_names']  
#%%
tm=res_bt['time']
eye_traces=np.array(res_bt['eyelid'])
idx_CS_US=res_bt['idx_CS_US']
idx_US=res_bt['idx_US']
idx_CS=res_bt['idx_CS']

idx_ALL=np.sort(np.hstack([idx_CS_US,idx_US,idx_CS]))
eye_traces,amplitudes_at_US, trig_CRs=gc.process_eyelid_traces(eye_traces,tm,idx_CS_US,idx_US,idx_CS,thresh_CR=.05,time_CR_on=-.1,time_US_on=.05)

idxCSUSCR = trig_CRs['idxCSUSCR']
idxCSUSNOCR = trig_CRs['idxCSUSNOCR']
idxCSCR = trig_CRs['idxCSCR']
idxCSNOCR = trig_CRs['idxCSNOCR']
idxNOCR = trig_CRs['idxNOCR']
idxCR = trig_CRs['idxCR']
idxUS = trig_CRs['idxUS']

pl.plot(tm,np.mean(eye_traces[idxCSUSCR],0))       
pl.plot(tm,np.mean(eye_traces[idxCSUSNOCR],0))     
pl.plot(tm,np.mean(eye_traces[idxCSCR],0))
pl.plot(tm,np.mean(eye_traces[idxCSNOCR],0))    
pl.plot(tm,np.mean(eye_traces[idx_US],0))
pl.legend(['idxCSUSCR','idxCSUSNOCR','idxCSCR','idxCSNOCR','idxUS'])

pl.xlim([-.5,1])
#%%
bins=np.arange(0,1,.01)
pl.hist(amplitudes_at_US[idxCR],bins=bins)
pl.hist(amplitudes_at_US[idxNOCR],bins=bins)


#%%
wheel_traces, movement_at_CS, trigs_mov = gc.process_wheel_traces(np.array(res_bt['wheel']),tm,thresh_MOV_iqr=3,time_CS_on=-.25,time_US_on=0)    
           
#%% 
f_results= glob('*.results_analysis.npz')
f_results.sort()
for rs in f_results:
    print rs
#%% load results and put them in lists
A_s,C_s,YrA_s, Cn_s, b_s, f_s, shape =  gc.load_results(f_results)     
#%%
B_s, lab_imgs, cm_s  = gc. threshold_components(A_s,shape, min_size=5,max_size=50,max_perc=.5)
#%%
for i,A_ in enumerate(B_s):
     sizes=np.array(A_.sum(0)).squeeze()
     pl.subplot(2,3,i+1)
     pl.imshow(np.reshape(A_.sum(1),shape,order='F'),cmap='gray',vmax=.5)
#%% compute mask distances 
max_dist=30
D_s=gc.distance_masks(B_s,cm_s,max_dist)       
np.savez('distance_masks.npz',D_s=D_s)
#%%
with np.load('distance_masks.npz') as ld:
    D_s=ld['D_s']
#%%
for ii,D in enumerate(D_s):
    pl.subplot(3,3,ii+1)
    pl.imshow(D,interpolation='None')
#%% find matches
matches,costs =  gc.find_matches(D_s, print_assignment=False)
#%%
neurons=[]
num_neurons=0
Yr_tot=[]
num_chunks=len(C_s)
for idx in range(len(matches[0][0])):
    neuron=[]
    neuron.append(idx)
    Yr=YrA_s[0][idx]+C_s[0][idx]
    for match,cost,chk in zip(matches,costs,range(1,num_chunks)):
        rows,cols=match        
        m_neur=np.where(rows==neuron[-1])[0].squeeze()
        if m_neur.size > 0:                           
            if cost[m_neur]<=.6:
                neuron.append(cols[m_neur])
                Yr=np.hstack([Yr,YrA_s[chk][idx]+C_s[chk][idx]])
            else:                
                break
        else:
            break
    if len(neuron)>len(matches):           
        num_neurons+=1        
        neurons.append(neuron)
        Yr_tot.append(Yr)
        
print num_neurons    
neurons=np.array(neurons).T
Yr_tot=np.array(Yr_tot)
#%%
np.savez('neurons_matching.npz',matches=matches,costs=costs,neurons=neurons,D_s=D_s)
#%%
downs_factor=.3
tmpl_name='20160705103903_00-template_total.npz'
with np.load(tmpl_name) as ld:
    mov_names_each=ld['movie_names']

for idx,mvs in enumerate(mov_names_each):
    print idx 
    
    chunk_sizes=[]
    
    for mv in mvs:
        base_name=os.path.splitext(os.path.split(mv)[-1])[0]
#        mov_chunk_name=glob(base_name+'*.mmap')[0]
        with np.load(base_name+'.npz') as ld:
            TT=len(ld['shifts'])            
#        _,_,TT=cse.utilities.load_memmap(mov_chunk_name)
        chunk_sizes.append(TT)
        
        
    num_chunks=np.sum(chunk_sizes)
    
    A = A_s[idx][:,neurons[idx]] 
    nA = (A.power(2)).sum(0)
    b = b_s[idx]
    f = f_s[idx]
    bckg=cb.movie(cb.to_3D(b.dot(f).T,(-1,shape[0],shape[1])),fr=33*downs_factor)
    b_size=np.shape(bckg)[0]
    bckg=bckg.resize(1,1,1.*num_chunks/b_size)
    if num_chunks != bckg.shape[0]:
        raise Exception('The number of frames are not matching')
    counter=0
    for jj,mv in enumerate(mvs):
        mov_chunk_name=os.path.splitext(os.path.split(mv)[-1])[0]+'.hdf5'        
        print mov_chunk_name
        m=cb.load(mov_chunk_name,fr=33)
        
        m=m-bckg[counter:counter+chunk_sizes[jj]]            
        counter+=chunk_sizes[jj]
        
#        (m).play(backend='opencv',gain=10.,fr=33)
        m=np.reshape(m,(np.prod(shape),-1),order='F')
        Y_r=A.T.dot(m)
        Y_r= scipy.sparse.linalg.spsolve(scipy.sparse.spdiags(np.sqrt(nA),0,nA.size,nA.size),Y_r)
#        Y_r=A.T.dot(Yr-)
                               
#%%
nA = full(sum(A.^2))';  % energy of each row
Y_r = spdiags(sqrt(nA),0,length(nA),length(nA))\(A'*(Y-full(b)*f)); 
    % raw data trace filtered with ROI
C_sorted = spdiags(sqrt(nA),0,length(nA),length(nA))*C;
[~,ind_sor] = sort(max(C_sorted,[],2),'descend');
figure;
for i = 1:nr
    plot(1:T,Y_r(ind_sor(i),:),1:T,C_sorted(ind_sor(i),:)); 
    %title(sprintf('Sorted ROI %i, plane %i',i,ind_neur(ind_sor(i))));
    xlabel('Timestep','fontsize',14,'fontweight','bold');
    legend('Raw data (averaged over ROI)','Temporal Fit','fontsize',14,'fontweight','bold');
    %subplot(212);plot(1:T,Y_r(ind_sor(i),:) - C_sorted(ind_sor(i),:));    
    drawnow; pause; 
end

    
#%%            
for idx,B in enumerate(A_s):
     pl.subplot(2,3,idx+1)
     pl.imshow(np.reshape(B[:,neurons[idx]].sum(1),shape,order='F'))
#%%
for neuro in range(num_neurons):
    for idx,B in enumerate(A_s):
         pl.subplot(2,3,idx+1)
         pl.imshow(np.reshape(B[:,neurons[idx][neuro]].sum(1),shape,order='F'))
    pl.pause(.01)     
    for idx,B in enumerate(A_s):
        pl.subplot(2,3,idx+1)
        pl.cla()       

#%%
idx=0
for  row, column in zip(matches[idx][0],matches[idx][1]):
    value = D_s[idx][row,column]
    if value < .5:
#        pl.subplot(1,2,1)
        pl.cla() 
        pl.imshow(np.reshape(B_s[idx][:,row].todense(),(512,512),order='F'),cmap='gray',interpolation='None')    
#        pl.subplot(1,2,2)
#        pl.cla() 
        pl.imshow(np.reshape(B_s[idx+1][:,column].todense(),(512,512),order='F'),alpha=.5,cmap='hot',interpolation='None')               
        if B_s[idx][:,row].T.dot(B_s[idx+1][:,column]).todense() == 0:
            print 'Flaw'
            
        pl.pause(.3)


#%%

#%%
Ftraces=C+YrA

min_chunk=np.inf
max_chunk=0

for fr in frames_per_chink:
       min_chunk=np.int(np.minimum(min_chunk,fr))
       max_chunk=np.int(np.maximum(max_chunk,fr))

#%%
Ftraces_mat=np.zeros([len(frames_per_chink),len(C),max_chunk])
abs_frames=np.arange(max_chunk)
idx_read=0
crs=idxCR[idxCR>=121]-120
nocrs=idxNOCR[idxNOCR>=121]-120
uss=idxUS[idxUS>=121]-120
for idx,fr in enumerate(frames_per_chink):
    print idx

    if fr!=max_chunk:

        f1=scipy.interpolate.interp1d(np.arange(fr) , Ftraces[:,idx_read:idx_read+fr] ,axis=1, bounds_error=False, kind='linear')  
        Ftraces_mat[idx]=np.array(f1(abs_frames))
        
    else:
        
        Ftraces_mat[idx]=Ftraces[:,idx_read:idx_read+fr]
    
    
    idx_read=idx_read+fr
#%%
pl.close()
for cell in range(Ftraces_mat.shape[1])[57:]:   
    pl.cla()
    print cell
    tr_cr=np.median(Ftraces_mat[crs,cell,:],axis=(0))    
    tr_nocr=np.median(Ftraces_mat[nocrs,cell,:],axis=(0))    
    tr_us=np.median(Ftraces_mat[uss,cell,:],axis=(0))    
    pl.plot(tr_cr,'b')
    pl.plot(tr_nocr,'g')
    pl.plot(tr_us,'r')
    pl.legend(['CR+','CR-','US'])
    pl.pause(1)
#%%
pl.close()
for cell in [57,56,47,44,34,33,6,23]:
    
    a=np.mean(Ftraces_mat[crs,cell,28:31],-1)
    b=np.mean(Ftraces_mat[nocrs,cell,28:31],-1)
    tts=scipy.stats.ttest_ind(a,b)
    tts.pvalue
    
    tmf=(np.arange(max_chunk)-29)/(30*.3)
    tr_cr=np.median(Ftraces_mat[crs,cell,:],axis=(0))    
    tr_nocr=np.median(Ftraces_mat[nocrs,cell,:],axis=(0))    
    tr_us=np.median(Ftraces_mat[uss,cell,:],axis=(0))    
    pl.subplot(1,2,1)
    pl.cla()
    pl.plot(tmf,tr_cr-np.median(tr_cr[10:23]))
    pl.plot(tmf,tr_nocr-np.median(tr_nocr[10:23]))
    pl.plot(tmf+ISI,tr_us-np.median(tr_us[10:30])) 
    pl.xlim([-.5,1])
    pl.xlabel('time from US (s)')
    pl.ylabel('A.U.')
    pl.legend(['CR+','CR-','US'])
    pl.subplot(1,2,2)
    pl.axis('off')
    #cse.utilities.plot_contours(A[:,cell:cell+1],Cn,thr=0.9)
    pl.imshow(np.reshape(A[:,cell],(512,512),order='F'),cmap='gray')
    pl.pause(2)
    
#%%        

#
#base_name='20160518133747_'
#cam1=base_name+'cam1.h5'
#cam2=base_name+'cam2.h5'
#meta_inf=base_name+'data.h5'
#
#mtot=[]
#eye_traces=[]
#tims=[]
#trial_info=[]
#
#with h5py.File(cam2) as f:
#
#    with h5py.File(meta_inf) as dt:
#
#        rois=np.asarray(dt['roi'],np.float32)
#        
#        trials = f.keys()
#        trials.sort(key=lambda(x): np.int(x.replace('trial_','')))
#        trials_idx=[np.int(x.replace('trial_',''))-1 for x in trials]
#
#        
#   
#        
#        for tr,idx_tr in zip(trials,trials_idx):
#            
#            print tr
#
#            trial=f[tr]  
#
#            mov=np.asarray(trial['mov'])        
#
#            if 0:
#
#                pl.imshow(np.mean(mov,0))
#                pts=pl.ginput(-1)
#                pts = np.asarray(pts, dtype=np.int32)
#                data = np.zeros(np.shape(mov)[1:], dtype=np.int32)
#        #        if CV_VERSION == 2:
#                #lt = cv2.CV_AA
#        #        elif CV_VERSION == 3:
#                lt = cv2.LINE_AA
#                cv2.fillConvexPoly(data, pts, (1,1,1), lineType=lt)
#                rois[0]=data
##            eye_trace=np.mean(mov*rois[0],axis=(1,2))
##            mov_trace=np.mean((np.diff(np.asarray(mov,dtype=np.float32),axis=0)**2)*rois[1],axis=(1,2))
#            mov=np.transpose(mov,[0,2,1])
#            
#            mov=mov[:,:,::-1]
#
#            if  mov.shape[0]>0:
#                ts=np.array(trial['ts'])
#                if np.size(ts)>0:
#        #            print (ts[-1,0]-ts[0,0])
#                    new_ts=np.linspace(0,ts[-1,0]-ts[0,0],np.shape(mov)[0])
#                    
#                    print 1/np.mean(np.diff(new_ts))
#                    tims.append(new_ts)
#                    
#                mov=cb.movie(mov*rois[0][::-1].T,fr=1/np.mean(np.diff(new_ts)))
#                x_max,y_max=np.max(np.nonzero(np.max(mov,0)),1)
#                x_min,y_min=np.min(np.nonzero(np.max(mov,0)),1)
#                mov=mov[:,x_min:x_max,y_min:y_max]                                
#                mov=np.mean(mov, axis=(1,2))
#        
#                if mov.ndim == 3:
#                    window_hp=(177,1,1)
#                    window_lp=(7,1,1)
#                    bl=signal.medfilt(mov,window_hp)
#                    mov=signal.medfilt(mov-bl,window_lp)
#
#                else:
#                    window_hp=201
#                    window_lp=3
#                    bl=signal.medfilt(mov,window_hp)
##                    bl=cse.utilities.mode_robust(mov)
#                    mov=signal.medfilt(mov-bl,window_lp)
#
#                    
#                if mov.ndim == 3:
#                    eye_traces.append(np.mean(mov, axis=(1,2)))
#                else:
#                    eye_traces.append(mov)
#                    
#                mtot.append(mov)   
#                trial_info.append(dt['trials'][idx_tr,:])
#        #            cb.movie(mov,fr=1/np.mean(np.diff(new_ts)))

#%%
#%%
sub_trig_img=downsample_triggers(triggers_img.copy(),fraction_downsample=.3)
#%%
if num_frames_movie != triggers[-1,-1]:
        raise Exception('Triggers values do not match!')
        
#%% 
#fnames=[]
#sub_trig_names=trigger_names[39:95].copy()
#sub_trig=triggers[39:95].copy().T
#for a,b in zip(sub_trig_names,sub_trig):
#    fnames.append(a+'.hdf5')
#
#fraction_downsample=.333333333333333333333; # useful to downsample the movie across time. fraction_downsample=.1 measn downsampling by a factor of 10
#sub_trig[:2]=np.round(sub_trig[:2]*fraction_downsample)
#sub_trig[-1]=np.floor(sub_trig[-1]*fraction_downsample)
#sub_trig[-1]=np.cumsum(sub_trig[-1])
#fname_new=cse.utilities.save_memmap(fnames,base_name='Yr',resize_fact=(1,1,fraction_downsample),remove_init=0,idx_xy=(slice(90,-10,None),slice(30,-120,None)))
##%%
#m=cb.load(fname_new,fr=30*fraction_downsample)
#T,d1,d2=np.shape(m)
#%%
#if T != sub_trig[-1,-1]:
#    raise Exception('Triggers values do not match!')
#%% how to take triggered aligned movie
wvf=mmm.take(trg)
#%%
newm=m.take(trg,axis=0)
newm=newm.mean(axis=1)
#%%
(newm-np.mean(newm,0)).play(backend='opencv',fr=3,gain=2.,magnification=1,do_loop=True)
#%%v
Yr,d1,d2,T=cse.utilities.load_memmap(fname_new)
d,T=np.shape(Yr)
Y=np.reshape(Yr,(d1,d2,T),order='F') # 3D version of the movie 

#%%

pl.plot(np.nanmedian(np.array(eye_traces).T,1))

#%%
mov = np.concatenate(mtot,axis=0)           
m1=cb.movie(mov,fr=1/np.mean(np.diff(new_ts)))
#x_max,y_max=np.max(np.nonzero(np.max(m,0)),1)
#x_min,y_min=np.min(np.nonzero(np.max(m,0)),1)
#m1=m[:,x_min:x_max,y_min:y_max]
#%% filters
b, a = signal.butter(8, [.05, .5] ,'bandpass')
pl.plot(np.mean(m1,(1,2))-80)
pl.plot(signal.lfilter(b,a,np.mean(m1,(1,2))),linewidth=2)
#%%
m1.play(backend='opencv',gain=1.,fr=30,magnification=3)
#%% NMF
comps, tim,_=cb.behavior.extract_components(np.maximum(0,m1-np.min(m1,0)),n_components=4,init='nndsvd',l1_ratio=1,alpha=0,max_iter=200,verbose=True)
pl.plot(np.squeeze(np.array(tim)).T)
#%% ICA
from sklearn.decomposition import FastICA
fica=FastICA(n_components=3,whiten=True,max_iter=200,tol=1e-6)
X=fica.fit_transform(np.reshape(m1,(m1.shape[0],m1.shape[1]*m1.shape[2]),order='F').T,)
pl.plot(X)
#%%
for count,c in enumerate(comps):
    pl.subplot(2,3,count+1)
    pl.imshow(c)
    
#%%
md=cse.utilities.mode_robust(m1,0)
mm1=m1*(m1<md)
rob_std=np.sum(mm1**2,0)/np.sum(mm1>0,0)
rob_std[np.isnan(rob_std)]=0
mm2=m1*(m1>(md+rob_std))
#%%
            
dt = h5py.File('20160423165229_data.h5')   
#sync for software
np.array(dt['sync'])
dt['sync'].attrs['keys']     
dt['trials']
dt['trials'].attrs
dt['trials'].attrs['keys']
# you needs to apply here the sync on dt['sync'], like, 
us_time_cam1=np.asarray(dt['trials'])[:,3] - np.array(dt['sync'])[1]
# main is used as the true time stamp, and you can adjust the value with respect to main sync value
np.array(dt['sync']) # these are the values read on a unique clock from the three threads
#%%
from skimage.external import tifffile

tf=tifffile.TiffFile('20160423165229_00001_00001.tif')   
imd=tf.pages[0].tags['image_description'].value
for pag in tf.pages:
    imd=pag.tags['image_description'].value
    i2cd=si_parse(imd)['I2CData']
    print (i2cd)
##%%
#with h5py.File('20160705103903_cam2.h5') as f1:
#    for k in f1.keys()[:1]:
#        m = np.array(f1[k]['mov'])
#        
#        
#pl.imshow(np.mean(m,0),cmap='gray')
##%%
#with h5py.File('20160705103903_data.h5') as f1:
#    print f1.keys()    
#    rois= np.array(f1['roi'])
##%%
#with h5py.File('20160705103903_cam2.h5') as f1:
#    for k in f1.keys()[:1]:
#        m = np.array(f1[k]['mov'])
#        
#        
#pl.imshow(np.mean(m,0),cmap='gray')
#pl.imshow(rois[0],alpha=.3)
#pl.imshow(rois[1],alpha=.3)
#       