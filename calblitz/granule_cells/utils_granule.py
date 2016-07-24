# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 17:56:14 2016

@author: agiovann
"""
#%
try:
    %load_ext autoreload
    %autoreload 2
    print 1
except:
    print 1

import cv2
import h5py
import numpy as np
import pylab as pl
import glob
from skimage.external import tifffile
import time
import ca_source_extraction as cse

import calblitz as cb
from scipy import signal
import glob
import scipy
import sys
from ipyparallel import Client
#%% start server
backend='local'
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

print 'Using '+ str(len(c)) + ' processes'
dview=c[:len(c)]
#%% Process triggers
def extract_triggers(file_list,read_dictionaries=False): 
    
    """Extract triggers from Bens' tiff file and create readable dictionaries

    Parameterskdkd
    -----------
    file_list: list of tif files or npz files containing the iage description
    
    Returns
    -------
    triggers: list 
        [idx_CS, idx_US, trial_type, number_of_frames]. Trial types: 0 CS alone, 1 US alone, 2 CS US
   
   trigger_names: list
        file name associated (without extension)
        
    Example: 
    
    fls=glob.glob('2016*.tif')     
    fls.sort()     
    triggers,trigger_names=extract_triggers(fls[:5],read_dictionaries=False)     
    np.savez('all_triggers.npz',triggers=triggers,trigger_names=trigger_names)     

    """
    triggers=[]
    
    trigger_names=[]
    
    for fl in file_list:
        
        print fl   
        
        fn=fl[:-4]+'_ImgDescr.npz'
        
        if read_dictionaries:
            
            with np.load(fn) as idr:
                
                image_descriptions=idr['image_descriptions']
                
        else:
            
            image_descriptions=cb.utils.get_image_description_SI(fl)
            
            print '*****************'
            
            np.savez(fn,image_descriptions=image_descriptions)
            

        trig_vect=np.zeros(4)*np.nan    
        
        for idx,image_description in enumerate(image_descriptions): 
            
            i2cd=image_description['I2CData']
            
            if type(i2cd) is str:
                
                if i2cd.find('US_ON')>=0:
                    
                    trig_vect[1]=image_description['frameNumberAcquisition']-1
                    
                if i2cd.find('CS_ON')>=0:
                    
                    trig_vect[0]=image_description['frameNumberAcquisition']-1  
                    
        
        if np.nansum(trig_vect>0)==2:

            trig_vect[2]=2
            
        elif trig_vect[0]>0:
    
            trig_vect[2]=0
    
        elif trig_vect[1]>0:
    
            trig_vect[2]=1  
        else:
            raise Exception('No triggers present in trial')        
        
        trig_vect[3]=idx+1
        
        triggers.append(trig_vect)
        
        trigger_names.append(fl[:-4])
        
        print triggers[-1]
        
    return triggers,trigger_names



#%%   
with np.load('all_triggers.npz') as at:
    triggers_img=at['triggers']
    trigger_names_img=at['trigger_names']  
    
#%%
def downsample_triggers(triggers,fraction_downsample=1):
    """ downample triggers so as to make them in line with the movies
    Parameters
    ----------
    
    triggers: list
        output of  extract_triggers function
    
    fraction_downsample: float
        fraction the data is shrinked in the time axis
    """
    
    triggers[:,[0,1,3]]=np.round(triggers[:,[0,1,3]]*fraction_downsample)
#    triggers[-1,[0,1,3]]=np.floor(triggers[-1,[0,1,3]]*fraction_downsample)
#    triggers[-1]=np.cumsum(triggers[-1])
    
#    real_triggers=triggers[:-1]+np.concatenate([np.atleast_1d(0), triggers[-1,:-1]])[np.newaxis,:]
#
#    trg=real_triggers[1][triggers[-2]==2]+np.arange(-5,8)[:,np.newaxis]  
#
#    trg=np.int64(trg)

    return triggers
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
#%% SCRIPT IMAGING



#%%
#time_st=time.time()
#
#range_frame=np.arange(180,230)
#base_name='20160514163145_'
#fls=glob.glob(base_name+'*.tif')
##fls=glob.glob('20160423181901*.tif')
##fls=glob.glob('20160423165229*.tif')
##range_frame=np.arange(20,175)
#
#fls.sort()
#frCS=np.zeros([len(fls),len(range_frame)-1]);
#frUS=np.zeros([len(fls),len(range_frame)-1]);
#traces=np.zeros([len(fls),len(range_frame)-1]);
#
#with tifffile.TiffFile(fls[0])as tf:  
#    d1,d2=tf.pages[0].shape
#    
#mov_avg=np.zeros([len(range_frame)-1,d1,d2])
#showit=False
#for idxfl,fl in enumerate(fls):
#    print fl
#    with tifffile.TiffFile(fl) as tf:   
#        for idx,pag in enumerate(tf.pages[range_frame[0]:range_frame[-1]]):
#    #        i2cd=si_parse(pag.tags['image_description'].value)['I2CData']
#            mov_curr = pag.asarray() 
#            mov_avg[idx] = mov_avg[idx] + mov_curr          
#            field=pag.tags['image_description'].value
#            idx_start = field.find('I2CData') 
#            idx_end=idx_start+field[idx_start:].find('\n')
#            i2cd=field[idx_start:idx_end].split('=')[-1].strip().strip('{}')
#            traces[idxfl,idx]=np.mean(mov_curr)
#            if len(i2cd)>0:
#                #print (i2cd)
#                if i2cd.find('US_ON')>=0:
#                    frUS[idxfl,idx]=1                    
#                if i2cd.find('CS_ON')>=0:
#                    showit=True
#                    frCS[idxfl,idx]=1;
#        if showit:    
#            pl.subplot(2,1,1)
#            pl.imshow(frUS+2*frCS,aspect='auto',interpolation='none')
#            pl.axis('tight')
#            pl.subplot(2,1,2)
#        #    pl.cla()
#            
#            tr=traces[idxfl]
#            tr=(tr-np.min(tr))
#            tr=tr/np.max(tr)
#            tr_tr=np.mean(frUS+frCS,axis=0)
#            pl.plot(tr,color=[.9,.9,.9])
#            pl.plot(tr_tr/np.max(tr_tr),'r')
#            pl.axis('tight')    
#    #    pl.imshow(traces,aspect='auto',interpolation='none')
#        print time.time()-time_st
#        pl.pause(.1)    
#        showit=False
#
#mean_tr=np.mean(traces,axis=0)
#mean_tr=mean_tr-np.min(mean_tr)
#mean_tr=mean_tr/np.max(mean_tr)                
#pl.plot(mean_tr,'k',linewidth=2)
#%% PLAY AVG MOVIE
#mov_avg=mov_avg/np.max(mov_avg)  
##mov_avg=mov_avg-np.median(mov_avg,0)          
#mov_avg[:,20:40,1:20]=np.mean(frUS,0)[:,np.newaxis,np.newaxis]
#mov_avg[:,1:20,1:20]=np.mean(frCS,0)[:,np.newaxis,np.newaxis]
#m=cb.movie(mov_avg,fr=30)
#
#
#m.play(backend='opencv',fr=8,gain=3.,magnification=1)
#%%
mtot=[]
eye_traces=[]
tims=[]
trial_info=[]


def get_behavior_traces(fname,t0,t1,freq,ISI,draw_rois=False):
    CS_ALONE=0
    US_ALONE=1
    CS_US=2
    meta_inf = fname[:-7]+'data.h5'

    time_abs=np.linspace(t0,t1,freq*(t1-t0))

    T=len(time_abs)
    t_us=0
    t_cs=0
    n_samples_ISI=np.int(ISI*freq)
    t_uss=[]
    ISIs=[]
    with h5py.File(fname) as f:

        with h5py.File(meta_inf) as dt:
            
            rois=np.asarray(dt['roi'],np.float32)

            trials = f.keys()

            trials.sort(key=lambda(x): np.int(x.replace('trial_','')))

            trials_idx=[np.int(x.replace('trial_',''))-1 for x in trials]
            
            trials_idx_=[]
            
        
            
            for tr,idx_tr in zip(trials[:],trials_idx[:]):
                pl.cla()

                print tr
               
               
                
                trial=f[tr]  

                mov=np.asarray(trial['mov'])        

                if draw_rois:
                    
                    pl.imshow(np.mean(mov,0))

                    pts=pl.ginput(-1)

                    pts = np.asarray(pts, dtype=np.int32)

                    data = np.zeros(np.shape(mov)[1:], dtype=np.int32)
            #        if CV_VERSION == 2:
                    #lt = cv2.CV_AA
            #        elif CV_VERSION == 3:
                    lt = cv2.LINE_AA
                    
                    cv2.fillConvexPoly(data, pts, (1,1,1), lineType=lt)

                    rois[0]=data
    #            eye_trace=np.mean(mov*rois[0],axis=(1,2))
    #            mov_trace=np.mean((np.diff(np.asarray(mov,dtype=np.float32),axis=0)**2)*rois[1],axis=(1,2))
                mov=np.transpose(mov,[0,2,1])
                
                mov=mov[:,:,::-1]

                if  mov.shape[0]>0:

                    ts=np.array(trial['ts'])

                    if np.size(ts)>0:

                        print (ts[0,0])
                        new_ts=np.linspace(0,ts[-1,0]-ts[0,0],np.shape(mov)[0])
                        
                        if dt['trials'][idx_tr,-1] == US_ALONE:
                            t_us=np.maximum(t_us,dt['trials'][idx_tr,3]-dt['trials'][idx_tr,0])  
                            print n_samples_ISI                                                      
                            mmm=mov[:n_samples_ISI].copy()
                            mov=mov[:-n_samples_ISI]
                            mov=np.concatenate([mmm,mov])
                            print mov.shape                   
                            
                        elif dt['trials'][idx_tr,-1] == CS_US: 
                            t_cs=np.maximum(t_cs,dt['trials'][idx_tr,2]-dt['trials'][idx_tr,0])                                
                            t_us=np.maximum(t_us,dt['trials'][idx_tr,3]-dt['trials'][idx_tr,0])   
                            t_uss.append(t_us)                                                     
                            ISI=t_us-t_cs
                            ISIs.append(ISI)
                            n_samples_ISI=np.int(ISI*freq)
                        else:
                            t_cs=np.maximum(t_cs,dt['trials'][idx_tr,2]-dt['trials'][idx_tr,0])                                
                        
                        print 1/np.mean(np.diff(new_ts))
                        
                        new_ts=new_ts
                        tims.append(new_ts)
                        
                    mov=cb.movie(mov*rois[0][::-1].T,fr=1/np.mean(np.diff(new_ts)))
                    
                    x_max,y_max=np.max(np.nonzero(np.max(mov,0)),1)

                    x_min,y_min=np.min(np.nonzero(np.max(mov,0)),1)

                    mov=mov[:,x_min:x_max,y_min:y_max] 
                         
                         
#                    mpart=mov[:20].copy()
#                    md=cse.utilities.mode_robust(mpart.flatten())
#                    N=np.sum(mpart<=md)
#                    mpart[mpart>md]=md
#                    mpart[mpart==0]=md
#                    mpart=mpart-md
#                    std=np.sqrt(np.sum(mpart**2)/N)
#                    thr=md+10*std
#                    
#                    thr=np.minimum(255,thr)
#                    return mov     
                    if 1:
                        mov=np.mean(mov, axis=(1,2))
            
            
                    pl.plot(mov-np.min(mov))
                    if mov.ndim == 3:

                        window_hp=(177,1,1)

                        window_lp=(7,1,1)

                        bl=signal.medfilt(mov,window_hp)

                        mov=signal.medfilt(mov-bl,window_lp)
    
                    else:

                        window_hp=201

                        window_lp=3

                        bl=signal.medfilt(mov,window_hp)
    #                    bl=cse.utilities.mode_robust(mov)
                        mov=signal.medfilt(mov-bl,window_lp)The value of the ocmponents 
    
                        
                    if mov.ndim == 3:
                        
                        eye_=np.atleast_2d(np.mean(mov, axis=(1,2)))

                    else:

                        eye_=np.atleast_2d(mov)
                    
                    
                    if np.abs(new_ts[-1]  - time_abs[-1])>.5:

                       raise Exception('Time duration is significantly larger or smaller than reference time')

                    
                    eye_=np.squeeze(eye_)


#                    new_trace_,new_time_=scipy.signal.resample(eye_, T, t=new_ts)
                    
                    f1=scipy.interpolate.interp1d(new_ts , eye_,bounds_error=False,kind='linear')                    
                    new_trace_=np.array(f1(time_abs))
                    pl.plot(new_trace_,'r')    

                    pl.pause(.01)
                                            
                    trials_idx_.append(idx_tr)

                    eye_traces.append(new_trace_)
                                            
                    mtot.append(mov)   
                    
                    trial_info.append(dt['trials'][idx_tr,:])   
            
            official_T=scipy.stats.mode(T)
            official_T=np.squeeze(official_T.mode)
            
#            new_traces=[]
#            for trace_,time_ in zip(eye_traces,tims):
#
#                print time_
#
#                if np.abs(time_[-1] - time_abs[-1])>.5:
#
#                    raise Exception('Time duration is significantly larger or smaller than reference time')
#
#                
#                trace_=np.squeeze(trace_)
#
#                pl.plot(trace_,'b')
#
#                new_trace_=np.interp(time_abs, time_ , trace_)
#                
##                new_trace_,new_time_=scipy.signal.resample(trace_, official_T, t=time_)
#from munkres import Munkres
#                pl.plot(new_trace_,'r')
#
#                pl.pause(.01)
#
#                pl.cla()
#                
#                new_traces.append(new_trace_)
            
            res=dict()
            
            res['eyelid'] =  eye_traces              
            res['time'] = time_abs - np.median(t_uss) 
            res['trials'] = trials_idx_
            res['trial_info']=trial_info
            
            return res

#%%            
res_bt=get_behavior_traces('20160705103903_cam2.h5',t0=0,t1=8.0,freq=60,ISI=.25,draw_rois=False)   
#%%
CS_ALONE=0
US_ALONE=1
CS_US=2
from munkres import Munkres
trial_info=res_bt['trial_info']
tm=res_bt['time']

traces=np.array(res_bt['eyelid'])
idx_original=np.arange(len(traces))

idx_CS_US=np.where(map(int,np.array(trial_info)[:,-1]==CS_US))[0]
idx_US=np.where(map(int,np.array(trial_info)[:,-1]==US_ALONE))[0]
idx_CS=np.where(map(int,np.array(trial_info)[:,-1]==CS_ALONE))[0]

traces=traces/np.nanmax(np.nanmedian(traces[np.hstack([idx_CS_US,idx_US])],0))



#%%
thresh_CR=.05
t_thresh_CR=-.1
amplitudes_at_US=np.mean(traces[:,np.logical_and(tm>t_thresh_CR ,tm<=0.05)],1)
print np.mean(amplitudes_at_US[idx_CS_US])
print np.mean(amplitudes_at_US[idx_US])

idxCSUSCR=idx_CS_US[np.where(amplitudes_at_US[idx_CS_US]>.1)[-1]]
idxCSUSNOCR=idx_CS_US[np.where(amplitudes_at_US[idx_CS_US]<.1)[-1]]
idxCSCR=idx_CS[np.where(amplitudes_at_US[idx_CS]>.1)[-1]]
idxCSNOCR=idx_CS[np.where(amplitudes_at_US[idx_CS]<.1)[-1]]
idxNOCR=np.union1d(idxCSUSNOCR,idxCSNOCR)
idxCR=np.union1d(idxCSUSCR,idxCSCR)
idxUS=idx_US
#%%
pl.plot(tm,np.mean(traces[idxCSUSCR],0))       
pl.plot(tm,np.mean(traces[idxCSUSNOCR],0))     
pl.plot(tm,np.mean(traces[idxCSCR],0))
pl.plot(tm,np.mean(traces[idxCSNOCR],0))    
pl.plot(tm,np.mean(traces[idx_US],0))
pl.legend(['idxCSUSCR','idxCSUSNOCR','idxCSCR','idxCSNOCR','idxUS'])

pl.xlim([-.5,1])
#%%
pl.plot(tm,np.mean(traces[idxCR],0))       
pl.plot(tm,np.mean(traces[idxCSNOCR],0))     
pl.legend(['idxNOCR','idxCR'])

pl.xlim([-.5,1])
#%%
from scipy.sparse import csc,csr,coo_matrix
f_results=[
'20160705103903_00001_00001-#-30_d1_512_d2_512_d3_1_order_C_frames_2095_.results_analysis.npz',
'20160705103903_00031_00001-#-30_d1_512_d2_512_d3_1_order_C_frames_2096_.results_analysis.npz',
'20160705103903_00061_00001-#-30_d1_512_d2_512_d3_1_order_C_frames_2096_.results_analysis.npz',
'20160705103903_00091_00001-#-30_d1_512_d2_512_d3_1_order_C_frames_2094_.results_analysis.npz',
'20160705103903_00121_00001-#-57_d1_512_d2_512_d3_1_order_C_frames_3977_.results_analysis.npz']
Cns=[]
i=0
A_s=[]
C_s=[]
YrA_s=[]
Cn_s=[]
shape = None
for f_res in f_results:
    print f_res
    i+=1
##    pl.subplot(2,3,i)
    with  np.load(f_res) as ld:
        A_s.append(csc.csc_matrix(ld['A2']))
        C_s.append(ld['C2'])
        YrA_s.append(ld['YrA'])
        Cn_s.append(ld['Cn'])
#        pl.imshow(Cn_s[-1],cmap='gray')
        if shape is not None:
            shape_new=(ld['d1'],ld['d2'])
            if shape_new != shape:
                raise Exception('Shapes of FOVs not matching')
            else:
                shape = shape_new
        else:            
            shape=(ld['d1'],ld['d2'])

#%%
#B_s=[];
#for i,A_ in enumerate(A_s):
#    print i
#    B_s.append(csc.csc_matrix(((A_.todense()/A_.max(0).todense())>.5)*1.))
#%%
from scipy.spatial.distance import cdist

def feature_dist(input):
    """
    Takes a labeled array as returned by scipy.ndimage.label and 
    returns an intra-feature distance matrix.
    """
    I, J = np.nonzero(input)
    labels = input[I,J]
    coords = np.column_stack((I,J))

    sorter = np.argsort(labels)
    labels = labels[sorter]
    coords = coords[sorter]

    sq_dists = cdist(coords, coords, 'sqeuclidean')

    start_idx = np.flatnonzero(np.r_[1, np.diff(labels)])
    nonzero_vs_feat = np.minimum.reduceat(sq_dists, start_idx, axis=1)
    feat_vs_feat = np.minimum.reduceat(nonzero_vs_feat, start_idx, axis=0)

    return np.sqrt(feat_vs_feat)

#%% threshold and remove spurious components
min_size=5
max_size=np.inf
max_perc=.5
from scipy import ndimage
B_s=[]
lab_imgs=[]

cm_s=[]
for A_ in A_s:
    print '*'
    max_comps=A_.max(0).todense().T
    tmp=[]
    cm=[]
    lim=np.zeros(shape)
    for idx,a in enumerate(A_.T):        
        #create mask by thresholding to 50% of the max
        mask=np.reshape(a.todense()>(max_comps[idx]*max_perc),shape)        
        label_im, nb_labels = ndimage.label(mask)
        sizes = ndimage.sum(mask, label_im, range(nb_labels + 1))
        l_largest=(label_im==np.argmax(sizes))
        cm.append(scipy.ndimage.measurements.center_of_mass(l_largest,l_largest))
        lim[l_largest] = (idx+1)
#       #remove connected components that are too small
        mask_size=np.logical_or(sizes<min_size,sizes>max_size)
        if np.sum(mask_size[1:])>1:
            print 'removing ' + str( np.sum(mask_size[1:])-1) + ' components'
        remove_pixel=mask_size[label_im]
        label_im[remove_pixel] = 0           
        label_im=(label_im>0)*1        
        tmp.append(label_im.flatten())
    
    
    cm_s.append(cm)    
    lab_imgs.append(lim)        
    B_s.append(csc.csc_matrix(np.array(tmp)).T)

#%%
for i,A_ in enumerate(B_s):
     sizes=np.array(A_.sum(0)).squeeze()
     pl.subplot(2,3,i+1)
     pl.imshow(np.reshape(A_[:,sizes<40].sum(1),(512,512),order='F'),cmap='gray',vmax=.5)
    
#%% compute mask distances
    
    
def distance_masks(M1,M2,cm1,cm2,max_dist,dview=None):
    d_1=np.shape(M1)[-1]
    d_2=np.shape(M2)[-1]
    D = np.zeros((d_1,d_2));
    cm1=np.array(cm1)
    cm2=np.array(cm2)
    for i in range(d_1):
        print i
        k=M1[:,np.repeat(i,d_2)]+M2
#        h=M1[:,np.repeat(i,d_2)].copy()
#        h.multiply(M2)
        for j  in range(d_2): 

            dist = np.linalg.norm(cm1[i]-cm2[j])
            if dist<max_dist:
                union = k[:,j].sum()
#                intersection = h[:,j].nnz
                intersection= np.array(M1[:,i].T.dot(M2[:,j]).todense()).squeeze()
    ##            intersect= np.sum(np.logical_xor(M1[:,i],M2[:,j]))
    ##            union=np.sum(np.logical_or(M1[:,i],M2[:,j]))
                D[i,j] = 1-1.*intersection/(union-intersection)
            else:
                D[i,j] = 1
   
    return D            
#%%
#dists=feature_dist(lab_imgs[0])    
max_dist=100   
D=distance_masks(B_s[0],B_s[1],cm_s[0],cm_s[1],max_dist)       
pl.imshow(D,interpolation='None')
#%%
from sklearn.utils.linear_assignment_ import linear_assignment

DD=D.copy()
indexes = linear_assignment(DD)
#indexes=m.compute(DD)
DD=D.copy()

total = []
for row, column in indexes:
    value = DD[row,column]
    total.append(value)  
    print '(%d, %d) -> %f' % (row, column, value)
print 'total cost: %f' % np.sum(total)
#%%
pl.close()
for row, column in indexes:
    value = DD[row,column]
    if value > .1 and value < .11:
#        pl.subplot(1,2,1)
        pl.cla() 
        pl.imshow(np.reshape(B_s[0][:,row].todense(),(512,512),order='F'),cmap='gray',interpolation='None')    
#        pl.subplot(1,2,2)
#        pl.cla() 
        pl.imshow(np.reshape(B_s[1][:,column].todense(),(512,512),order='F'),alpha=.5,cmap='hot',interpolation='None')               
        if B_s[0][:,row].T.dot(B_s[1][:,column]).todense() == 0:
            print 'Flaw'
            
        pl.pause(.3)
        break
#%%
matched = ([idx for idx,t in enumerate(total) if t<.5])
indexes[matched]

#%%
from matplotlib.pyplot import Figure    
x1 = rand(103, 53) 
figure = pl.figure(figsize=(4, 4), dpi=100)
ax1 = figure.add_subplot(2,2,1)
pl.imshow(x1)
ax2 = figure.add_subplot(2,2,2)
pl.imshow(x1)
ax3 = figure.add_subplot(2,2,3)
pl.imshow(x1)
ax4 = figure.add_subplot(2,2,4)
pl.imshow(x1)
x = pl.ginput(2) 
pl.show()
print(x)

#%%
#axes = figure.add_subplot(212)
imshow(x1)
x = ginput(2) 
print(x)
show()
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