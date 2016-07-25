# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 17:56:14 2016

@author: agiovann
"""
import cv2
import h5py
import numpy as np
import pylab as pl
import glob
import ca_source_extraction as cse
import calblitz as cb
from scipy import signal
import scipy
import sys
from ipyparallel import Client
from time import time
from scipy.sparse import csc,csr,coo_matrix
from scipy.spatial.distance import cdist
from scipy import ndimage
from scipy.optimize import linear_sum_assignment   

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
def get_behavior_traces(fname,t0,t1,freq,ISI,draw_rois=False,plot_traces=False,mov_filt_1d=True,window_hp=201,window_lp=3):
    """
    From hdf5 movies extract eyelid closure and wheel movement
    
    
    Parameters
    ----------
    fname: str    
        file name of the hdf5 file
        
    t0,t1: float. 
        Times of beginning and end of trials (in general 0 and 8 for our dataset) to build the absolute time vector
    
    freq: float
        frequency used to build the final time vector    
        
    ISI: float
        inter stimulu interval
        
    draw_rois: bool
        whether to manually draw the eyelid contour

    plot_traces: bool
        whether to plot the traces during extraction        
       
    mov_filt_1d: bool 
        whether to filter the movie after extracting the average or ROIs. The alternative is a 3D filter that can be very computationally expensive
    
    window_lp, window_hp: ints
        number of frames to be used to median filter the data. It is needed because of the light IR artifact coming out of the eye
        
    Returns
    -------
    res: dict
        dictionary with fields 
            'eyelid': eyelid trace
            'wheel': wheel trace
            'time': absolute tim vector
            'trials': corresponding indexes of the trials
            'trial_info': for each trial it returns start trial, end trial, time CS, time US, trial type  (CS:0 US:1 CS+US:2)
            'idx_CS_US': idx trial CS US
            'idx_US': idx trial US
            'idx_CS': idx trial CS 
    """
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
    eye_traces=[]
    wheel_traces=[]
    trial_info=[]
    tims=[]
    with h5py.File(fname) as f:

        with h5py.File(meta_inf) as dt:
            
            rois=np.asarray(dt['roi'],np.float32)

            trials = f.keys()

            trials.sort(key=lambda(x): np.int(x.replace('trial_','')))

            trials_idx=[np.int(x.replace('trial_',''))-1 for x in trials]
            
            trials_idx_=[]
            
        
            
            for tr,idx_tr in zip(trials[:],trials_idx[:]):
                if plot_traces:
                    pl.cla()

                print tr
               
               
                
                trial=f[tr]  

                mov=np.asarray(trial['mov'])        

                if draw_rois:
                    
                    pl.imshow(np.mean(mov,0))
                    pl.xlabel('Draw eye')
                    pts=pl.ginput(-1)

                    pts = np.asarray(pts, dtype=np.int32)

                    data = np.zeros(np.shape(mov)[1:], dtype=np.int32)
            #        if CV_VERSION == 2:
                    #lt = cv2.CV_AA
            #        elif CV_VERSION == 3:
                    lt = cv2.LINE_AA
                    
                    cv2.fillConvexPoly(data, pts, (1,1,1), lineType=lt)

                    rois[0]=data
                    
                    pl.close()
                    
                    pl.imshow(np.mean(mov,0))
                    pl.xlabel('Draw wheel')            
                    pts=pl.ginput(-1)

                    pts = np.asarray(pts, dtype=np.int32)

                    data = np.zeros(np.shape(mov)[1:], dtype=np.int32)
            #        if CV_VERSION == 2:
                    #lt = cv2.CV_AA
            #        elif CV_VERSION == 3:
                    lt = cv2.LINE_AA
                    
                    cv2.fillConvexPoly(data, pts, (1,1,1), lineType=lt)

                    rois[1]=data
                    
                    pl.close()
    #            eye_trace=np.mean(mov*rois[0],axis=(1,2))
    #            mov_trace=np.mean((np.diff(np.asarray(mov,dtype=np.float32),axis=0)**2)*rois[1],axis=(1,2))
                mov=np.transpose(mov,[0,2,1])
                
                mov=mov[:,:,::-1]

                if  mov.shape[0]>0:

                    ts=np.array(trial['ts'])

                    if np.size(ts)>0:

                        new_ts=np.linspace(0,ts[-1,0]-ts[0,0],np.shape(mov)[0])
                        
                        if dt['trials'][idx_tr,-1] == US_ALONE:
                            t_us=np.maximum(t_us,dt['trials'][idx_tr,3]-dt['trials'][idx_tr,0])  
                            mmm=mov[:n_samples_ISI].copy()
                            mov=mov[:-n_samples_ISI]
                            mov=np.concatenate([mmm,mov])
                            
                        elif dt['trials'][idx_tr,-1] == CS_US: 
                            t_cs=np.maximum(t_cs,dt['trials'][idx_tr,2]-dt['trials'][idx_tr,0])                                
                            t_us=np.maximum(t_us,dt['trials'][idx_tr,3]-dt['trials'][idx_tr,0])   
                            t_uss.append(t_us)                                                     
                            ISI=t_us-t_cs
                            ISIs.append(ISI)
                            n_samples_ISI=np.int(ISI*freq)
                        else:
                            t_cs=np.maximum(t_cs,dt['trials'][idx_tr,2]-dt['trials'][idx_tr,0])                                
                        
                        
                        new_ts=new_ts
                        tims.append(new_ts)
                        
                    mov_e=cb.movie(mov*rois[0][::-1].T,fr=1/np.mean(np.diff(new_ts)))
                    mov_w=cb.movie(mov*rois[1][::-1].T,fr=1/np.mean(np.diff(new_ts)))
                    
                    x_max_w,y_max_w=np.max(np.nonzero(np.max(mov_w,0)),1)
                    x_min_w,y_min_w=np.min(np.nonzero(np.max(mov_w,0)),1)
                    
                    x_max_e,y_max_e=np.max(np.nonzero(np.max(mov_e,0)),1)
                    x_min_e,y_min_e=np.min(np.nonzero(np.max(mov_e,0)),1)


                    mov_e=mov_e[:,x_min_e:x_max_e,y_min_e:y_max_e] 
                    mov_w=mov_w[:,x_min_w:x_max_w,y_min_w:y_max_w] 
                         
                         
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
                    if mov_filt_1d:
                        mov_e=np.mean(mov_e, axis=(1,2))
                        window_hp_=window_hp
                        window_lp_=window_lp
                        if plot_traces:
                            pl.plot((mov_e-np.mean(mov_e))/(np.max(mov_e)-np.min(mov_e)))
                    
                    else: 

                        window_hp_=(window_hp,1,1)
                        window_lp_=(window_lp,1,1)
                        
                    
                    
                    bl=signal.medfilt(mov_e,window_hp_)
                    mov_e=signal.medfilt(mov_e-bl,window_lp_)
    
                        
                    if mov_filt_1d:
                        
                        eye_=np.atleast_2d(mov_e)

                    else:

                        eye_=np.atleast_2d(np.mean(mov_e, axis=(1,2)))
                        

                    
                    wheel_=np.concatenate([np.atleast_1d(0),np.nanmean(np.diff(mov_w,axis=0)**2,axis=(1,2))])                   

                    if np.abs(new_ts[-1]  - time_abs[-1])>.5:

                       raise Exception('Time duration is significantly larger or smaller than reference time')
                      
                      
                    wheel_=np.squeeze(wheel_)
                    eye_=np.squeeze(eye_)
                    
                    f1=scipy.interpolate.interp1d(new_ts , eye_,bounds_error=False,kind='linear')                    
                    eye_=np.array(f1(time_abs))
                    
                    f1=scipy.interpolate.interp1d(new_ts , wheel_,bounds_error=False,kind='linear')                    
                    wheel_=np.array(f1(time_abs))

                    if plot_traces:
                        pl.plot( (eye_) / (np.nanmax(eye_)-np.nanmin(eye_)),'r')
                        pl.plot( (wheel_ -np.nanmin(wheel_))/ np.nanmax(wheel_),'k')
                        pl.pause(.01)
                                            
                    trials_idx_.append(idx_tr)

                    eye_traces.append(eye_)
                    wheel_traces.append(wheel_)                        
                    
                    trial_info.append(dt['trials'][idx_tr,:])   
                   
            
            res=dict()
            
            
            res['eyelid'] =  eye_traces   
            res['wheel'] = wheel_traces
            res['time'] = time_abs - np.median(t_uss) 
            res['trials'] = trials_idx_
            res['trial_info'] = trial_info
            res['idx_CS_US'] = np.where(map(int,np.array(trial_info)[:,-1]==CS_US))[0]           
            res['idx_US'] = np.where(map(int,np.array(trial_info)[:,-1]==US_ALONE))[0]
            res['idx_CS'] = np.where(map(int,np.array(trial_info)[:,-1]==CS_ALONE))[0]
                        

            return res

#%%
def process_eyelid_traces(traces,time_vect,idx_CS_US,idx_US,idx_CS,thresh_CR=.1,time_CR_on=-.1,time_US_on=.05):
    """ 
    preprocess traces output of get_behavior_traces 
    
    Parameters:
    ----------
    
    traces: ndarray (N trials X t time points)
        eyelid traces output of get_behavior_traces. 
    
    thresh_CR: float
        fraction of eyelid closure considered a CR

    time_CR_on: float
        time of alleged beginning of CRs
    
    time_US_on: float
        time when US is considered to induce have a UR
        
        
    Returns:
    -------
    eye_traces: ndarray 
        normalized eyelid traces
        
    trigs: dict
        dictionary containing various subdivision of the triggers according to behavioral responses
        
        'idxCSUSCR': index of trials with  CS+US with CR
        'idxCSUSNOCR': index of trials with  CS+US without CR
        'idxCSCR':   
        'idxCSNOCR':
        'idxNOCR': index of trials with no CRs
        'idxCR': index of trials with CRs
        'idxUS':
            
    """
    #normalize by max amplitudes at US    

    eye_traces=traces/np.nanmax(np.nanmedian(traces[np.hstack([idx_CS_US,idx_US])][:,np.logical_and(time_vect>time_US_on,time_vect<time_US_on +.4 )],0))

    amplitudes_at_US=np.mean(eye_traces[:,np.logical_and( time_vect > time_CR_on , time_vect <= time_US_on )],1)
    
    trigs=dict()
    
    trigs['idxCSUSCR']=idx_CS_US[np.where(amplitudes_at_US[idx_CS_US]>thresh_CR)[-1]]
    trigs['idxCSUSNOCR']=idx_CS_US[np.where(amplitudes_at_US[idx_CS_US]<thresh_CR)[-1]]
    trigs['idxCSCR']=idx_CS[np.where(amplitudes_at_US[idx_CS]>thresh_CR)[-1]]
    trigs['idxCSNOCR']=idx_CS[np.where(amplitudes_at_US[idx_CS]<thresh_CR)[-1]]
    trigs['idxNOCR']=np.union1d(trigs['idxCSUSNOCR'],trigs['idxCSNOCR'])
    trigs['idxCR']=np.union1d(trigs['idxCSUSCR'],trigs['idxCSCR'])
    trigs['idxUS']=idx_US
    
    return eye_traces,amplitudes_at_US, trigs
    
    
#%%
def process_wheel_traces(traces,time_vect,thresh_MOV_iqr=3,time_CS_on=-.25,time_US_on=0):

    tmp = traces[:,time_vect<time_CS_on]
    wheel_traces=traces/(np.percentile(tmp,75)-np.percentile(tmp,25))

    movement_at_CS=np.max(wheel_traces[:,np.logical_and( time_vect > time_CS_on, time_vect <= time_US_on )],1)
    
    trigs=dict()
    
    trigs['idxMOV']=np.where(movement_at_CS>thresh_MOV_iqr)[-1]
    trigs['idxNO_MOV']=np.where(movement_at_CS<thresh_MOV_iqr)[-1]
    
    return wheel_traces, movement_at_CS, trigs
#%%
def load_results(f_results):
    """
    Load results from CNMF on various FOVs and merge them after some preprocessing
    
    """
    # load data
    i=0
    A_s=[]
    C_s=[]
    YrA_s=[]
    Cn_s=[]
    shape = None
    for f_res in f_results:
        print f_res
        i+=1
        with  np.load(f_res) as ld:
            A_s.append(csc.csc_matrix(ld['A2']))
            C_s.append(ld['C2'])
            YrA_s.append(ld['YrA'])
            Cn_s.append(ld['Cn'])
            if shape is not None:
                shape_new=(ld['d1'],ld['d2'])
                if shape_new != shape:
                    raise Exception('Shapes of FOVs not matching')
                else:
                    shape = shape_new
            else:            
                shape=(ld['d1'],ld['d2'])
                
    return A_s,C_s,YrA_s, Cn_s, shape  

#%% threshold and remove spurious components    
def threshold_components(A_s,shape,min_size=5,max_size=np.inf,max_perc=.5):        
    """
    Threshold components output of a CNMF algorithm (A matrices)
    
    Parameters:
    ----------
    
    A_s: list 
        list of A matrice output from CNMF
    
    min_size: int
        min size of the component in pixels

    max_size: int
        max size of the component in pixels
        
    max_perc: float        
        fraction of the maximum of each component used to threshold 
        
        
    Returns:
    -------        
    
    B_s: list of the thresholded components
    
    lab_imgs: image representing the components in ndimage format

    cm_s: center of masses of each components
    """
    
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
    
    return B_s, lab_imgs, cm_s           

#%% compute mask distances
def distance_masks(M_s,cm_s,max_dist):
    """
    Compute distance matrix based on an intersection over union metric. Matrix are compared in order, with matrix i compared with matrix i+1
    
    Parameters
    ----------
    M_s: list of ndarrays
        The thresholded A matrices (masks) to compare, output of threshold_components
    
    cm_s: list of list of 2-ples
        the centroids of the components in each M_s
    
    max_dist: float
        maximum distance among centroids allowed between components. This corresponds to a distance at which two components are surely disjoined
    
    
    
    Returns:
    --------
    D_s: list of matrix distances
    """
    D_s=[]

    for M1,M2,cm1,cm2 in zip(M_s[:-1],M_s[1:],cm_s[:-1],cm_s[1:]):
        print 'New Pair **'
        M1= M1.copy()[:,:]
        M2= M2.copy()[:,:]
        d_1=np.shape(M1)[-1]
        d_2=np.shape(M2)[-1]
        D = np.ones((d_1,d_2));
        
        cm1=np.array(cm1)
        cm2=np.array(cm2)
        for i in range(d_1):
            if i%100==0:
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
                    if union  > 0:
                        D[i,j] = 1-1.*intersection/(union-intersection)
                    else:
                        D[i,j] = 1
                        
                    if np.isnan(D[i,j]):
                        raise Exception('Nan value produced. Error in inputs')
                else:
                    D[i,j] = 1
        
        D_s.append(D)            
    return D_s   

#%% find matches
def find_matches(D_s, print_assignment=False):
    
    matches=[]
    costs=[]
    t_start=time()
    for ii,D in enumerate(D_s):
        DD=D.copy()    
        if np.sum(np.where(np.isnan(DD)))>0:
            raise Exception('Distance Matrix contains NaN, not allowed!')
        
       
    #    indexes = m.compute(DD)
    #    indexes = linear_assignment(DD)
        indexes = linear_sum_assignment(DD)
        indexes2=[(ind1,ind2) for ind1,ind2 in zip(indexes[0],indexes[1])]
        matches.append(indexes)
        DD=D.copy()   
        total = []
        for row, column in indexes2:
            value = DD[row,column]
            if print_assignment:
                print '(%d, %d) -> %f' % (row, column, value)
            total.append(value)      
        print  'FOV: %d, shape: %d,%d total cost: %f' % (ii, DD.shape[0],DD.shape[1], np.sum(total))
        print time()-t_start
    costs.append(total)        
    return matches,costs
      
