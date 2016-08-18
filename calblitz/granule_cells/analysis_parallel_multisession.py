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
from glob import glob
import os
import scipy
from ipyparallel import Client
import ca_source_extraction as cse
import calblitz as cb
import sys
import numpy as np
import pickle
from calblitz.granule_cells.utils_granule import load_data_from_stored_results,process_eyelid_traces,process_wheel_traces
import pandas as pd
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

#%% if this does not work look below
triggers_chunk_fluo, eyelid_chunk,wheel_chunk ,triggers_chunk_bh ,tm_behav,names_chunks,fluo_chunk,pos_examples_chunks,A_chunks=process_fast_process_day(base_folders,save_name='eyeblink_35_37.npz')
#%%
#import re
#triggers_chunk_fluo = []  
#eyelid_chunk = []
#wheel_chunk = []
#triggers_chunk_bh = []
#tm_behav=[]
#names_chunks=[]
#fluo_chunk=[]
#pos_examples_chunks=[]
# 
#A_chunks=[]  
#for base_folder in base_folders:
#    try:         
#        print (base_folder)
#        with np.load(os.path.join(base_folder,'all_triggers.npz')) as ld:
#            triggers=ld['triggers']
#            trigger_names=ld['trigger_names']
#        
#        with np.load(glob(os.path.join(base_folder,'*-template_total.npz'))[0]) as ld:
#            movie_names=ld['movie_names']
#            template_each=ld['template_each']
#        
#        
#        idx_chunks=[] 
#        for name_chunk in movie_names:
#            idx_chunks.append([np.int(re.search('_00[0-9][0-9][0-9]_0',nm).group(0)[2:6])-1 for nm in name_chunk])
#          
#        
#        
#        with np.load(base_folder+'behavioral_traces.npz') as ld: 
#            res_bt = dict(**ld)
#            tm=res_bt['time']
#            f_rate_bh=1/np.median(np.diff(tm))
#            ISI=np.median([rs[3]-rs[2] for rs in res_bt['trial_info'][res_bt['idx_CS_US']]])
#            trig_int=np.hstack([((res_bt['trial_info'][:,2:4]-res_bt['trial_info'][:,0][:,None])*f_rate_bh),res_bt['trial_info'][:,-1][:,np.newaxis]]).astype(np.int)
#            trig_int[trig_int<0]=-1
#            trig_int=np.hstack([trig_int,len(tm)+trig_int[:,:1]*0])
#            trig_US=np.argmin(np.abs(tm))
#            trig_CS=np.argmin(np.abs(tm+ISI))
#            trig_int[res_bt['idx_CS_US'],0]=trig_CS
#            trig_int[res_bt['idx_CS_US'],1]=trig_US
#            trig_int[res_bt['idx_US'],1]=trig_US
#            trig_int[res_bt['idx_CS'],0]=trig_CS
#            eye_traces=np.array(res_bt['eyelid']) 
#            wheel_traces=np.array(res_bt['wheel'])
#    
#            
#         
#         
#        fls=glob(os.path.join(base_folder,'*.results_analysis_traces.pk'))
#        fls.sort()
#        fls_m=glob(os.path.join(base_folder,'*.results_analysis_masks.npz'))
#        fls_m.sort()     
#         
#        
#        for indxs,name_chunk,fl,fl_m in zip(idx_chunks,movie_names,fls,fls_m):
#            if np.all([nmc[:-4] for nmc in name_chunk] == trigger_names[indxs]):
#                triggers_chunk_fluo.append(triggers[indxs,:])
#                eyelid_chunk.append(eye_traces[indxs,:])
#                wheel_chunk.append(wheel_traces[indxs,:])
#                triggers_chunk_bh.append(trig_int[indxs,:])
#                tm_behav.append(tm)
#                names_chunks.append(fl)
#                with open(fl,'r') as f: 
#                    tr_dict=pickle.load(f)   
#                    print(fl)
#                    fluo_chunk.append(tr_dict['traces_DFF'])
#                with np.load(fl_m) as ld:
#                    A_chunks.append(scipy.sparse.coo_matrix(ld['A']))
#                    pos_examples_chunks.append(ld['pos_examples'])                
#            else:
#                raise Exception('Names of triggers not matching!')
#    except:
#        print("ERROR in:"+base_folder)                
     
#%%
with np.load('eyeblink_35_37.npz')  as ld:
          locals().update(ld)     
       
#%%
time_CR_on=-.1
time_US_on=.05
thresh_MOV_iqr=100
time_CS_on_MOV=-.25
time_US_on_MOV=0
thresh_CR = 0.1,
threshold_responsiveness=0.1
time_bef=2.7
time_aft=4.5
f_rate_fluo=1/30.0
ISI=.25
min_trials=4
cr_ampl=pd.DataFrame()
for tr_fl,tr_bh,eye,whe,tm,fl,nm,pos_examples,A in zip(triggers_chunk_fluo, triggers_chunk_bh, eyelid_chunk, wheel_chunk, tm_behav, fluo_chunk,names_chunks,pos_examples_chunks,A_chunks):
    mouse=nm.split('/')[7]
    session=nm.split('/')[8]
    chunk=re.search('_00[0-9][0-9][0-9]_',nm.split('/')[9]).group(0)[3:-1]
    
    
    idx_CS_US=np.where(tr_bh[:,-2]==2)[0]
    idx_US=np.where(tr_bh[:,-2]==1)[0]
    idx_CS=np.where(tr_bh[:,-2]==0)[0]
    idx_ALL=np.sort(np.hstack([idx_CS_US,idx_US,idx_CS]))
    eye_traces,amplitudes_at_US, trig_CRs=process_eyelid_traces(eye,tm,idx_CS_US,idx_US,idx_CS,thresh_CR=thresh_CR,time_CR_on=time_CR_on,time_US_on=time_US_on)        
    idxCSUSCR = trig_CRs['idxCSUSCR']
    idxCSUSNOCR = trig_CRs['idxCSUSNOCR']
    idxCSCR = trig_CRs['idxCSCR']
    idxCSNOCR = trig_CRs['idxCSNOCR']
    idxNOCR = trig_CRs['idxNOCR']
    idxCR = trig_CRs['idxCR']
    idxUS = trig_CRs['idxUS']
    idxCSCSUS=np.concatenate([idx_CS,idx_CS_US]) 
    
    
    wheel_traces, movement_at_CS, trigs_mov = process_wheel_traces(np.array(whe),tm,thresh_MOV_iqr=thresh_MOV_iqr,time_CS_on=time_CS_on_MOV,time_US_on=time_US_on_MOV)    
    print 'fraction with movement:'    + str(len(trigs_mov['idxMOV'])*1./len(trigs_mov['idxNO_MOV']))
    
    mn_idx_CS_US =np.intersect1d(idx_CS_US,trigs_mov['idxNO_MOV'])
    nm_idx_US= np.intersect1d(idx_US,trigs_mov['idxNO_MOV'])
    nm_idx_CS= np.intersect1d(idx_CS,trigs_mov['idxNO_MOV'])
    nm_idxCSUSCR = np.intersect1d(idxCSUSCR,trigs_mov['idxNO_MOV'])
    nm_idxCSUSNOCR = np.intersect1d(idxCSUSNOCR,trigs_mov['idxNO_MOV'])
    nm_idxCSCR = np.intersect1d(idxCSCR,trigs_mov['idxNO_MOV'])
    nm_idxCSNOCR = np.intersect1d(idxCSNOCR,trigs_mov['idxNO_MOV'])
    nm_idxNOCR = np.intersect1d(idxNOCR,trigs_mov['idxNO_MOV'])
    nm_idxCR = np.intersect1d(idxCR,trigs_mov['idxNO_MOV'])
    nm_idxUS = np.intersect1d(idxUS,trigs_mov['idxNO_MOV'])
    nm_idxCSCSUS = np.intersect1d(idxCSCSUS,trigs_mov['idxNO_MOV'])  
    
    len_min=np.min([np.array(f).shape for f in fl])
#    f_flat=np.concatenate([f[:,:len_min] for f in fl],1)
#    f_mat=np.concatenate([f[:,:len_min][np.newaxis,:] for f in fl],0)
    selct = lambda cs,us: np.int(cs) if np.isnan(us) else np.int(us)
    trigs_US=[selct(cs,us) for cs,us in zip(tr_fl[:,0],tr_fl[:,1])]     
    
    samplbef=np.int(time_bef/f_rate_fluo)
    samplaft=np.int(time_aft/f_rate_fluo)
    f_flat=np.concatenate([f[:,tr - samplbef:samplaft+tr] for tr,f in zip(trigs_US,fl)],1)
    f_mat=np.concatenate([f[:,tr -samplbef:samplaft+tr][np.newaxis,:] for tr,f in zip(trigs_US,fl)],0)
    time_fl=np.arange(-samplbef,samplaft)*f_rate_fluo
    
    f_mat_bl=f_mat-np.median(f_mat[:,:,np.logical_and(time_fl>-1,time_fl<-ISI)],axis=(2))[:,:,np.newaxis]   
    amplitudes_responses=np.mean(f_mat_bl[:,:,np.logical_and(time_fl>-.03,time_fl<.04)],-1)
    cell_responsiveness=np.median(amplitudes_responses[nm_idxCSCSUS],axis=0)
    idx_responsive = np.where(cell_responsiveness>threshold_responsiveness)[0]
    fraction_responsive=len(np.where(cell_responsiveness>threshold_responsiveness)[0])*1./np.shape(f_mat_bl)[1]
#    a=pd.DataFrame(data=f_mat[0,idx_components[:10],:],columns=np.arange(-30,30)*.033,index=idx_components[:10])
    
    if 0:
        idx_components, fitness, erfc = cse.utilities.evaluate_components(f_flat,N=5,robust_std=True)
        print len(idx_components[fitness<-25])*1./len(idx_components)    
        idx_components_final=np.intersect1d(idx_components[fitness<-25],idx_responsive)
        idx_components_final=np.intersect1d(idx_components_final,pos_examples)
        print len(idx_components_final)*1./len(idx_components)  
    else:
        idx_components_final=np.intersect1d(idx_responsive,pos_examples)
        
    fluo_crpl=np.nanmedian(amplitudes_responses[idxCR,:][:,idx_components_final],0)
    fluo_crmn=np.nanmedian(amplitudes_responses[idxNOCR,:][:,idx_components_final],0)
        
    ampl_CR=pd.DataFrame()
#    ampl_no_CR=pd.DataFrame(np.median(amplitudes_responses[idxNOCR,:][:,idx_components_final],0))
    if len(nm_idxCR)>min_trials:
        
        ampl_CR['fluo_plus']=fluo_crpl
        ampl_CR['ampl_eyelid_CR']=np.mean(amplitudes_at_US[nm_idxCR])
    else:        
        ampl_CR['fluo_plus']=np.nan
        ampl_CR['ampl_eyelid_CR']=np.nan
        
    if len(nm_idxNOCR)>min_trials:       
        ampl_CR['fluo_minus']=fluo_crmn
    else:
        ampl_CR['fluo_minus']=np.nan
        
    ampl_CR['session']=session;
    ampl_CR['mouse']=mouse;
    ampl_CR['chunk']=chunk    
    ampl_CR['idx_component']=idx_components_final;
    ampl_CR['perc_CR']=len(nm_idxCR)*1./len(nm_idxCSCSUS)
    ampl_CR['ampl_eyelid_CSCSUS']=np.mean(amplitudes_at_US[nm_idxCSCSUS])
#    ampl_CR['ampl_eyelid_CSCSUS_sem']=scipy.stats.sem(amplitudes_at_US[nm_idxCSCSUS])
    
#    ampl_no_CR['session']=session;
#    ampl_no_CR['mouse']=mouse;
#    ampl_no_CR['CR']=0;
#    ampl_no_CR['idx_component']=idx_components_final;
#    ampl_no_CR['ampl_eyelid_CR']=np.mean(amplitudes_at_US[nm_idxCSCSUS])
#    ampl_no_CR['perc_CR']=len(nm_idxCR)*1./len(nm_idxCSCSUS)
#    
    cr_ampl=pd.concat([cr_ampl,ampl_CR])
    
    
#    pl.plot(tm,np.mean(eye_traces[nm_idxCSUSCR],0))
#    pl.plot(time_fl,np.nanmedian(f_mat_bl[nm_idxCR,:,:][:,idx_components_final,:],(0,1)),'b')
#    pl.plot(time_fl,np.nanmedian(f_mat_bl[nm_idxNOCR,:,:][:,idx_components_final,:],(0,1)),'g')
#%%
pl.rcParams['pdf.fonttype'] = 42
font = {'family' : 'Myriad Pro',
        'weight' : 'regular',
        'size'   : 15}
grouped_session=cr_ampl.groupby(['mouse','session'])    
grouped_session.mean().loc['b35'][['ampl_eyelid_CR','perc_CR']].plot(kind='line',subplots=True,layout=(2,1),marker='o',markersize=15,xticks=range(len(grouped_session.mean().loc['b35'])))
pl.rc('font', **font)
#pl.ylim([0,.5])
grouped_session.mean().loc['b37'][['ampl_eyelid_CR','perc_CR']].plot(kind='line',subplots=True,layout=(2,1),marker='o',markersize=15,xticks=range(len(grouped_session.mean().loc['b37'])))
#pl.ylim([0,.5])
pl.rc('font', **font)
#%%

cr_ampl_m=cr_ampl#[cr_ampl['mouse']=='b37']
bins=[0,.3, .5, 1]
grouped_session=cr_ampl_m.groupby(pd.cut(cr_ampl_m['perc_CR'],bins,include_lowest=True)) 
means=grouped_session.mean()[['fluo_plus','fluo_minus']]
sems=grouped_session.sem()[['fluo_plus','fluo_minus']]
means.plot(kind='line',yerr=sems,marker='o',xticks=range(3),markersize=15)
pl.xlim([-.1, 2.1])
pl.legend(['CR+','CR-'],loc=3)
pl.xlabel('Fraction of CRs')
pl.ylabel('DF/F')

pl.rc('font', **font)
#%%
grouped_session=cr_ampl.groupby(['session','chunk'])    
print grouped_session.count().mean().idx_component, grouped_session.count().std().idx_component