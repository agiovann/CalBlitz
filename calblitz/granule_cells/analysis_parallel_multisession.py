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
from calblitz.granule_cells.utils_granule import load_data_from_stored_results,process_eyelid_traces,process_wheel_traces,process_fast_process_day
import pandas as pd
import re
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
              
base_folders.sort()
print base_folders              
#%%
results=dview.map_sync(cb.granule_cells.utils_granule.fast_process_day,base_folders)     

#%% if this does not work look below
triggers_chunk_fluo, eyelid_chunk,wheel_chunk ,triggers_chunk_bh ,tm_behav,names_chunks,fluo_chunk,pos_examples_chunks,A_chunks=process_fast_process_day(base_folders,save_name='eyeblink_35_37_sorted.npz')
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
thresh_middle=.2
thresh_late=.8
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
mouse_now=''
session_now=''
session_id = 0
idx_sorted=names_chunks.argsort()
names_chunks=names_chunks[idx_sorted]

triggers_chunk_fluo=  triggers_chunk_fluo[idx_sorted]
triggers_chunk_bh=  triggers_chunk_bh[idx_sorted]
eyelid_chunk=  eyelid_chunk[idx_sorted] 
wheel_chunk=  wheel_chunk[idx_sorted] 
tm_behav=  tm_behav[idx_sorted]
fluo_chunk=  fluo_chunk[idx_sorted]
pos_examples_chunks=  pos_examples_chunks[idx_sorted]
A_chunks=  A_chunks[idx_sorted]
cell_counter=0
for tr_fl,tr_bh,eye,whe,tm,fl,nm,pos_examples,A in zip(triggers_chunk_fluo, triggers_chunk_bh, eyelid_chunk, wheel_chunk, tm_behav, fluo_chunk,names_chunks,pos_examples_chunks,A_chunks):
    session=nm.split('/')[8]
    day=nm.split('/')[8][:8]
    
    mouse=nm.split('/')[7]
    if mouse != mouse_now:
        mouse_now=mouse
        session_id = 0
        session_now=''
        learning_phase=0
        print 'early'
    else:
        if day != session_now:
            session_id += 1
            session_now=day
        
            
        
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
    
    trial_names=['']*wheel_traces.shape[0]
    
    for CSUSNoCR in nm_idxCSUSNOCR:
        trial_names[CSUSNoCR]='CSUSNoCR'  
    for CSUSwCR in nm_idxCSUSCR:
        trial_names[CSUSwCR]='CSUSwCR'  
    for US in nm_idx_US:
        trial_names[US]='US'
    for CSwCR in nm_idxCSCR:
        trial_names[CSwCR]='CSwCR'
    for CSNoCR in nm_idxCSNOCR:
        trial_names[CSNoCR]='CSNoCR'
        
      
    
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
    
#    use_plus_minus=1
#    if use_plus_minus:    
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
    if  len(nm_idxCR)*1./len(nm_idxCSCSUS)> thresh_middle and learning_phase==0:
        learning_phase=1
        print 'middle'
    elif len(nm_idxCR)*1./len(nm_idxCSCSUS)> thresh_late and learning_phase==1:
        learning_phase=2
        print 'late'
    ampl_CR['learning_phase']= learning_phase          
    ampl_CR['ampl_eyelid_CSCSUS']=np.mean(amplitudes_at_US[nm_idxCSCSUS])
    ampl_CR['session_id']=session_id
    cr_ampl=pd.concat([cr_ampl,ampl_CR])
#    else:
#        tmp_cr_ampl_tmp=pd.DataFrame()
#        tmp_cr_ampl_tmp_2=pd.DataFrame()
#        for counter,resp in enumerate(amplitudes_responses[:,idx_components_final]):
#            tmp_cr_ampl_tmp['fluo']=resp
#            tmp_cr_ampl_tmp['trial_counter']=cell_counter+np.arange(amplitudes_responses[:,idx_components_final].shape[-1])
#            tmp_cr_ampl_tmp['trialName']=trial_names[counter]
#            tmp_cr_ampl_tmp['trials']=np.nan
#            tmp_cr_ampl_tmp['trialsTypeOrig']=np.nan
#            tmp_cr_ampl_tmp['session']=session
#            tmp_cr_ampl_tmp['mouse']=mouse
#            tmp_cr_ampl_tmp['day']=day
#            tmp_cr_ampl_tmp['session_id']=session_id
#            tmp_cr_ampl_tmp['amplAtUs']=amplitudes_at_US[counter]
#            if any(sstr in trial_names[counter] for sstr in ['CSUSwCR','CSwCR']):        
#                tmp_cr_ampl_tmp['type_CR']=1
#            elif any(sstr in trial_names[counter] for sstr in ['US']):
#                tmp_cr_ampl_tmp['type_CR']=np.nan
#            else:
#                tmp_cr_ampl_tmp['type_CR']=0
#            tmp_cr_ampl_tmp_2 = pd.concat([tmp_cr_ampl_tmp_2,tmp_cr_ampl_tmp])
#        
#        cell_counter=tmp_cr_ampl_tmp['trial_counter'].values[-1]    
#        cr_ampl = pd.concat([cr_ampl,tmp_cr_ampl_tmp_2])
#    ampl_CR['ampl_eyelid_CSCSUS_sem']=scipy.stats.sem(amplitudes_at_US[nm_idxCSCSUS])
    
#    ampl_no_CR['session']=session;
#    ampl_no_CR['mouse']=mouse;
#    ampl_no_CR['CR']=0;
#    ampl_no_CR['idx_component']=idx_components_final;
#    ampl_no_CR['ampl_eyelid_CR']=np.mean(amplitudes_at_US[nm_idxCSCSUS])
#    ampl_no_CR['perc_CR']=len(nm_idxCR)*1./len(nm_idxCSCSUS)
#  
#%%
#bins_trials=pd.cut(cr_ampl['session_id'],[0,8,12,14],include_lowest=True)    
#grouped_session=cr_ampl.groupby(['mouse','session','type_CR'])  
#mean_plus=grouped_session.mean().loc['b35'].loc[(slice(None),[1]),:]
#mean_minus=grouped_session.mean().loc['b35'].loc[(slice(None),[0]),:]
#std_plus=grouped_session.sem().loc['b35'].loc[(slice(None),[1]),:]
#std_minus=grouped_session.sem().loc['b35'].loc[(slice(None),[0]),:]
#
#
#mean_plus['amplAtUs'].plot(kind='line',marker='o',markersize=15)
#mean_minus['amplAtUs'].plot(kind='line',marker='o',markersize=15)
#
#mean_plus['fluo'].plot(kind='line',yerr=std_plus,marker='o',markersize=15)
#mean_minus['fluo'].plot(kind='line',yerr=std_minus,marker='o',markersize=15)    
    
    
#    pl.plot(tm,np.mean(eye_traces[nm_idxCSUSCR],0))
#    pl.plot(time_fl,np.nanmedian(f_mat_bl[nm_idxCR,:,:][:,idx_components_final,:],(0,1)),'b')
#    pl.plot(time_fl,np.nanmedian(f_mat_bl[nm_idxNOCR,:,:][:,idx_components_final,:],(0,1)),'g')
#%%
mat_summaries=['/mnt/xfs1/home/agiovann/imaging/eyeblink/MAT_SUMMARIES/gc-AGGC6f-031213-03/python_out.mat',
'/mnt/xfs1/home/agiovann/imaging/eyeblink/MAT_SUMMARIES/gc-AG052014-02/python_out.mat',
'/mnt/xfs1/home/agiovann/imaging/eyeblink/MAT_SUMMARIES/AG052014-01/python_out.mat',
'/mnt/xfs1/home/agiovann/imaging/eyeblink/MAT_SUMMARIES/AG051514-01/python_out.mat']
for mat_summary in mat_summaries:
    ld=scipy.io.loadmat('/mnt/xfs1/home/agiovann/python_out.mat')
    cr_ampl_dic=dict()
    cr_ampl_dic['trials']=np.array([a[0][0][0] for a in ld['python_trials']])
    cr_ampl_dic['trialsTypeOrig']=[css[0][0] for css in ld['python_trialsTypeOrig']]
    cr_ampl_dic['trialName']=[css[0][0] for css in ld['python_trialName']]
    cr_ampl_dic['session']=[css[0][0] for css in ld['python_session']]
    cr_ampl_dic['animal']=[css[0][0] for css in ld['python_animal']]
    cr_ampl_dic['day']=[css[0][0] for css in ld['python_day']]
    cr_ampl_dic['realDay']=[css[0][0] for css in ld['python_realDay']]
    
    mat_time=np.array([css[0] for css in ld['python_time']])
    mat_wheel=np.array([css for css in ld['python_wheel']])
    mat_eyelid=np.array([css for css in ld['python_eyelid']])
    mat_ampl_at_US=np.nanmedian(mat_eyelid[np.logical_and(mat_time > -.05,mat_time < time_US_on) ,:],0)
    mat_fluo=np.concatenate([np.atleast_3d(css) for css in ld['python_fluo_traces']],-1)
    mat_idxCR=np.where([np.logical_and(t in ['CSUS','CS'],ampl>=thresh_CR) for t,ampl in zip(cr_ampl_dic['trialsTypeOrig'],mat_ampl_at_US)])[0]
    mat_idxNOCR=np.where([np.logical_and(t in ['CSUS','CS'],ampl<thresh_CR) for t,ampl in zip(cr_ampl_dic['trialsTypeOrig'],mat_ampl_at_US)])[0]
    
    
    mouse_now=''
    session_now=''
    sess_,order_,_,_=np.unique(cr_ampl_dic['session'],return_index=True, return_inverse=True, return_counts=True)
    sess_=sess_[np.argsort(order_)]
    #cr_ampl=pd.DataFrame()
    for ss in sess_:
        idx_sess=np.where([item == ss for item in cr_ampl_dic['session']])[0]
        print ss
        mouse=cr_ampl_dic['animal'][idx_sess[0]]
        session=cr_ampl_dic['day'][idx_sess[0]]    
        day=cr_ampl_dic['realDay'][idx_sess[0]]   
        if mouse != mouse_now:
            mouse_now=mouse
            session_id = 0
            session_now=''
            learning_phase=0
            print 'early'
        else:
            if day != session_now:
                session_id += 1
                session_now=day
                
        idx_CR=np.intersect1d(idx_sess,mat_idxCR)
        idx_NOCR=np.intersect1d(idx_sess,mat_idxNOCR)
        
        idx_neurons=np.arange(mat_fluo.shape[1])
        ampl_CR=pd.DataFrame()
        if len(idx_CR)>min_trials:    
            fluo_crpl=np.nanmedian(mat_fluo[idx_CR,:,:][:,:,np.logical_and(mat_time > -.05,mat_time < time_US_on)],(0,-1))
            ampl_CR['fluo_plus']=fluo_crpl
            ampl_CR['ampl_eyelid_CR']=np.mean(mat_ampl_at_US[idx_CR])
        else:        
            ampl_CR['fluo_plus']=np.nan*idx_neurons
            ampl_CR['ampl_eyelid_CR']=np.nan*idx_neurons
            
        if len(idx_NOCR)>min_trials:       
            fluo_crmn=np.nanmedian(mat_fluo[idx_NOCR,:,:][:,:,np.logical_and(mat_time > -.05,mat_time < time_US_on)],(0,-1))
            ampl_CR['fluo_minus']=fluo_crmn
        else:
            ampl_CR['fluo_minus']=np.nan*idx_neurons
            
        ampl_CR['session']=session
        ampl_CR['mouse']=mouse;
        ampl_CR['chunk']=ss    
        ampl_CR['idx_component']=idx_neurons
        ampl_CR['perc_CR']=len(idx_CR)*1./(len(idx_NOCR)+len(idx_CR))
        if  len(idx_CR)*1./(len(idx_NOCR)+len(idx_CR))> thresh_middle and learning_phase==0:
            learning_phase=1
            print 'middle'
        elif len(idx_CR)*1./(len(idx_NOCR)+len(idx_CR))> thresh_late and learning_phase==1:
            learning_phase=2
            print 'late'
        ampl_CR['learning_phase']= learning_phase          
        ampl_CR['ampl_eyelid_CSCSUS']=np.mean(mat_ampl_at_US[np.union1d(idx_NOCR,idx_CR)])
        ampl_CR['session_id']=session_id
        cr_ampl=pd.concat([cr_ampl,ampl_CR])        
    
    
#print mat_idxCR.size
#amplitudes_responses=np.nanmedian(mat_fluo[:,:,np.logical_and(mat_time > -.05,mat_time < time_US_on)],(-1))
#
#
#
#
#mat_cr_ampl=pd.DataFrame()
#mat_cr_ampl_tmp=pd.DataFrame()
#for counter,resp in enumerate(amplitudes_responses):
#    mat_cr_ampl_tmp['fluo']=resp
#    mat_cr_ampl_tmp['trial_counter']=counter
#    mat_cr_ampl_tmp['trialName']=cr_ampl_dic['trialName'][counter]
#    mat_cr_ampl_tmp['trials']=cr_ampl_dic['trials'][counter]
#    mat_cr_ampl_tmp['trialsTypeOrig']=cr_ampl_dic['trialsTypeOrig'][counter]
#    mat_cr_ampl_tmp['session']=cr_ampl_dic['session'][counter]
#    mat_cr_ampl_tmp['mouse']=cr_ampl_dic['animal'][counter]
#    mat_cr_ampl_tmp['day']=cr_ampl_dic['day'][counter]
#    mat_cr_ampl_tmp['session_id']=cr_ampl_dic['realDay'][counter]
#    mat_cr_ampl_tmp['amplAtUs']=mat_ampl_at_US[counter]
#    if any(sstr in cr_ampl_dic['trialName'][counter] for sstr in ['CSUSwCR','CSwCR']):        
#        mat_cr_ampl_tmp['type_CR']=1
#    elif any(sstr in cr_ampl_dic['trialName'][counter] for sstr in ['US']):
#        mat_cr_ampl_tmp['type_CR']=np.nan
#    else:
#        mat_cr_ampl_tmp['type_CR']=0
#    mat_cr_ampl = pd.concat([mat_cr_ampl,mat_cr_ampl_tmp])
#%%
#sess_grp=mat_cr_ampl.groupby('session')
#for nm,idxs in sess_grp.indices.iteritems():
#    print nm
#    mat_cr_ampl_tmp=mat_cr_ampl[idxs]
#    nm_idxCR=np.where(mat_cr_ampl_tmp['type_CR'])[0]    
#    print (len(nm_idxCR))
#    if len(nm_idxCR)>min_trials:
#        
#        ampl_CR['fluo_plus']=fluo_crpl
#        ampl_CR['ampl_eyelid_CR']=np.mean(amplitudes_at_US[nm_idxCR])
#    else:        
#        ampl_CR['fluo_plus']=np.nan
#        ampl_CR['ampl_eyelid_CR']=np.nan
#        
#    if len(nm_idxNOCR)>min_trials:       
#        ampl_CR['fluo_minus']=fluo_crmn
#    else:
#        ampl_CR['fluo_minus']=np.nan
#        
#    ampl_CR['session']=session;
#    ampl_CR['mouse']=mouse;
#    ampl_CR['chunk']=chunk    
#    ampl_CR['idx_component']=idx_components_final;
#    ampl_CR['perc_CR']=len(nm_idxCR)*1./len(nm_idxCSCSUS)
#    if  len(nm_idxCR)*1./len(nm_idxCSCSUS)> thresh_middle and learning_phase==0:
#        learning_phase=1
#        print 'middle'
#    elif len(nm_idxCR)*1./len(nm_idxCSCSUS)> thresh_late and learning_phase==1:
#        learning_phase=2
#        print 'late'
#    ampl_CR['learning_phase']= learning_phase          
#    ampl_CR['ampl_eyelid_CSCSUS']=np.mean(amplitudes_at_US[nm_idxCSCSUS])
#    ampl_CR['session_id']=session_id
#    cr_ampl=pd.concat([cr_ampl,ampl_CR])
#mat_eyelid=mat_eyelid-np.nanmean(mat_eyelid[mat_time < -.44,:],0)[np.newaxis,:]
#UR_size=np.median(np.nanmax(mat_eyelid[np.logical_and(mat_time > .03,mat_time < .25) ,:],0))
#mat_eyelid=mat_eyelid/UR_size
#%%
#bins_trials=pd.cut(mat_cr_ampl['trial_counter'],[0,200,600,822],include_lowest=True)    
#grouped_session=mat_cr_ampl.groupby(['mouse',bins_trials,'type_CR'])  
#mean_plus=grouped_session.mean().loc['AG051514-01'].loc[(slice(None),[1.0]),:]
#mean_minus=grouped_session.mean().loc['AG051514-01'].loc[(slice(None),[0.0]),:]
#std_plus=grouped_session.sem().loc['AG051514-01'].loc[(slice(None),[1.0]),:]
#std_minus=grouped_session.sem().loc['AG051514-01'].loc[(slice(None),[0.0]),:]
#
#
##mean_plus['amplAtUs'].plot(kind='line',marker='o',markersize=15)
##mean_minus['amplAtUs'].plot(kind='line',marker='o',markersize=15)
#
#mean_plus['fluo'].plot(kind='line',yerr=std_plus,marker='o',markersize=15)
#mean_minus['fluo'].plot(kind='line',yerr=std_minus,marker='o',markersize=15)
#%%

#%%
pl.rcParams['pdf.fonttype'] = 42
font = {'family' : 'Myriad Pro',
        'weight' : 'regular',
        'size'   : 15}
grouped_session=cr_ampl.groupby(['mouse','learning_phase'])    
grouped_session.mean().loc['b35'][['ampl_eyelid_CR','perc_CR']].plot(kind='line',subplots=True,layout=(2,1),marker='o',markersize=15,xticks=range(len(grouped_session.mean().loc['b35'])))
pl.rc('font', **font)
#pl.ylim([0,.5])
grouped_session.mean().loc['b37'][['ampl_eyelid_CR','perc_CR']].plot(kind='line',subplots=True,layout=(2,1),marker='o',markersize=15,xticks=range(len(grouped_session.mean().loc['b37'])))
#pl.ylim([0,.5])
pl.rc('font', **font)
#%%
pl.subplot(2,1,1)
grouped_session=cr_ampl.groupby(['learning_phase'])    
sems=grouped_session.sem()[['fluo_plus','fluo_minus']]
grouped_session.median()[['fluo_plus','fluo_minus']].plot(kind='line',yerr=sems,marker='o',markersize=15,xticks=range(len(grouped_session.mean())))
pl.rc('font', **font)
pl.xticks(np.arange(3),['Naive','Learning','Trained'])
pl.xlabel('Learning Phase')
pl.ylabel('DF/F')
pl.xlim([-.1 ,2.1])
#%%
pl.subplot(2,1,2)
cr_ampl_m=cr_ampl#[cr_ampl['mouse']=='b37']
bins=[0,.2, .5, 1]
grouped_session=cr_ampl_m.groupby(pd.cut(cr_ampl_m['perc_CR'],bins,include_lowest=True)) 
means=grouped_session.median()[['fluo_plus','fluo_minus']]
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
#%%
bins=[0,.1, .5, 1]
grouped_session=cr_ampl.groupby([pd.cut(cr_ampl_m['ampl_eyelid_CSCSUS'],bins,include_lowest=True)])    
means=grouped_session.mean()[['fluo_plus','fluo_minus']]
sems=grouped_session.sem()[['fluo_plus','fluo_minus']]
means.plot(kind='line',yerr=sems,marker='o',xticks=range(3),markersize=15)
#pl.xlim([-.1, 2.1])
pl.legend(['CR+','CR-'],loc=3)
pl.xlabel('Fraction of CRs')
pl.ylabel('DF/F')

pl.rc('font', **font)