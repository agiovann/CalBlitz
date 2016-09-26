# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 11:30:44 2016

@author: agiovann
"""
#%%
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
import glob
import os
import scipy
from ipyparallel import Client
import calblitz as cb
import openpyxl
#%%
excel_file='/Users/agiovann/Dropbox/2p_gcamp6fhsyn (1)/andrea/Ephys_good cells_withMaskNo.xlsx'

wb = openpyxl.load_workbook(excel_file)
sheet = wb.get_sheet_by_name(wb.get_sheet_names()[0])
vals=[]
for sh in sheet.rows:
    vls=[vl.value for vl in sh[:11]]
    vals.append(vls)
    print vls
#%%
ephys_folder='/Users/agiovann/Dropbox/2p_gcamp6fhsyn/slice2016/ephys/'    
img_folder='/Users/agiovann/Dropbox/2p_gcamp6fhsyn/slice2016/Substacks/'

params=[]
for vl in vals[1:]:
    if vl[0 is not None]:
        tmp1=['{0}'.format(kk.zfill(2)) for kk in  vl[0].split('-')]
        tmp1[-1]=tmp1[-1][-2:]
        fn=('_').join(tmp1)

        fls=glob.glob(os.path.join(img_folder,fn,vl[7]+'*.tif'))
        fls.sort(key = lambda x: int(x[-6:-4].replace('s','')))
        params=params+[vl]
    else:
        break
#%%

vl=vals[1]
file_names=[]
file_ids=[]
f_rates=[]
trials=[]
params=[]
for vl in vals[1:]:
    if vl[0 is not None]:
        tmp1=['{0}'.format(kk.zfill(2)) for kk in  vl[0].split('-')]
        tmp1[-1]=tmp1[-1][-2:]
        fn=('_').join(tmp1)
        
        fls=glob.glob(os.path.join(img_folder,fn,vl[7]+'*.tif'))
        fls.sort(key = lambda x: int(x[-6:-4].replace('s','')))
        trials=trials+[int(x[-6:-4].replace('s','')) for x in fls]
        file_names=file_names+fls
        file_ids=file_ids+[np.int(vl[3])]*len(fls)
        f_rates=f_rates+[np.float(vl[-2])]*len(fls)
        params=params+[vl]*len(fls)
        print fls
    else:
        break
#%%
file_names=np.array(file_names)
file_ids=np.array(file_ids)   
f_rates=np.array(f_rates)   
params=np.array(params)
#%%
for vl in np.unique(file_ids): 
      idx_movie = np.where(file_ids == vl)    
      print idx_movie      
      m=cb.load_movie_chain(list(file_names[idx_movie]),fr=f_rates[idx_movie[0]][0])
      m.save(os.path.join(img_folder,'MOV_'+ str(vl) + '.hdf5'))
      np.savez(os.path.join(img_folder,'MOV_'+ str(vl) + '.npz'),pars=params[idx_movie])
#%%   
templates=[]
shifts=[]
for vl in np.unique(file_ids): 
    m=cb.load(os.path.join(img_folder,'MOV_'+ str(vl) + '.hdf5'))
    mc,shift,xcorr,template=m.motion_correct(10,10,remove_blanks=True)
    templates.append(template)
    shifts.append(shift)
    mc.file_name=[]
    mc.save(os.path.join(img_folder,'MOV_'+ str(vl) + '.hdf5'))
#%%
fls=[]
for vl in np.unique(file_ids): 
    fls.append(os.path.join(img_folder,'MOV_'+ str(vl) + '.hdf5'))
    pl.subplot(2,1,1)
    pl.plot(shifts[vl])
    pl.subplot(2,1,2)
    pl.imshow(templates[vl],cmap='gray')
    pl.pause(.1)
    pl.cla()
#%%
vl=vals[1]
params=[]
for vl in vals:
    tmp1=['{0}'.format(kk.zfill(2)) for kk in  vl[0].split('-')]
    tmp1[-1]=tmp1[-1][-2:]
    fn=('_').join(tmp1)
    
    fls=glob.glob(os.path.join(img_folder,fn,vl[7]+'*.tif'))
    fls.sort(key = lambda x: int(x[-6:-4].replace('s','')))
    if len(fls)==0:
        print('NOT FOUND: ')
       
    else:    
        print(vl[3])
        params.append(vl)
        m=cb.load_movie_chain(fls,fr=vl[8])        
        print('Saving '+fn+'_'+vl[6]+'_mov.hdf5')
        m.save(fn+'_'+vl[6]+'_mov.hdf5')        
        np.savez(fn+'_'+vl[6]+'_mov.npz',pars=vl)
#        img=m.local_correlations()            
#        pl.imshow(img>.3)
#        pl.pause(1)
#        pl.cla()
#            print fl
#        print '*'

#%%
c[:].map_sync(place_holder,fls) 
#%%
map(place_holder,fls) 
#%%      
def place_holder(fl):
    import calblitz as cb
    import ca_source_extraction as cse
    import numpy as np
    m=cb.load(fl)
    Cn = m.local_correlations()
    m=m[:,]
    cnmf=cse.CNMF(1, k=4,gSig=[8,8],merge_thresh=0.8,p=2,dview=None,Ain=None)
    cnmf=cnmf.fit(m)
    A,C,b,f,YrA=cnmf.A,cnmf.C,cnmf.b,cnmf.f,cnmf.YrA    
    np.savez(fl[:-5]+'_result.npz',A=A,C=C,b=b,f=f,YrA=YrA,Cn=Cn)
    return fl[:-5]+'_result.npz'
#%%  
import glob    
fls=glob.glob(os.path.join(img_folder,'*.hdf5'))
fls.sort(key = lambda x: int(x[-7:-5].replace('_','')))
pars=[]
for fl in fls[12:]:
    with np.load(fl[:-5]+'_result.npz') as ld:
        A=ld['A'][()]
        C=ld['C']
        b=ld['b']
        f=ld['f']
        YrA=ld['YrA']
        Cn=ld['Cn']
        
        traces=C+YrA
        m=cb.load(fl)
        
        T,d1,d2=m.shape
        Y=m.transpose([1,2,0])
        with np.load(fl[:-4]+'npz') as ld:
            pars=ld['pars']
        
        if T%len(pars):

            raise Exception('Issue with the number of components!')
        
        num_trials=len(pars)    
        traces_f=[]
        traces_dff=[]
        time=range(T/len(pars))/m.fr
        for tr in traces:
            tr_tmp=np.reshape(tr,(num_trials,-1)).T
            traces_f.append(tr_tmp)
            f=np.median(tr_tmp[time<1.2,:],0)
            f=np.maximum(f,1)
            traces_dff.append((tr_tmp-f)/f)
#            traces_dff.append(np.reshape(tr,(num_trials,-1)))
        flam=lambda p: '' if p is None else p
        pars_=map(flam,[map(flam,ppp) for ppp in pars])
        pars_=np.array(pars_,dtype=object)
        masks=np.reshape(np.array(A.tocsc().todense()),[d1,d2,A.shape[-1]],order='F').transpose([2,0,1])           
        scipy.io.savemat(fl[:-5]+'_result.mat',{'traces_f':traces_f,'traces_dff':traces_dff,'masks':masks,'time_img':time,'pars':pars_})  
        print(fl[:-5]+'_result.mat')             
        if 1:
            tB = np.minimum(-2,np.floor(-5./30*m.fr))
            tA = np.maximum(5,np.ceil(25./30*m.fr))
            Npeaks=10
            #        traces_a=traces-scipy.ndimage.percentile_filter(traces,8,size=[1,np.shape(traces)[-1]/5])
            #        traces_b=np.diff(traces,axis=1)
            fitness_raw, fitness_delta, erfc_raw, erfc_delta, r_values, significant_samples\
            = cse.utilities.evaluate_components(Y, traces, A, C, b,\
            f, remove_baseline=True, N=5, robust_std=False,Athresh = 0.1, Npeaks = Npeaks, tB=tB, tA = tA, thresh_C = 0.3)
            
            idx_components_r=np.where(r_values>=.4)[0]
            idx_components_raw=np.where(fitness_raw<-20)[0]        
            idx_components_delta=np.where(fitness_delta<-10)[0]   
            
            idx_components=np.union1d(idx_components_r,idx_components_raw)
            idx_components=np.union1d(idx_components,idx_components_delta)  
            idx_components_bad=np.setdiff1d(range(len(traces)),idx_components)
            idx_components=idx_components[np.argsort(idx_components_r)]
    #        masks,pos,neg=cse.utilities.extract_binary_masks_blob(A.tocsc()[:,idx_components],5,(d1,d2))
    #        pos=np.argsort(r_values)
            pos=idx_components[0:1]
            masks=np.reshape(np.array(A.tocsc()[:,idx_components].todense()),[d1,d2,-1],order='F').transpose([2,0,1])       
            pl.close() 
            pl.subplot(2,1,1)
            crd = cse.utilities.plot_contours(A.tocsc()[:,idx_components],Cn,cmap='gray')  
            
            if len(pos)>0:
                print fl
                pl.subplot(2,1,2)   
                
                pl.imshow(masks[pos[0]])
            print(r_values)
            pl.pause(1)
#%%
           
#%%        
Cn = m.local_correlations()
pl.imshow(Cn,cmap='gray')    
#%%
crd = cse.utilities.plot_contours(A,Cn)
#%%
traces=C+YrA
idx_components, fitness, erfc, r_values,num_significant_samples = cse.utilities.evaluate_components(np.transpose(m,[1,2,0]),traces,A, N=5,robust_std=True)
#%%
crd = cse.utilities.plot_contours(A.tocsc()[:,idx_components],Cn)

