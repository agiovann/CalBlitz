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
excel_file='/mnt/xfs1/home/agiovann/dropbox/2p_GCaMP6fhsyn/slice2016/Erasmus_AB_Gao_Ephys_good cells.xlsx'

wb = openpyxl.load_workbook(excel_file)
sheet = wb.get_sheet_by_name(wb.get_sheet_names()[0])
vals=[]
for sh in sheet.rows:
    vls=[vl.value for vl in sh[:9]]
    vals.append(vls)
    print vls
#%%
ephys_folder='/mnt/ceph/users/epnevmatikakis/slice2016/ephys/'    
img_folder='/mnt/ceph/users/epnevmatikakis/slice2016/Substacks/'

vl=vals[1]
params=[]
for vl in vals:
    tmp1=['{0}'.format(kk.zfill(2)) for kk in  vl[0].split('-')]
    tmp1[-1]=tmp1[-1][-2:]
    fn=('_').join(tmp1)
    
    fls=glob.glob(os.path.join(img_folder,fn,vl[6]+'*.tif'))
    fls.sort(key = lambda x: int(x[-6:-4].replace('s','')))
    if len(fls)==0:
        print('NOT FOUND: ')
        print(vl)
    else:    
        
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
fls=glob.glob('*.hdf5') 
#%%
c[:].map_sync(place_holder,fls) 
#%%      
def place_holder(fl):
    import calblitz as cb
    import ca_source_extraction as cse
    import numpy as np
    m=cb.load(fl)
    Cn = m.local_correlations()
    cnmf=cse.CNMF(1, k=4,gSig=[8,8],merge_thresh=0.8,p=2,dview=None,Ain=None)
    cnmf=cnmf.fit(m)
    A,C,b,f,YrA=cnmf.A,cnmf.C,cnmf.b,cnmf.f,cnmf.YrA    
    np.savez(fl[:-5]+'_result.npz',A=A,C=C,b=b,f=f,YrA=YrA,Cn=Cn)
    return fl[:-5]+'_result.npz'
#%%  
import glob    
fls=glob.glob('*.hdf5')
pars=[]
for fl in fls:
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
        with np.load(fl[:-4]+'npz') as ld:
            pars=ld['pars']
        
        if T%len(m.file_name[0]):

            raise exception('Issue with the number of components!')
        
        num_trials=len(m.file_name[0])    
        traces_f=[]
        traces_dff=[]
        time=range(T/len(m.file_name[0]))/m.fr
        for tr in traces:
            tr_tmp=np.reshape(tr,(num_trials,-1)).T
            traces_f.append(tr_tmp)
            f=np.median(tr_tmp[time<1.2,:],0)
            f=np.maximum(f,1)
            traces_dff.append((tr_tmp-f)/f)
#            traces_dff.append(np.reshape(tr,(num_trials,-1)))
        pars_=map(lambda p: '' if p is None else p,pars)
        pars_=np.array(pars_,dtype=object)
        masks=np.reshape(np.array(A.tocsc().todense()),[d1,d2,A.shape[-1]],order='F').transpose([2,0,1])           
        scipy.io.savemat(fl[:-5]+'_result.mat',{'traces_f':traces_f,'traces_dff':traces_dff,'masks':masks,'time_img':time,'pars':pars_})  
        print(fl[:-5]+'_result.mat')             
        if 0:
            idx_components, fitness, erfc, r_values,num_significant_samples = cse.utilities.evaluate_components(np.transpose(m,[1,2,0]),traces,A, N=5,robust_std=True)                                      
    #        masks,pos,neg=cse.utilities.extract_binary_masks_blob(A.tocsc()[:,idx_components],5,(d1,d2))
    #        pos=np.argsort(r_values)
            pos=idx_components[0:1]
            masks=np.reshape(np.array(A.tocsc()[:,idx_components].todense()),[d1,d2,A.shape[-1]],order='F').transpose([2,0,1])       
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

