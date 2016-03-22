# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 18:33:53 2016

@author: agiovann
"""

# -*- coding: utf-8 -*-
"""
Function for implementing parallel scalable segmentation of two photon imaging data

Created on Wed Feb 17 14:58:26 2016

@author: agiovann
"""
%load_ext autoreload
%autoreload 2

from ipyparallel import Client
import numpy as np
from sklearn.decomposition import NMF
from scipy.sparse import lil_matrix,coo_matrix

import os
import ca_source_extraction as cse 
import calblitz as cb
import time
import psutil
import sys 

fnames=[]
for file in os.listdir("./"):
    if file.endswith(".tif"):
        fnames.append(file)
fnames.sort()
print fnames  
_,d1,d2=np.shape(cb.load(fnames[0][:-3]+'hdf5',subindices=range(3),fr=10))
st=time.time()
n_processes = np.maximum(psutil.cpu_count() - 2,1) # roughly number of cores on your machine minus 1
#print 'using ' + str(n_processes) + ' processes'
p=2 # order of the AR model (in general 1 or 2)
print "Stopping  cluster to avoid unnencessary use of memory...."
sys.stdout.flush()  
cse.utilities.stop_server() 
cse.utilities.start_server(n_processes)
file_name='/media/agiovann/10c9a792-5dbd-4bb9-b569-6c9d2435640e/andrea/Jeff/challenge/mov_stable/Yr.npy'
#%%
def nmf_patches(args_in):
    import numpy as np
    from sklearn.decomposition import NMF,PCA
    
    file_name, idx_,shapes,perctl,n_components,tol,max_iter=args_in
    
    Yr=np.load(file_name,mmap_mode='r')  
    
    y_m=Yr[idx_,:]    
    
    if perctl is not None:
         y_m=y_m-np.percentile(y_m,perctl,axis=-1)[:,np.newaxis]
    else:
         y_m=y_m-np.mean(y_m,axis=-1)[:,np.newaxis]
         
    nmf = NMF(n_components=n_components,tol=tol,max_iter=max_iter)
    y_m=np.maximum(y_m,0)
    flt = nmf.fit_transform(y_m)
    ca = nmf.components_
    
    return idx_,flt,ca,shapes
#%%
def NCA_nmf_patches(args_in):
    import numpy as np
    from sklearn.decomposition import NMF,PCA
    
    file_name, idx_,shapes,window_frames,n_components,tol,max_iter=args_in
    
    Yr=np.load(file_name,mmap_mode='r')  
    
    y_m=Yr[idx_,:]
    
    y_=np.reshape(y_m,shapes,order='F')    
    
        
    
    nmf = NMF(n_components=n_components,tol=tol,max_iter=max_iter)
    y_m=np.maximum(y_m,0)
    flt = nmf.fit_transform(y_m)
    ca = nmf.components_
    
    return idx_,flt,ca,shapes

 
#%%
def extract_patch_coordinates(d1,d2,rf=7,stride = 5):
    """
    Function that partition the FOV in patches and return the indexed in 2D and 1D (flatten, order='F') formats
    Parameters
    ----------    
    d1,d2: int
        dimensions of the original matrix that will be  divided in patches
    rf: int
        radius of receptive field, corresponds to half the size of the square patch        
    stride: int
        degree of overlap of the patches
    """
    coords_flat=[]
    coords_2d=[]
    for xx in range(rf,d1,2*rf-stride):   
        for yy in range(rf,d2,2*rf-stride):
            coords_x=np.array(range(xx - rf, xx + rf + 1))     
            coords_y=np.array(range(yy - rf, yy + rf + 1))  
            coords_y = coords_y[(coords_y >= 0) & (coords_y < d2)]
            coords_x = coords_x[(coords_x >= 0) & (coords_x < d1)]
            idxs = np.meshgrid( coords_x,coords_y)
            coords_2d.append(idxs)
            coords_ =np.ravel_multi_index(idxs,(d1,d2),order='F')
            coords_flat.append(coords_.flatten())
            
    return coords_flat,coords_2d
#%%
def extract_rois_patch(file_name,d1,d2,rf=5,stride = 2):
    rf=6
    stride = 2
    idx_flat,idx_2d=extract_patch_coordinates(d1, d2, rf=rf,stride = stride)
    perctl=95
    n_components=2
    tol=1e-6
    max_iter=5000
    args_in=[]    
    for id_f,id_2d in zip(idx_flat,idx_2d):        
        args_in.append((file_name, id_f,id_2d[0].shape, perctl,n_components,tol,max_iter))
    st=time.time()
    try:
        if 1:
            c = Client()   
            dview=c[:]
            file_res = dview.map_sync(nmf_patches, args_in)                         
        else:
            file_res = map(nmf_patches, args_in)                         
    finally:
        dview.results.clear()   
        c.purge_results('all')
        c.purge_everything()
        c.close()
    
    print time.time()-st
    
    A1=lil_matrix((d1*d2,len(file_res)))
    C1=[]
    A2=lil_matrix((d1*d2,len(file_res)))
    C2=[]
    A_tot=lil_matrix((d1*d2,n_components*len(file_res)))
    C_tot=[];
    count_out=0
    for count,f in enumerate(file_res):
        idx_,flt,ca,d=f
        print count_out
        #flt,ca,_=cse.order_components(coo_matrix(flt),ca)
        
#        A1[idx_,count]=flt[:,0][:,np.newaxis]/np.sqrt(np.sum(flt[:,0]**2))      
#        A2[idx_,count]=flt[:,1][:,np.newaxis] /np.sqrt(np.sum(flt[:,1]**2))              
#        C1.append(ca[0,:])
#        C2.append(ca[1,:])
        for ccc in range(n_components):
            A_tot[idx_,count_out]=flt[:,ccc][:,np.newaxis]/np.sqrt(np.sum(flt[:,ccc]**2))      
            C_tot.append(ca[ccc,:])
            count_out+=1
#        pl.imshow(np.reshape(flt[:,0],d,order='F'),vmax=10)
#        pl.pause(.1)
        
    correlations=np.corrcoef(np.array(C_tot))
    centers=cse.com(A_tot.todense(),d1,d2)
    distances=sklearn.metrics.pairwise.euclidean_distances(centers)
    pl.imshow((correlations>0.8) & (distances<10))  
    
    Yr=np.load('Yr.npy',mmap_mode='r')
    [d,T]=Yr.shape
    Y=np.reshape(Yr,(d1,d2,T),order='F')
    options=cse.utilities.CNMFSetParms(Y,p=0)    
    res_merge=cse.merge_components(Yr,A_tot,[],np.array(C_tot),[],np.array(C_tot),[],options['temporal_params'],options['spatial_params'],thr=0.8)
    A_m,C_m,nr_m,merged_ROIs,S_m,bl_m,c1_m,sn_m,g_m=res_merge
    A_norm=np.array([A_m[:,rr].toarray()/np.sqrt(np.sum(A_m[:,rr].toarray()**2)) for rr in range(A_m.shape[-1])]).T
    
    options=cse.utilities.CNMFSetParms(Y,p=2,K=np.shape(A_m)[-1])   
    
    Yr,sn,g=cse.pre_processing.preprocess_data(Yr,**options['preprocess_params'])
    
    epsilon=1e-2
    pixels_bckgrnd=np.nonzero(A_norm.sum(axis=-1)<epsilon)[0]
    f=np.sum(Yr[pixels_bckgrnd,:],axis=0)
    A2,b2,C2 = cse.spatial.update_spatial_components(Yr, C_m, f, A_m, sn=sn, **options['spatial_params'])
    A_or2, C_or2, srt2 = cse.utilities.order_components(A2,C2)
    A_norm2=np.array([A_or2[:,rr]/np.sqrt(np.sum(A_or2[:,rr]**2)) for rr in range(A_or2.shape[-1])]).T
    options['temporal_params']['p'] = 2 # set it back to original value to perform full deconvolution
    C2,f2,S2,bl2,c12,neurons_sn2,g21,YrA = cse.temporal.update_temporal_components(Yr,A2,b2,C2,f,bl=None,c1=None,sn=None,g=None,**options['temporal_params'])
    A_or, C_or, srt = cse.utilities.order_components(A2,C2)
    
    return A1,A2,C1
#%%

