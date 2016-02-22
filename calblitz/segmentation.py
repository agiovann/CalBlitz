# -*- coding: utf-8 -*-
"""
Function for implementing parallel scalable segmentation of two photon imaging data

Created on Wed Feb 17 14:58:26 2016

@author: agiovann
"""
from ipyparallel import Client
import numpy as np
from sklearn.decomposition import NMF
from scipy.sparse import lil_matrix,coo_matrix
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
def cnmf_patches(args_in):
    file_name, idx_,shapes,perctl,n_components,tol,max_iter=args_in
    p=2
    
    Yr=np.load(file_name,mmap_mode='r') 
    d,T=Yr.shape      
    Y=np.reshape(Yr,(d1,d2,T),order='F')
     
    options = cse.utilities.CNMFSetParms(Y,p=p,gSig=[7,7],K=n_components)
    Yr,sn,g=cse.pre_processing.preprocess_data(Yr,**options['preprocess_params'])
    Atmp, Ctmp, b_in, f_in, center=cse.initialization.initialize_components(Y, **options['init_params'])                                                    

    A,b,Cin = cse.spatial.update_spatial_components(Yr, Cin, f_in, Ain, sn=sn, **options['spatial_params'])
  
    options['temporal_params']['p'] = 0 # set this to zero for fast updating without deconvolution
    options['temporal_params']['fudge_factor'] = 0.96 
    C,f,S,bl,c1,neurons_sn,g,YrA = cse.temporal.update_temporal_components(Yr,A,b,Cin,f_in,bl=None,c1=None,sn=None,g=None,**options['temporal_params'])
    
 
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
def extract_rois_patch(file_name,d1,d2,rf=5,stride = 5):
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
    for count,f in enumerate(file_res):
        idx_,flt,ca,d=f
        #flt,ca,_=cse.order_components(coo_matrix(flt),ca)
        A1[idx_,count]=flt[:,0][:,np.newaxis]        
        A2[idx_,count]=flt[:,1][:,np.newaxis]        
        C1.append(ca[0,:])
        C2.append(ca[1,:])
#        pl.imshow(np.reshape(flt[:,0],d,order='F'),vmax=10)
#        pl.pause(.1)
        
        
    return A1,A2,C1,C2
#%%
    #%%
import os
import ca_source_extraction as cse 
import calblitz as cb
import time

fnames=[]
for file in os.listdir("./"):
    if file.endswith(".tif"):
        fnames.append(file)
fnames.sort()
print fnames  
_,d1,d2=np.shape(cb.load(fnames[0][:-3]+'hdf5',subindices=range(3),fr=10))
st=time.time()