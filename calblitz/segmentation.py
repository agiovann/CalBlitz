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
    for xx in range(rf,d1+stride,2*rf-stride):   
        for yy in range(rf,d2+stride,2*rf-stride):
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
    print len(idx_flat)
    try:
        if 1:
            c = Client()   
            dview=c[:10]
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
def cnmf_patches(args_in):
    import numpy as np
    import ca_source_extraction as cse
    
#    file_name, idx_,shapes,p,gSig,K,fudge_fact=args_in
    file_name, idx_,shapes,options=args_in
    
    p=options['temporal_params']['p']
    
    Yr=np.load(file_name,mmap_mode='r')
    Yr=Yr[idx_,:]
    d,T=Yr.shape      
    Y=np.reshape(Yr,(shapes[0],shapes[1],T),order='F')
     
#    options = cse.utilities.CNMFSetParms(Y,p=p,gSig=gSig,K=K)
    options['spatial_params']['d1']=shapes[0]
    options['spatial_params']['d2']=shapes[1]
    options['spatial_params']['backend']='single_thread'
    options['temporal_params']['backend']='single_thread'
    Yr,sn,g=cse.pre_processing.preprocess_data(Yr,**options['preprocess_params'])
    Ain, Cin, b_in, f_in, center=cse.initialization.initialize_components(Y, **options['init_params'])                                                    
    print options
    A,b,Cin = cse.spatial.update_spatial_components(Yr, Cin, f_in, Ain, sn=sn, **options['spatial_params'])  
    options['temporal_params']['p'] = 0 # set this to zero for fast updating without deconvolution
    
    C,f,S,bl,c1,neurons_sn,g,YrA = cse.temporal.update_temporal_components(Yr,A,b,Cin,f_in,bl=None,c1=None,sn=None,g=None,**options['temporal_params'])
    
    A_m,C_m,nr_m,merged_ROIs,S_m,bl_m,c1_m,sn_m,g_m=cse.merging.merge_components(Yr,A,b,C,f,S,sn,options['temporal_params'], options['spatial_params'], bl=bl, c1=c1, sn=neurons_sn, g=g, thr=0.8, fast_merge = True)
    
    A2,b2,C2 = cse.spatial.update_spatial_components(Yr, C_m, f, A_m, sn=sn, **options['spatial_params'])
    options['temporal_params']['p'] = p # set it back to original value to perform full deconvolution
    C2,f2,S2,bl2,c12,neurons_sn2,g21,YrA = cse.temporal.update_temporal_components(Yr,A2,b2,C2,f,bl=None,c1=None,sn=None,g=None,**options['temporal_params'])

    return idx_,shapes,A2,b2,C2,f2,S2,bl2,c12,neurons_sn2,g21,sn,options

#%%
def run_CNMF_patches(file_name,options,rf=16,stride = 4):
    rf=16
    stride = 4
    K=5
    Yr=np.load(file_name,mmap_mode='r')
    d,T=np.shape(Yr)
    Y=np.reshape(Yr,(d1,d2,T),order='F')
    p=2

    options = cse.utilities.CNMFSetParms(Y,p=p,gSig=[7,7],K=K)    
    
    options['temporal_params']['fudge_factor'] = 0.96 
    options['preprocess_params']['n_pixels_per_process']=np.int((rf*rf*4)/n_processes/(T/2000.))
    options['spatial_params']['n_pixels_per_process']=np.int((rf*rf*4)/n_processes/(T/2000.))
    options['temporal_params']['n_pixels_per_process']=np.int((rf*rf*4)/n_processes/(T/2000.))
    idx_flat,idx_2d=extract_patch_coordinates(d1, d2, rf=rf, stride = stride)
    
    args_in=[]    
    for id_f,id_2d in zip(idx_flat[:],idx_2d[:]):        
        args_in.append((file_name, id_f,id_2d[0].shape, options))

    print len(idx_flat)

    st=time.time()        
    try:
        if 1:
            c = Client()   
            dview=c[:]
            file_res = dview.map_sync(cnmf_patches, args_in)                         
        else:
            file_res = map(cnmf_patches, args_in)                         
    finally:
        dview.results.clear()   
        c.purge_results('all')
        c.purge_everything()
        c.close()    
    print time.time()-st

    
    cse.utilities.stop_server() 
    

    A_tot=scipy.sparse.csc_matrix((d,K*len(file_res)))
    C_tot=np.zeros((K*len(file_res),T))
    sn_tot=np.zeros((d1*d2))
    count=0
    
    for idx_,shapes,A,b,C,f,S,bl,c1,neurons_sn,g,sn,_ in file_res[:]:
        sn_tot[idx_]=sn
        print count
        for ii in range(np.shape(A)[-1]):            
            new_comp=A.tocsc()[:,ii]/np.sqrt(np.sum(np.array(A.tocsc()[:,ii].todense())**2))
            if new_comp.sum()>0:
                A_tot[idx_,count]=new_comp
                C_tot[count,:]=C[ii,:]
                count+=1
                
    A_tot=A_tot[:,:count]
    C_tot=C_tot[:count,:]    
    
    np.savez('results_analysis_patch.npz',A_tot=A_tot.todense(), C_tot=C_tot, sn_tot=sn_tot,d1=d1,d2=d2)    

    options['preprocess_params']['n_pixels_per_process']=np.int((d1*d2)/n_processes/(T/2000.))
    options['spatial_params']['n_pixels_per_process']=np.int((d1*d2)/n_processes/(T/2000.))
    options['temporal_params']['n_pixels_per_process']=np.int((d1*d2)/n_processes/(T/2000.))
    
    st=time.time() 
    options['temporal_params']['p'] = 0              
    A_m,C_m,nr_m,merged_ROIs,S_m,bl_m,c1_m,sn_m,g_m=cse.merge_components(Yr,A_tot,[],np.array(C_tot),[],np.array(C_tot),[],options['temporal_params'],options['spatial_params'],thr=0.8,mx=np.Inf)     
    print time.time()-st


#    Cn = cse.utilities.local_correlations(Y)
    Cn=np.std(Y,axis=-1)
    crd = cse.utilities.plot_contours(A_m,Cn,thr=0.9)
    
    pixels_bckgrnd=np.nonzero(A_m.sum(axis=-1)==0)[0]
    
    f=np.sum(Yr[pixels_bckgrnd,:],axis=0)
    
    cse.utilities.start_server(n_processes)
    t1 = time.time()
    A2,b2,C2 = cse.spatial.update_spatial_components(Yr, C_m, f, A_m, sn=sn_tot, **options['spatial_params'])
    print time.time() - t1
    cse.utilities.stop_server() 

    #crd = cse.utilities.plot_contours(A2,Cn,thr=0.9)
    

#    correlations=np.corrcoef(np.array(C2))
#    centers=cse.com(A2.todense(),d1,d2)
#    distances=sklearn.metrics.pairwise.euclidean_distances(centers)
#    np.fill_diagonal(correlations,0)
#    np.fill_diagonal(distances,np.Inf)
#    
#    pl.imshow((correlations>.7) & (distances<8))
#    
#    crd = cse.utilities.plot_contours(A2,Cn,thr=0.9)
    
    cse.utilities.start_server(n_processes)
    
    t1 = time.time()
    options['temporal_params']['p'] = p # set it back to original value to perform full deconvolution
    C2,f2,S2,bl2,c12,neurons_sn2,g21,YrA = cse.temporal.update_temporal_components(Yr,A2,b2,C2,f,bl=None,c1=None,sn=None,g=None,**options['temporal_params'])
    print time.time() - t1
    
    cse.utilities.stop_server() 
    A_or, C_or, srt = cse.utilities.order_components(A2,C2)
    #cse.utilities.view_patches(Yr,coo_matrix(A_or),C_or,b2,f2,d1,d2,YrA = YrA[srt,:], secs=1)
    cse.utilities.view_patches_bar(Yr,coo_matrix(A_or),C_or,b2,f2, d1,d2, YrA=YrA[srt,:])  
    np.savez('results_analysis.npz',Cn=Cn,A_tot=A_tot.todense(), C_tot=C_tot, sn_tot=sn_tot, A2=A2.todense(),C2=C2,b2=b2,S2=S2,f2=f2,bl2=bl2,c12=c12, neurons_sn2=neurons_sn2, g21=g21,YrA=YrA,d1=d1,d2=d2)    
    scipy.io.savemat('output_analysis_matlab.mat',{'A_or':A_or,'C_or':C_or , 'YrA_or':YrA[srt,:], 'S_or': S2[srt,:] })
    A_norm=np.array([A_or[:,rr]/np.sqrt(np.sum(A_or[:,rr]**2)) for rr in range(A_or.shape[-1])]).T
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

A1,A2,C1,C2=extract_rois_patch(file_name,d1,d2,rf=8,stride = 5)
#%%
import os
import ca_source_extraction as cse 
import calblitz as cb
import time
import psutil
import sys
import scipy
import pylab as pl
import numpy as np
%load_ext autoreload
%autoreload 2

n_processes = np.maximum(psutil.cpu_count() - 2,1) # roughly number of cores on your machine minus 1
#print 'using ' + str(n_processes) + ' processes'
print "Stopping  cluster to avoid unnencessary use of memory...."
sys.stdout.flush()  
cse.utilities.stop_server() 
cse.utilities.start_server(n_processes)

#file_name='/home/agiovann/Dropbox (Simons Foundation)/Jeff/SmallExample/Yr_small.npy'
#file_name='/home/agiovann/Dropbox (Simons Foundation)/Jeff/SmallExample/Yr.npy'
file_name='/home/agiovann/Dropbox (Simons Foundation)/Jeff/challenge/Yr.npy'
fnames=[]
for file in os.listdir("./"):
    if file.endswith(".tif"):
        fnames.append(file)
fnames.sort()
print fnames  
_,d1,d2=np.shape(cb.load(fnames[0][:-3]+'hdf5',subindices=range(3),fr=10))
#%%
st=time.time()

A1,A2,C1,C2=extract_rois_patch(file_name,d1,d2,rf=8,stride = 5)