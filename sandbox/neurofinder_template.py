# example python script for loading neurofinder data
#
# for more info see:
#
# - http://neurofinder.codeneuro.org
# - https://github.com/codeneuro/neurofinder
#
# requires three python packages
#
# - numpy
# - scipy
# - matplotlib
#
#%%
%load_ext autoreload
%autoreload 2
import matplotlib.pyplot as plt
import ca_source_extraction as cse
import calblitz as cb
from scipy.misc import imread
from glob import glob
import numpy as np
import matplotlib as mpl
mpl.use('TKAgg')
from matplotlib import pyplot as plt
import sys
import numpy as np
import ca_source_extraction as cse
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

#%%
fname_mov='neuro_0101.tif'
files=sorted(glob('images/*.tiff'))
#%% LOAD MOVIE HERE USE YOUR METHOD, Movie is frames x dim2 x dim2
m=cb.load_movie_chain(files,fr=30)
#m.save(fname_mov)

#%%
backend = 'local'
if backend == 'SLURM':
    n_processes = np.int(os.environ.get('SLURM_NPROCS'))
else:
    n_processes = np.maximum(np.int(psutil.cpu_count()),1) # roughly number of cores on your machine minus 1
print 'using ' + str(n_processes) + ' processes'
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

    print 'Using '+ str(len(c)) + ' processes'
    dview=c[:len(c)]
    
#%%
downsample_factor = .3  
base_name ='Yr'

name_new=cse.utilities.save_memmap_each([fname_mov], dview=dview,base_name=base_name, resize_fact=(1, 1, downsample_factor), remove_init=0,idx_xy=None )
fname_new=cse.utilities.save_memmap_join(name_new,base_name='Yr', n_chunks=6, dview=dview)

#%%    
#fname_new='Yr_d1_501_d2_398_d3_1_order_F_frames_369_.mmap'
Yr,dims,T=cse.utilities.load_memmap(fname_new)
d1,d2=dims
Y=np.reshape(Yr,dims+(T,),order='F')
#%%
Cn = cse.utilities.local_correlations(Y[:,:,:3000])
pl.imshow(Cn,cmap='gray')  

#%%
rf=15 # half-size of the patches in pixels. rf=25, patches are 50x50
stride = 2 #amounpl.it of overlap between the patches in pixels    
K=4 # number of neurons expected per patch
gSig=[5,5] # expected half size of neurons
merge_thresh=0.8 # merging threshold, max correlation allowed
p=2 #order of the autoregressive system
memory_fact=1; #unitless number accounting how much memory should be used. You will need to try different values to see which one would work the default is OK for a 16 GB system
save_results=True
#%% RUN ALGORITHM ON PATCHES
options_patch = cse.utilities.CNMFSetParms(Y,n_processes,p=0,gSig=gSig,K=K,ssub=1,tsub=4,thr=merge_thresh)
A_tot,C_tot,b,f,sn_tot, optional_outputs = cse.map_reduce.run_CNMF_patches(fname_new, (d1, d2, T), options_patch,rf=rf,stride = stride,
                                                                        dview=dview,memory_fact=memory_fact)
print 'Number of components:' + str(A_tot.shape[-1])      

#%%
if save_results:
    np.savez('results_analysis_patch.npz',A_tot=A_tot.todense(), C_tot=C_tot, sn_tot=sn_tot,d1=d1,d2=d2,b=b,f=f)    
#%% if you have many components this might take long!
pl.figure()
crd = cse.utilities.plot_contours(A_tot,Cn,thr=0.9)
#%% set parameters for full field of view analysis
options = cse.utilities.CNMFSetParms(Y,n_processes,p=0,gSig=gSig,K=A_tot.shape[-1],thr=merge_thresh)
pix_proc=np.minimum(np.int((d1*d2)/n_processes/(T/2000.)),np.int((d1*d2)/n_processes)) # regulates the amount of memory used
options['spatial_params']['n_pixels_per_process']=pix_proc
options['temporal_params']['n_pixels_per_process']=pix_proc
#%% merge spatially overlaping and temporally correlated components      
A_m,C_m,nr_m,merged_ROIs,S_m,bl_m,c1_m,sn_m,g_m=cse.merge_components(Yr,A_tot,[],np.array(C_tot),[],np.array(C_tot),[],options['temporal_params'],options['spatial_params'],dview=dview,thr=options['merging']['thr'],mx=np.Inf)     
#%% update temporal to get Y_r
options['temporal_params']['p']=0
options['temporal_params']['fudge_factor']=0.96 #change ifdenoised traces time constant is wrong
options['temporal_params']['backend']='ipyparallel'
C_m,f_m,S_m,bl_m,c1_m,neurons_sn_m,g2_m,YrA_m = cse.temporal.update_temporal_components(Yr,A_m,np.atleast_2d(b).T,C_m,f,dview=dview,bl=None,c1=None,sn=None,g=None,**options['temporal_params'])

#%% get rid of evenrually noisy components. 
# But check by visual inspection to have a feeling fot the threshold. Try to be loose, you will be able to get rid of more of them later!

traces=C_m+YrA_m
idx_components, fitness, erfc = cse.utilities.evaluate_components(traces,N=5,robust_std=False)
idx_components=idx_components[np.logical_and(True ,fitness < -5)]
print(len(idx_components))
cse.utilities.view_patches_bar(Yr,scipy.sparse.coo_matrix(A_m.tocsc()[:,idx_components]),C_m[idx_components,:],b,f_m, d1,d2, YrA=YrA_m[idx_components,:]
                ,img=Cn)  
#%%
A_m=A_m[:,idx_components]
C_m=C_m[idx_components,:]   

#%% display components  DO NOT RUN IF YOU HAVE TOO MANY COMPONENTS
pl.figure()
crd = cse.utilities.plot_contours(A_m,Cn,thr=0.9)
#%%
print 'Number of components:' + str(A_m.shape[-1])  
#%% UPDATE SPATIAL OCMPONENTS
t1 = time()
A2,b2,C2 = cse.spatial.update_spatial_components(Yr, C_m, f, A_m, sn=sn_tot,dview=dview, **options['spatial_params'])
print time() - t1
#%% UPDATE TEMPORAL COMPONENTS
options['temporal_params']['p']=p
options['temporal_params']['fudge_factor']=0.96 #change ifdenoised traces time constant is wrong
C2,f2,S2,bl2,c12,neurons_sn2,g21,YrA = cse.temporal.update_temporal_components(Yr,A2,b2,C2,f,dview=dview, bl=None,c1=None,sn=None,g=None,**options['temporal_params'])
#%% Order components
#A_or, C_or, srt = cse.utilities.order_components(A2,C2)
#%% stop server and remove log files
cse.utilities.stop_server(is_slurm = (backend == 'SLURM')) 
log_files=glob('Yr*_LOG_*')
for log_file in log_files:
    os.remove(log_file)
#%% order components according to a quality threshold and only select the ones wiht qualitylarger than quality_threshold. 
quality_threshold=-0
traces=C2+YrA
idx_components, fitness, erfc = cse.utilities.evaluate_components(traces,N=5,robust_std=False)
idx_components=idx_components[fitness<quality_threshold]
_,_,idx_components=cse.utilities.order_components(A2,C2)
print(idx_components.size*1./traces.shape[0])
#%%
pl.figure();
crd = cse.utilities.plot_contours(A2.tocsc()[:,idx_components],Cn,thr=0.9)
#%%
cse.utilities.view_patches_bar(Yr,scipy.sparse.coo_matrix(A2.tocsc()[:,idx_components]),C2[idx_components,:],b2,f2, d1,d2, YrA=YrA[idx_components,:])  
#%% save analysis results in python and matlab format
if save_results:
    np.savez('results_analysis.npz',Cn=Cn,A_tot=A_tot.todense(), C_tot=C_tot, sn_tot=sn_tot, A2=A2.todense(),C2=C2,b2=b2,S2=S2,f2=f2,bl2=bl2,c12=c12, neurons_sn2=neurons_sn2, g21=g21,YrA=YrA,d1=d1,d2=d2,idx_components=idx_components, fitness=fitness, erfc=erfc)    
    scipy.io.savemat('output_analysis_matlab.mat',{'A2':A2,'C2':C2 , 'YrA':YrA, 'S2': S2 ,'YrA': YrA, 'd1':d1,'d2':d2,'idx_components':idx_components, 'fitness':fitness })
#%%
if save_results:
    import sys
    import numpy as np
    import ca_source_extraction as cse
    from scipy.sparse import coo_matrix
    import scipy
    import pylab as pl
    import calblitz as cb
    
    
    
    with np.load('results_analysis.npz')  as ld:
          locals().update(ld)
    
    fname_new='Yr0_d1_512_d2_512_d3_1_order_C_frames_600_.mmap'
    
    Yr,(d1,d2),T=cse.utilities.load_memmap(fname_new)
    d,T=np.shape(Yr)
    Y=np.reshape(Yr,(d1,d2,T),order='F') # 3D version of the movie
    A2=scipy.sparse.coo_matrix(A2)

    
    traces=C2+YrA
    idx_components, fitness, erfc = cse.utilities.evaluate_components(traces,N=5,robust_std=False)
    #cse.utilities.view_patches(Yr,coo_matrix(A_or),C_or,b2,f2,d1,d2,YrA = YrA[srt,:], secs=1)
    cse.utilities.view_patches_bar(Yr,A2.tocsc()[:,idx_components],C2[idx_components,:],b2,f2, d1,d2, YrA=YrA[idx_components,:])  
    dims=(d1,d2)

#%%
import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage as ndi
from skimage.feature import canny
from skimage.filters import sobel
from skimage.morphology import watershed
import skimage
 
def extract_binary_masks_blob(A, neuron_radius,max_fraction=.3, minCircularity= 0.2, minInertiaRatio = 0.2,minConvexity = .2):
    """
    Function to extract masks from data. It will also perform a preliminary selectino of good masks based on criteria like shape and size
    
    Parameters:
    ----------
    A: scipy.sparse matris
        contains the components as outputed from the CNMF algorithm
        
    neuron_radius: float 
        neuronal radius employed in the CNMF settings (gSiz)
    
    max_fraction: float
        fraction of the maximum of a components use to threshold the 
    
    min_elevation_map=30
    
    max_elevation_map=150
    
    minCircularity= 0.2
    
    minInertiaRatio = 0.2
    
    minConvexity = .2
    
    Returns:
    --------
    
    """    

    params = cv2.SimpleBlobDetector_Params()
    params.minCircularity = minCircularity
    params.minInertiaRatio = minInertiaRatio 
    params.minConvexity = minConvexity    
    
    # Change thresholds
    params.blobColor=255
    
    params.minThreshold = 0.5
    params.maxThreshold = 1.5
    params.thresholdStep= .5
    
    min_elevation_map=max_fraction*255
    max_elevation_map=0.9*255
    
    params.minArea = np.pi*((neuron_radius*.75)**2)
    #params.maxArea = 4*np.pi*((gSig[0]-1)**2)
    
    
    
    params.filterByArea = True
    params.filterByCircularity = True
    params.filterByConvexity = True
    params.filterByInertia = True
    
    detector = cv2.SimpleBlobDetector_create(params)
    
    
    masks_ws=[]
    pos_examples=[] 
    neg_examples=[]


    for count,comp in enumerate(A.tocsc()[:].T):

        print count
        comp_d=np.array(comp.todense())
        comp_d=comp_d*(comp_d>(np.max(comp_d)*.3))
        comp_orig=np.reshape(comp.todense(),dims,order='F')
        comp_orig=(comp_orig-np.min(comp_orig))/(np.max(comp_orig)-np.min(comp_orig))*255
        gray_image=np.reshape(comp_d,dims,order='F')
        gray_image=(gray_image-np.min(gray_image))/(np.max(gray_image)-np.min(gray_image))*255
        gray_image=gray_image.astype(np.uint8)    

        
        # segment using watershed
        markers = np.zeros_like(gray_image)
        elevation_map = sobel(gray_image)
        markers[gray_image < min_elevation_map] = 1
        markers[gray_image > max_elevation_map] = 2    
        edges = watershed(elevation_map, markers)-1
         
        # only keep largest object 
        label_objects, nb_labels = ndi.label(edges)
        sizes = np.bincount(label_objects.ravel())
        idx_largest = np.argmax(sizes[1:])    
        edges=(label_objects==(1+idx_largest))
        edges=ndi.binary_fill_holes(edges)
        
        
        masks_ws.append(edges)
        keypoints = detector.detect(edges.astype(np.uint8)*200)
        
        if len(keypoints)>0:
    #        im_with_keypoints = cv2.drawKeypoints(gray_image, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            pos_examples.append(count)

        else:
            
            neg_examples.append(count)

    return np.array(masks_ws),np.array(pos_examples),np.array(neg_examples)
#%% extract binary masks
min_radius=gSig[0]
masks_ws,pos_examples,neg_examples=cse.utilities.extract_binary_masks_blob(A2, min_radius, dims, max_fraction=.3, minCircularity= 0.2, minInertiaRatio = 0.2,minConvexity = .2)
#%%
pl.subplot(1,2,1)
final_masks=np.array(masks_ws)[pos_examples]
pl.imshow(np.reshape(final_masks.mean(0),dims,order='F'),vmax=.001)
pl.subplot(1,2,2)
neg_examples_masks=np.array(masks_ws)[neg_examples]
pl.imshow(np.reshape(neg_examples_masks.mean(0),dims,order='F'),vmax=.001)
#%%
areas=np.sum(masks_ws,axis=(1,2))
min_area=(min_radius**2)*np.pi/2
quality_threshold=-15
traces=C2+YrA
idx_components, fitness, erfc = cse.utilities.evaluate_components(traces,N=5,robust_std=False)

idx_components=np.where(np.logical_and(fitness<quality_threshold , areas >= min_area))[0]
#_,_,idx_components=cse.utilities.order_components(A2,C2)
print(idx_components.size*1./traces.shape[0])
#%%
pl.subplot(1,2,1)
final_masks=np.array(masks_ws)[np.union1d(pos_examples,idx_components)]
pl.imshow(np.reshape(final_masks.mean(0),dims,order='F'),vmax=.001)
pl.subplot(1,2,2)
neg_examples_masks=np.array(masks_ws)[np.setdiff1d(neg_examples,idx_components)]
pl.imshow(np.reshape(neg_examples_masks.mean(0),dims,order='F'),vmax=.001)
#%%
fname='regions_cnmf.json'
regions_cnmf=cse.utilities.nf_masks_to_json( final_masks,fname)
#%%

#%%
# load the images
masks=cse.utilities.nf_load_masks('regions/regions.json',np.shape(m)[1:])
# show the outputs
plt.figure()
plt.subplot(2, 2, 1)
plt.imshow(final_masks.sum(axis=0), cmap='gray')
plt.subplot(2, 2, 2)
plt.imshow(masks.sum(axis=0), cmap='gray')
plt.subplot(2, 2, 3)
pl.imshow(np.reshape(A2.mean(1),dims,order='F'),vmax=.0005)
plt.imshow(masks.sum(axis=0), alpha=.3,cmap='hot')
plt.subplot(2, 2, 4)
pl.imshow(m.sum(0),cmap='gray')
plt.imshow(masks.sum(axis=0), alpha=.2,cmap='hot')
plt.show()
#%%
pl.imshow(np.std(m,0),cmap='gray',vmax=1000)

#%%
pl.close()
neg_examples_masks=np.array(masks_ws)[np.sort(neg_examples)][240:241]
pl.imshow(np.mean(neg_examples_masks,0))
#%%
params = cv2.SimpleBlobDetector_Params()
params.blobColor=255
params.minThreshold = max_fraction*255;
params.maxThreshold = 255;
params.thresholdStep= 10
params.minArea = np.pi*((gSig[0]-1)**2)
params.minCircularity= 0.2
params.minInertiaRatio = 0.2
params.filterByArea = False
params.filterByCircularity = True
params.filterByConvexity = True
params.minConvexity = .2
params.filterByInertia = True
detector = cv2.SimpleBlobDetector_create(params)
for m in neg_examples_masks:
    m1=m.astype(np.uint8)*200    
    m1=ndi.binary_fill_holes(m1)
    keypoints = detector.detect(m1.astype(np.uint8)*200)
    if len(keypoints)>0:
        pl.cla()
        pl.imshow(np.reshape(m,dims,order='F'),vmax=.001)
        pl.pause(1)
    else:
        print 'skipped'

#pl.colorbar()
#%%

#%%
masks_ben=ut.nf_read_roi_zip('neurofinder01.01_combined.zip',dims)
regions2=cse.utilities.nf_masks_to_json( masks_ben,'masks_ben.json')
pl.imshow(np.sum(masks_ben>0,0)>0)
pl.pause(3)
pl.imshow(5*np.sum(masks,0),alpha=.3)

#%%
# load the images
masks=cse.utilities.nf_load_masks('regions/regions.json',np.shape(m)[1:])
# show the outputs
plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(m.sum(axis=0), cmap='gray')
plt.subplot(1, 2, 2)
plt.imshow(masks.sum(axis=0), cmap='gray')
plt.show()
#%%
fname='regions_2.json'
regions2=cse.utilities.nf_masks_to_json(np.roll( masks,20,axis=1) ,fname)
#regions2=cse.utilities.nf_masks_to_json(masks,fname)
#%%  
from neurofinder import match,load,centers,shapes
a=load('regions/regions.json')
b=load(fname)
print match(a,b,threshold=90)
print centers(a,b,threshold=30)
print shapes(a,b)