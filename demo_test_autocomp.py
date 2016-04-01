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
import calblitz as cb
import cv2
import scipy
#%%
n_processes = np.maximum(psutil.cpu_count() - 2,1) # roughly number of cores on your machine minus 1
#print 'using ' + str(n_processes) + ' processes'
p=2 # order of the AR model (in general 1 or 2)

#%% start cluster for efficient computation
print "Stopping  cluster to avoid unnencessary use of memory...."
sys.stdout.flush()  
cse.utilities.stop_server()

#%% LOAD MOVIE AND MAKE DIMENSIONS COMPATIBLE WITH CNMF
reload=0
filename='patch.tif'
#filename='PCsforPC.tif'
#filename='movies/demoMovie.tif'
t = tifffile.TiffFile(filename) 
Yr = t.asarray().astype(dtype=np.float32) 
#Yr=Yr[-3000:,:]
Yr = np.transpose(Yr,(1,2,0))
d1,d2,T=Yr.shape
Yr=np.reshape(Yr,(d1*d2,T),order='F')
#np.save('Y',Y)
np.save('Yr',Yr)
#Y=np.load('Y.npy',mmap_mode='r')
Yr=np.load('Yr.npy',mmap_mode='r')        
Y=np.reshape(Yr,(d1,d2,T),order='F')
Cn = cse.utilities.local_correlations(Y)
#n_pixels_per_process=d1*d2/n_processes # how to subdivide the work among processes

pl.imshow(Cn,cmap=pl.cm.gray)
#%%
options = cse.utilities.CNMFSetParms(Y,p=p,gSig=[7,7],K=30)
cse.utilities.start_server(options['spatial_params']['n_processes'])

#%% PREPROCESS DATA AND INITIALIZE COMPONENTS
t1 = time()
Yr,sn,g,psx = cse.pre_processing.preprocess_data(Yr,**options['preprocess_params'])
Atmp, Ctmp, b_in, f_in, center=cse.initialization.initialize_components(Y, **options['init_params'])                                                    
print time() - t1

#%% Refine manually component by clicking on neurons 
refine_components=False
if refine_components:
    Ain,Cin = cse.utilities.manually_refine_components(Y,options['init_params']['gSig'],coo_matrix(Atmp),Ctmp,Cn,thr=0.9)
else:
    Ain,Cin = Atmp, Ctmp
#%% plot estimated component
crd = cse.utilities.plot_contours(coo_matrix(Ain),Cn,thr=0.9)  
pl.show()
#%% UPDATE SPATIAL COMPONENTS
pl.close()
t1 = time()
A,b,Cin = cse.spatial.update_spatial_components(Yr, Cin, f_in, Ain, sn=sn, **options['spatial_params'])
t_elSPATIAL = time() - t1
print t_elSPATIAL 
plt.figure()
crd = cse.utilities.plot_contours(A,Cn,thr=0.9)
#%% update_temporal_components
pl.close()
t1 = time()
options['temporal_params']['p'] = 0 # set this to zero for fast updating without deconvolution
C,f,S,bl,c1,neurons_sn,g,YrA = cse.temporal.update_temporal_components(Yr,A,b,Cin,f_in,bl=None,c1=None,sn=None,g=None,**options['temporal_params'])
t_elTEMPORAL = time() - t1
print t_elTEMPORAL 
#%% merge components corresponding to the same neuron
t1 = time()
A_m,C_m,nr_m,merged_ROIs,S_m,bl_m,c1_m,sn_m,g_m=cse.merging.merge_components(Yr,A,b,C,f,S,sn,options['temporal_params'], options['spatial_params'], bl=bl, c1=c1, sn=neurons_sn, g=g, thr=0.8, mx=50, fast_merge = True)
t_elMERGE = time() - t1
print t_elMERGE  

#%%
plt.figure()
crd = cse.plot_contours(A_m,Cn,thr=0.9)
#%% refine spatial and temporal 
pl.close()
t1 = time()
A2,b2,C2 = cse.spatial.update_spatial_components(Yr, C_m, f, A_m, sn=sn, **options['spatial_params'])
options['temporal_params']['p'] = p # set it back to original value to perform full deconvolution
C2,f2,S2,bl2,c12,neurons_sn2,g21,YrA = cse.temporal.update_temporal_components(Yr,A2,b2,C2,f,bl=None,c1=None,sn=None,g=None,**options['temporal_params'])
print time() - t1
#%%
A_or, C_or, srt = cse.utilities.order_components(A2,C2)
#cse.utilities.view_patches(Yr,coo_matrix(A_or),C_or,b2,f2,d1,d2,YrA = YrA[srt,:], secs=1)
cse.utilities.view_patches_bar(Yr,coo_matrix(A_or),C_or,b2,f2, d1,d2, YrA=YrA[srt,:])  
#plt.show(block=True) 
plt.show()  
 
#%%

plt.figure()
crd = cse.utilities.plot_contours(A_or,Cn,thr=0.9)

#%% STOP CLUSTER
pl.close()
cse.utilities.stop_server()
#%%
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA,NMF
from sklearn.mixture import GMM

Ys=Yr
thresh_probability=0.5
num_psd_elms_high_freq=49;
cl_thr=0.8
#[sn,psx] = get_noise_fft(Ys,options);
#P.sn = sn(:);
#fprintf('  done \n');
psdx = np.sqrt(psx[:,3:]);
X = psdx[:,1:np.minimum(np.shape(psdx)[1],150)];
X = X-np.mean(X,axis=1)[:,np.newaxis]#     bsxfun(@minus,X,mean(X,2));     % center
#X = X/sn[:,np.newaxis]# 
X = X/np.percentile(X,90,axis=1)[:,np.newaxis]

#X = X/(+1e-5+np.std(X,axis=1)[:,np.newaxis])
#epsilon=1e-9
#X = X/(epsilon+np.linalg.norm(X,axis=1,ord=1)[:,np.newaxis])



pc=PCA(n_components=5)
cp=pc.fit_transform(X)

#nmf=NMF(n_components=2)
#nmr=nmf.fit_transform(X)

gmm=GMM(n_components=2)
Cx=gmm.fit_predict(cp)

L=gmm.predict_proba(cp)
Cx1=np.vstack([np.mean(X[Cx==0],0),np.mean(X[Cx==1],0)])

ind=np.argmin(np.mean(Cx1[:,-num_psd_elms_high_freq:],axis=1))
active_pixels = (L[:,ind]>thresh_probability)
active_pixels = L[:,ind]
pl.imshow(np.reshape((active_pixels),(d1,d2),order='F'))
#%%
ff=np.zeros(np.shape(A_or)[-1])
cl_thr=0.2
#ff = false(1,size(Am,2));
for i in range(np.shape(A_or)[-1]):
    a1 = A_or[:,i]
    a2 = A_or[:,i]*active_pixels
    if np.sum(a2**2) >= cl_thr**2*np.sum(a1**2):
        ff[i] = 1

id_set=1
cse.utilities.view_patches_bar(Yr,coo_matrix(A_or[:,ff==id_set]),C_or[ff==id_set,:],b2,f2, d1,d2, YrA=YrA[srt[ff==id_set],:])  


#km=KMeans(n_clusters=2)
#Cx=km.fit_transform(X)
#Cx=km.fit_transform(cp)
#Cx=km.cluster_centers_
#L=km.labels_
#ind=np.argmin(np.mean(Cx[:,-49:],axis=1))
#active_pixels = (L==ind)
#centroids = Cx;
#%% GUIDED FILTER
pl.imshow(np.mean(Y,axis=-1),cmap=pl.cm.gray)
#%%
pl.imshow(cv2.bilateralFilter(np.mean(Y,axis=-1),3,5,0),cmap=pl.cm.gray)
#%% BILATERAL FILTER EXAMPLE
N=10000
#     
m=cb.movie(np.transpose(np.array(Y[:,:,:N]),[2,0,1]),fr=30)
m=m.resize(1,1,.1)


mn2=m.copy()
mn1=m.copy().bilateral_blur_2D(diameter=10,sigmaColor=10000,sigmaSpace=0)     

mn1,shifts,xcorrs, template=mn1.motion_correct()
mn2=mn2.apply_shifts(shifts)     
#mn1=cb.movie(np.transpose(np.array(Y_n),[2,0,1]),fr=30)
mn=cb.concatenate([mn1,m],axis=1)
(mn-np.mean(mn)).play(gain=2.,magnification=4,backend='opencv',fr=20)
#%% GUIDED FILTER EXAMPLE
N=0#Y.shape[-1]
N1=30000
#     

m=cb.movie(np.transpose(np.array(Y[:,:,N:N1]),[2,0,1]),fr=30)
#m=m.IPCA_denoise(components=100,batch=10000)
#m=m.resize(1,1,.5)
#m=cb.movie(scipy.ndimage.median_filter(m, size=(3,2,2), mode='nearest'),fr=30)
#m=m.bilateral_blur_2D(diameter=10,sigmaColor=10000,sigmaSpace=0)
#m=cb.movie(scipy.ndimage.percentile_filter(m, 90, size=(3,2,2), mode='nearest'),fr=30)
m=cb.movie(scipy.ndimage.gaussian_filter(m, sigma=(1,1,1), mode='nearest',truncate=2),fr=30)
#m=cb.movie(scipy.ndimage.percentile_filter(m, 90, size=(3,2,2), mode='nearest'),fr=30)
m=m.resize(1,1,.1)
(m-np.mean(m)).play(gain=5.,magnification=4,backend='opencv',fr=300)
#%% MITYA's APPROACH
from sklearn.decomposition import MiniBatchDictionaryLearning,PCA,NMF,IncrementalPCA

m1=m[:]
alpha= 10e2# PC 
#mdl = IncrementalPCA(n_components=10,batch_size=500)
mdl = NMF(n_components=15,verbose=True,init='nndsvd',tol=1e-10,max_iter=200,shuffle=True,alpha=alpha,l1_ratio=1)



T,d1,d2=np.shape(m1)
d=d1*d2
yr=np.reshape(m1,[T,d],order='F')
yr=yr.T
X=mdl.fit(yr)
X=mdl.components_.T
X=mdl.transform(yr)
pl.figure()
for idx,mm in enumerate(X.T):
    pl.subplot(6,5,idx+1)
    pl.imshow(np.reshape(mm,(d1,d2),order='F'),cmap=pl.cm.gray)

#%%
#ym=np.median(m,axis=0)
ym=m.bin_median()
#ym=np.std(m,axis=0)
#clahe = cv2.createCLAHE(clipLimit=100., tileGridSize=(11,11))
guide_filter=np.uint8(255*(ym-np.min(ym))/(np.max(ym)-np.min(ym)))
#guide_filter=clahe.apply(guide_filter)

mn2=m.copy()
mn1=m.copy()#.bilateral_blur_2D(diameter=0,sigmaColor=10000,sigmaSpace=0)     
mn1=mn1.guided_filter_blur_2D(guide_filter,radius=3, eps=0)   
   
mn=cb.concatenate([mn1,m],axis=1)
(mn-np.mean(mn)).play(gain=5.,magnification=4,backend='opencv',fr=300)

#%%
from sklearn.decomposition import MiniBatchDictionaryLearning,SparsePCA,NMF
#m=mn1

alpha= 10e2# PC                                     
#    alpha=10e1# Jeff
mdl = NMF(n_components=15,verbose=True,init='nndsvd',tol=1e-10,max_iter=200,shuffle=True,alpha=alpha,l1_ratio=1)



perc=8 #PC   
m1= np.maximum(0,m-np.percentile(m,perc,axis=0))[:]#[:,20:35,20:35].resize(1,1,.05)
#m1= np.maximum(0,m-mmm)[:]#[:,20:35,20:35].resize(1,1,.05)
                                           
#    mdl = NMF(n_components=50,verbose=True,init='nndsvd',tol=1e-10,max_iter=600,shuffle=True,alpha=50e-2,l1_ratio=1)                                              
#    m1= np.maximum(0,m[:,10:-10,10:-10].computeDFF()[0])#[:,20:35,20:35].resize(1,1,.05)

#    mdl = NMF(n_components=50,verbose=True,init='nndsvd',tol=1e-10,max_iter=600,shuffle=True,alpha=10e-1,l1_ratio=1)
#    win_loc=5#PC
#    win_loc=12
#    m1=np.maximum(0,m[:,10:-10,10:-10].local_correlations_movie(window=win_loc))


T,d1,d2=np.shape(m1)
d=d1*d2
yr=np.reshape(m1,[T,d],order='F')
yr=yr.T
X=mdl.fit(yr)
X=mdl.components_.T
X=mdl.transform(yr)
pl.figure()
for idx,mm in enumerate(X.T):
    pl.subplot(6,5,idx+1)
    pl.imshow(np.reshape(mm,(d1,d2),order='F'),cmap=pl.cm.gray)
#%%
pl.plot(mdl.components_.T)    
mdl1 = NMF(n_components=15,verbose=True,init='nndsvd',tol=1e-10,max_iter=200,shuffle=True,alpha=alpha,l1_ratio=1)
X1=mdl1.fit_transform(mdl.components_)
#%% ONLINE NMF
perc=8
myfloat=np.float32
m1= np.maximum(0,m-np.percentile(m,perc,axis=0))
tm,sp=m1.online_NMF(n_components=15)
pl.figure()
for idx,mm in enumerate(sp):
    pl.subplot(6,5,idx+1)
    pl.imshow(mm,cmap=pl.cm.gray)
#%%
import spams
from PIL import Image
import time

perc=8
myfloat=np.float32
m1= np.maximum(0,m-np.percentile(m,perc,axis=0))[:]#[:,20:35,20:35].resize(1,1,.05)
T,d1,d2=np.shape(m1)
d=d1*d2
yr=np.reshape(m1,[T,d],order='F')
#img_file = 'boat.png'
#try:
#    img = Image.open(img_file)
#except:
#    print "Cannot load image %s : skipping test" %img_file
#    
#I = np.array(img) / 255.
#if I.ndim == 3:
#    A = np.asfortranarray(I.reshape((I.shape[0],I.shape[1] * I.shape[2])),dtype = myfloat)
#    rgb = True
#else:
#    A = np.asfortranarray(I,dtype = myfloat)
#    rgb = False
#
#m = 16;n = 16;
#X = spams.im2col_sliding(A,m,n,rgb)
#X = X[:,::10]
X = np.asfortranarray(yr,dtype = myfloat)
########## FIRST EXPERIMENT ###########
tic = time.time()
#(U,V) = spams.nmf(X,return_lasso= True,K = 15,numThreads=4,iter = -5)
(U,V) = spams.nnsc(X,return_lasso=True,K=15,lambda1=100)


tac = time.time()
t = tac - tic
pl.figure()
for idx,mm in enumerate(V):
    pl.subplot(6,5,idx+1)
    pl.imshow(np.reshape(mm.todense(),(d1,d2),order='F'),cmap=pl.cm.gray)

#%%
m=cb.movie(np.transpose(np.array(Y[:,:,N:N1]),[2,0,1]),fr=30)
num_samp=np.round(m.shape[0])
traces=mdl.components_    
new_tr=np.zeros((traces.shape[0],num_samp))
for idx,tr in enumerate(traces):
    print idx
    new_tr[idx]=np.maximum(scipy.signal.resample(tr,num_samp),0)

mdl1 = NMF(n_components=15,verbose=True,init='custom',tol=1e-10,max_iter=200,shuffle=True,alpha=alpha,l1_ratio=1)


m1= np.maximum(0,m-np.percentile(m,perc,axis=0))[:]#[:,20:35,20:35].resize(1,1,.05)
T,d1,d2=np.shape(m1)
d=d1*d2
yr=np.reshape(m1,[T,d],order='F')
newX=mdl1.fit_transform(yr.T,W=X.copy(),H=new_tr)
pl.figure()
for idx,mm in enumerate(newX.T):
    pl.subplot(6,5,idx+1)
    pl.imshow(np.reshape(mm,(d1,d2),order='F'),cmap=pl.cm.jet)    
#%%
id_c=7
N=10
pl.plot(np.convolve(mdl.components_[id_c].T, np.ones((N,))/N, mode='valid'))


#%%
def mode(inputData, axis=None, dtype=None):

   """
   Robust estimator of the mode of a data set using the half-sample mode.
   
   .. versionadded: 1.0.3
   """
   import numpy
   if axis is not None:
      fnc = lambda x: mode(x, dtype=dtype)
      dataMode = numpy.apply_along_axis(fnc, axis, inputData)
   else:
      # Create the function that we can use for the half-sample mode
      def _hsm(data):
         if data.size == 1:
            return data[0]
         elif data.size == 2:
            return data.mean()
         elif data.size == 3:
            i1 = data[1] - data[0]
            i2 = data[2] - data[1]
            if i1 < i2:
               return data[:2].mean()
            elif i2 > i1:
               return data[1:].mean()
            else:
               return data[1]
         else:
#            wMin = data[-1] - data[0]
            wMin=np.inf
            N = data.size/2 + data.size%2 
            for i in xrange(0, N):
               w = data[i+N-1] - data[i] 
               if w < wMin:
                  wMin = w
                  j = i

            return _hsm(data[j:j+N])
            
      data = inputData.ravel()
      if type(data).__name__ == "MaskedArray":
         data = data.compressed()
      if dtype is not None:
         data = data.astype(dtype)
         
      # The data need to be sorted for this to work
      data = numpy.sort(data)
      
      # Find the mode
      dataMode = _hsm(data)
      
   return dataMode


def std(inputData, Zero=False, axis=None, dtype=None):
   """
   Robust estimator of the standard deviation of a data set.  
   
   Based on the robust_sigma function from the AstroIDL User's Library.
   
   .. versionchanged:: 1.0.3
      Added the 'axis' and 'dtype' keywords to make this function more
      compatible with numpy.std()
   """
   
   if axis is not None:
      fnc = lambda x: std(x, dtype=dtype)
      sigma = numpy.apply_along_axis(fnc, axis, inputData)
   else:
      data = inputData.ravel()
      if type(data).__name__ == "MaskedArray":
         data = data.compressed()
      if dtype is not None:
         data = data.astype(dtype)
         
      if Zero:
         data0 = 0.0
      else:
         data0 = numpy.median(data)
      maxAbsDev = numpy.median(numpy.abs(data-data0)) / 0.6745
      if maxAbsDev < __epsilon:
         maxAbsDev = (numpy.abs(data-data0)).mean() / 0.8000
      if maxAbsDev < __epsilon:
         sigma = 0.0
         return sigma
         
      u = (data-data0) / 6.0 / maxAbsDev
      u2 = u**2.0
      good = numpy.where( u2 <= 1.0 )
      good = good[0]
      if len(good) < 3:
         print "WARNING:  Distribution is too strange to compute standard deviation"
         sigma = -1.0
         return sigma
         
      numerator = ((data[good]-data0)**2.0 * (1.0-u2[good])**2.0).sum()
      nElements = (data.ravel()).shape[0]
      denominator = ((1.0-u2[good])*(1.0-5.0*u2[good])).sum()
      sigma = nElements*numerator / (denominator*(denominator-1.0))
      if sigma > 0:
         sigma = math.sqrt(sigma)
      else:
         sigma = 0.0
         
   return sigma
#%%
cmp_=np.reshape(X[:,id_c],(d1,d2), order='F')/(np.linalg.norm(X[:,id_c])**2)
vct=np.sum(m*cmp_,axis=(1,2))
pl.plot(vct-np.mean(vct))

#%%


Y_n=np.zeros(Y.shape)

N=1400
Y_n=Y_n[:,:,:N]
ym=np.median(Y,axis=-1)
clahe = cv2.createCLAHE(clipLimit=100., tileGridSize=(11,11))
ymm=np.uint8(255*(ym-np.min(ym))/(np.max(ym)-np.min(ym)))
ymm=clahe.apply(ymm)

y__=np.zeros(Y[:,:,0].shape,dtype=np.float32)
for fr in range(N):
     print fr
     y1=Y[:,:,fr].copy()
     y1 =   cv2.bilateralFilter(y1,5,10000,0)     
     y1=cv2.ximgproc.guidedFilter(ymm,y1,radius=1,eps=0)  
     Y_n[:,:,fr] =  y1

mn1=cb.movie(np.transpose(np.array(Y_n),[2,0,1]),fr=30)
mn2=cb.movie(np.transpose(np.array(Y[:,:,:N]),[2,0,1]),fr=30)
mn=cb.concatenate([mn1,mn2],axis=1)
(mn).play(gain=5.,magnification=4,backend='opencv',fr=100)
#%%
clahe = cv2.createCLAHE(clipLimit=100., tileGridSize=(11,11))
ymm=np.uint8(255*(ym-np.min(ym))/(np.max(ym)-np.min(ym)))
ymm=clahe.apply(ymm)
pl.imshow(ymm,cmap=pl.cm.gray)
#%%
pl.imshow(cv2.adaptiveThreshold(ymm,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2),cmap=pl.cm.gray)
pl.imshow(cv2.adaptiveThreshold(ymm,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2),cmap=pl.cm.gray)
pl.imshow(cv2.threshold(ymm,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[],cmap=pl.cm.gray)
#%%
def estimated_autocorrelation(x):
    """
    http://stackoverflow.com/q/14297012/190597
    http://en.wikipedia.org/wiki/Autocorrelation#Estimation
    """
    n = len(x)
    variance = x.var()
    x = x-x.mean()
    r = np.correlate(x, x, mode = 'full')[-n:]
    assert np.allclose(r, np.array([(x[:n-k]*x[-(n-k):]).sum() for k in range(n)]))
    result = r/(variance*(np.arange(n, 0, -1)))
    return result