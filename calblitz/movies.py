# -*- coding: utf-8 -*-
"""
Spyder Editor

author: agiovann
"""
#%%
import cv2
import os
import sys
import copy
import pims
import scipy.ndimage
import warnings
import numpy as np
from sklearn.decomposition import NMF,IncrementalPCA, FastICA
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
import pylab as plt
try:
    plt.ion()
except:
    1
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage import label

from skimage.transform import warp, AffineTransform
from skimage.feature import match_template
from skimage import data

import timeseries as ts
from traces import trace



#%%
class movie(ts.timeseries):
    """ 
    Class representing a movie.

    Example of usage
    
    input_arr='filename_tiff.tif'
    fr=33; # 33 Hz
    start_time=0
    m=movie(input_arr, start_time=0,fr=33);
    
    Parameters
    ----------
    input_arr: can be either a file name (tif, avi, npy) or an np.ndarray (width x height x time)
    start_time: time beginning movie
    fr: frame rate
    meta_data: dictionary including any custom meta data

    """
    def __new__(cls, input_arr,**kwargs):
        #         
        
        # case we load movie from file
        if type(input_arr) is str:
            file_name = input_arr
            
            
            kwargs['file_name']=file_name
                
            extension = os.path.splitext(file_name)[1]

            if extension == '.tif': # load avi file

                input_arr = np.array(pims.open(file_name))            
                # necessary for the way pims work with tiffs  
                input_arr = np.swapaxes(input_arr,1,2)       
                input_arr = input_arr[:,:,::-1]

            elif extension == '.avi': # load avi file

                cap = cv2.VideoCapture(file_name)
                length = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
                width  = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
                
                input_arr=np.zeros((length, height,width),dtype=np.uint8)
                counter=0
                ret=True
                while True:
                    # Capture frame-by-frame
                    ret, frame = cap.read()
                    if not ret:
                        break
                    input_arr[counter]=frame[:,:,0]
                    counter=counter+1
                    if not counter%100:
                        print counter
                
                # When everything done, release the capture
                cap.release()
                cv2.destroyAllWindows()   
                
            elif extension == '.npy': # load avi file                             
                input_arr=np.load(file_name)
            else:
                raise Exception('Unknown file type_')
            
        return super(movie, cls).__new__(cls, input_arr, **kwargs)
        
       
        

#    
    
    @staticmethod
    def load(file_name):
        '''
        load movie from file
        '''
        return movie(**np.load(file_name))  
        
    @staticmethod
    def extractROIsFromPCAICA(spcomps, numSTD=4, gaussiansigmax=2 , gaussiansigmay=2):
        """
        Given the spatial components output of the IPCA_stICA function extract possible regions of interest
        The algorithm estimates the significance of a components by thresholding the components after gaussian smoothing
        Parameters
        -----------
        spcompomps, 3d array containing the spatial components
        numSTD: number of standard deviation above the mean of the spatial component to be considered signiificant
        """        
        
        numcomps, width, height=spcomps.shape
        rowcols=int(np.ceil(np.sqrt(numcomps)));  
        
        #%
        allMasks=[];
        maskgrouped=[];
        for k in xrange(0,numcomps):
            comp=spcomps[k]
#            plt.subplot(rowcols,rowcols,k+1)
            comp=gaussian_filter(comp,[gaussiansigmay,gaussiansigmax])
            
            maxc=np.percentile(comp,99);
            minc=np.percentile(comp,1);
#            comp=np.sign(maxc-np.abs(minc))*comp;
            q75, q25 = np.percentile(comp, [75 ,25])
            iqr = q75 - q25
            minCompValuePos=np.median(comp)+numSTD*iqr/1.35;  
            minCompValueNeg=np.median(comp)-numSTD*iqr/1.35;            

            # got both positive and negative large magnitude pixels
            compabspos=comp*(comp>minCompValuePos)-comp*(comp<minCompValueNeg);


            #height, width = compabs.shape
            labeledpos, n = label(compabspos>0, np.ones((3,3)))
            maskgrouped.append(labeledpos)
            for jj in range(1,n+1):
                tmp_mask=np.asarray(labeledpos==jj)
                allMasks.append(tmp_mask)
#            labeledneg, n = label(compabsneg>0, np.ones((3,3)))
#            maskgrouped.append(labeledneg)
#            for jj in range(n):
#                tmp_mask=np.asarray(labeledneg==jj)
#                allMasks.append(tmp_mask)
#            plt.imshow(labeled)                             
#            plt.axis('off')         
        return allMasks,maskgrouped            

    
    def motion_correct(self, max_shift_w=5,max_shift_h=5, show_movie=False,template=None,method='opencv'):
        """
        Performs motion corretion using the opencv matchtemplate function. At every iteration a template is built by taking the median of all frames and then used to align the other frames.
         
        Parameters
        ----------
        max_shift: maximum pixel shifts allowed when correcting
        show_movie : display the movie wile correcting it
         
        Returns
        -------
        movCorr: motion corected movie              
        shifts : tuple, contains shifts in x and y and correlation with template
        template: the templates created at each iteration
        method: depends on what is installed 'opencv' or 'skimage' 
        """
        
        if np.percentile(self,1)<0:
            raise ValueError('The movie must only contain positive values')
            
        
        self=np.asanyarray(self,dtype=np.float32)
                    
        n_frames_,h_i, w_i = self.shape
        
        ms_w = max_shift_w
        ms_h = max_shift_h
        
        if template is None:
            template=np.median(self,axis=0)            
            
        template=template[ms_h:h_i-ms_h,ms_w:w_i-ms_w].astype(np.float32)    
        h,w = template.shape      # template width and height
        
        
        #% run algorithm, press q to stop it 
        shifts=[];   # store the amount of shift in each frame
        xcorrs=[];
        
        for i,frame in enumerate(self):
             if i%100==99:
                 print "Frame %i"%(i+1);
             if method == 'opencv':
                 res = cv2.matchTemplate(frame,template,cv2.TM_CCORR_NORMED)             
                 top_left = cv2.minMaxLoc(res)[3]
             elif method == 'skimage':
                 res = match_template(frame,template)                 
                 top_left = np.unravel_index(np.argmax(res),res.shape);
                 top_left=top_left[::-1]   

             avg_corr=np.mean(res);
             sh_y,sh_x = top_left
             bottom_right = (top_left[0] + w, top_left[1] + h)
        
             if (0 < top_left[1] < 2 * ms_h-1) & (0 < top_left[0] < 2 * ms_w-1):
                 # if max is internal, check for subpixel shift using gaussian
                 # peak registration
                 log_xm1_y = np.log(res[sh_x-1,sh_y]);             
                 log_xp1_y = np.log(res[sh_x+1,sh_y]);             
                 log_x_ym1 = np.log(res[sh_x,sh_y-1]);             
                 log_x_yp1 = np.log(res[sh_x,sh_y+1]);             
                 four_log_xy = 4*np.log(res[sh_x,sh_y]);
    
                 sh_x_n = -(sh_x - ms_h + (log_xm1_y - log_xp1_y) / (2 * log_xm1_y - four_log_xy + 2 * log_xp1_y))
                 sh_y_n = -(sh_y - ms_w + (log_x_ym1 - log_x_yp1) / (2 * log_x_ym1 - four_log_xy + 2 * log_x_yp1))
             else:
                 sh_x_n = -(sh_x - ms_h)
                 sh_y_n = -(sh_y - ms_w)
             if method == 'opencv':        
                 M = np.float32([[1,0,sh_y_n],[0,1,sh_x_n]])
                 self[i] = cv2.warpAffine(frame,M,(w_i,h_i),flags=cv2.INTER_CUBIC)
             elif method == 'skimage':
                 tform = AffineTransform(translation=(-sh_y_n,-sh_x_n))             
                 self[i] = warp(frame, tform,preserve_range=True,order=3)
                 

             shifts.append([sh_x_n,sh_y_n]) 
             xcorrs.append([avg_corr])
             
             if show_movie:        
                 fr = cv2.resize(self[i],None,fx=2, fy=2, interpolation = cv2.INTER_CUBIC)
                 cv2.imshow('frame',fr/255.0)
                 if cv2.waitKey(1) & 0xFF == ord('q'):
                     cv2.destroyAllWindows()
                     break 
                 
        cv2.destroyAllWindows()
        return (self,template,shifts,xcorrs)
        
    def motion_correct_scikit(self, max_shift_w=5,max_shift_h=5, show_movie=False,template=None):
        """
        Performs motion corretion using the opencv matchtemplate function. At every iteration a template is built by taking the median of all frames and then used to align the other frames.
         
        Parameters
        ----------
        max_shift: maximum pixel shifts allowed when correcting
        show_movie : display the movie wile correcting it
         
        Returns
        -------
        movCorr: motion corected movie              
        shifts : tuple, contains shifts in x and y and correlation with template
        template: the templates created at each iteration
        """
        
        if np.percentile(self,1)<0:
            raise ValueError('The movie must only contain positive values')
            
        
        self=np.asanyarray(self,dtype=np.float32)
        
#        self=(self-np.min(self))/(np.max(self)-np.min(self))        
            
            
        n_frames_,h_i, w_i = self.shape
        
        ms_w = max_shift_w
        ms_h = max_shift_h
        
        if template is None:
            template=np.median(self,axis=0)            
            
        template=template[ms_h:h_i-ms_h,ms_w:w_i-ms_w].astype(np.float32)    
        h,w = template.shape      # template width and height
        
        
        #% run algorithm, press q to stop it 
        shifts=[];   # store the amount of shift in each frame
        xcorrs=[];
        
        for i,frame in enumerate(self):
             if i%100==99:
                 print "Frame %i"%(i+1);
             res = match_template(frame,template)
             avg_corr=np.mean(res);
             top_left = np.unravel_index(np.argmax(res),res.shape);
             top_left=top_left[::-1]             
             sh_y,sh_x = top_left
             bottom_right = (top_left[0] + w, top_left[1] + h)
        
             if (0 < top_left[1] < 2 * ms_h-1) & (0 < top_left[0] < 2 * ms_w-1):
                 # if max is internal, check for subpixel shift using gaussian
                 # peak registration
                 log_xm1_y = np.log(res[sh_x-1,sh_y]);             
                 log_xp1_y = np.log(res[sh_x+1,sh_y]);             
                 log_x_ym1 = np.log(res[sh_x,sh_y-1]);             
                 log_x_yp1 = np.log(res[sh_x,sh_y+1]);             
                 four_log_xy = 4*np.log(res[sh_x,sh_y]);
    
                 sh_x_n = -(sh_x - ms_h + (log_xm1_y - log_xp1_y) / (2 * log_xm1_y - four_log_xy + 2 * log_xp1_y))
                 sh_y_n = -(sh_y - ms_w + (log_x_ym1 - log_x_yp1) / (2 * log_x_ym1 - four_log_xy + 2 * log_x_yp1))
             else:
                 sh_x_n = -(sh_x - ms_h)
                 sh_y_n = -(sh_y - ms_w)
                     
#             M = np.float32([[1,0,sh_y_n],[0,1,sh_x_n]])
#             tform = AffineTransform(matrix=M)
             tform = AffineTransform(translation=(-sh_y_n,-sh_x_n))             
             self[i] = warp(frame, tform,preserve_range=True,order=3)
             
#             M = np.float32([[1,0,sh_y_n],[0,1,sh_x_n]])
#             self[i] = cv2.warpAffine(frame,M,(w_i,h_i),flags=cv2.INTER_CUBIC)

             shifts.append([sh_x_n,sh_y_n]) 
             xcorrs.append([avg_corr])
             
                         
             
             if show_movie:        
                 fr = cv2.resize(self[i],None,fx=2, fy=2, interpolation = cv2.INTER_CUBIC)
                 cv2.imshow('frame',fr/255.0)
                 if cv2.waitKey(1) & 0xFF == ord('q'):
                     cv2.destroyAllWindows()
                     break 
                 
        cv2.destroyAllWindows()
        return (self,template,shifts,xcorrs)
        
        
        
    def apply_shifts(self, shifts,interpolation='linear'):
        """ 
        Apply precomputed shifts to a movie, using subpixels adjustment (cv2.INTER_CUBIC function)
        
        Parameters
        ------------
        shifts: array of tuples representing x and y shifts for each frame        
        interpolation: 'linear', 'cubic', 'nearest' or cvs.INTER_XXX
        """
        self=np.asanyarray(self,dtype=np.float32)
        
        if interpolation == 'cubic':            
            interpolation=cv2.INTER_CUBIC
            
        elif interpolation == 'nearest':            
            interpolation=cv2.INTER_NEAREST 
            
        elif interpolation == 'linear':            
            interpolation=cv2.INTER_LINEAR
    
            
        t,h,w=self.shape
        for i,frame in enumerate(self):
            
             if i%100==99:
                 print "Frame %i"%(i+1); 

             sh_x_n, sh_y_n = shifts[i]
             M = np.float32([[1,0,sh_y_n],[0,1,sh_x_n]])                 
             self[i] = cv2.warpAffine(frame,M,(w,h),flags=interpolation)



        return self
    

    def crop(self,crop_top=0,crop_bottom=0,crop_left=0,crop_right=0,crop_begin=0,crop_end=0):
        """ Crop movie        
        """
        
        t,h,w=self.shape
        
        return self[crop_begin:t-crop_end,crop_top:h-crop_bottom,crop_left:w-crop_right]
        
        
    def computeDFF(self,secsWindow=5,quantilMin=8,squared_F=True):
        """ 
        compute the DFF of the movie
        In order to compute the baseline frames are binned according to the window length parameter
        and then the intermediate values are interpolated. 
        Parameters
        ----------
        secsWindow: length of the windows used to compute the quantile
        quantilMin : value of the quantile

        """
        
        print "computing minimum ..."; sys.stdout.flush()
        minmov=np.min(self)
        
        if np.min(self)<=0:
            raise ValueError("All pixels must be positive")

        numFrames,linePerFrame,pixPerLine=np.shape(self)
        downsampfact=int(secsWindow*self.fr);
        elm_missing=int(np.ceil(numFrames*1.0/downsampfact)*downsampfact-numFrames)
        padbefore=int(np.floor(elm_missing/2.0))
        padafter=int(np.ceil(elm_missing/2.0))
       
        print 'Inizial Size Image:' + np.str(np.shape(self)); sys.stdout.flush()
        self=movie(np.pad(self,((padbefore,padafter),(0,0),(0,0)),mode='reflect'),**self.__dict__)
        numFramesNew,linePerFrame,pixPerLine=np.shape(self)
        
        #% compute baseline quickly
        print "binning data ..."; sys.stdout.flush()
        movBL=np.reshape(self,(downsampfact,int(numFramesNew/downsampfact),linePerFrame,pixPerLine));
        movBL=np.percentile(movBL,quantilMin,axis=0);
        print "interpolating data ..."; sys.stdout.flush()   
        print movBL.shape 
        
        movBL=scipy.ndimage.zoom(np.array(movBL,dtype=np.float32),[downsampfact ,1, 1],order=0, mode='constant', cval=0.0, prefilter=False)
        
        #% compute DF/F
        if squared_F:
            self=(self-movBL)/np.sqrt(movBL)
        else:
            self=(self-movBL)/movBL
            
        self=self[padbefore:len(movBL)-padafter,:,:]; 
        print 'Final Size Movie:' +  np.str(self.shape) 
        return self         
        
    
    def NonnegativeMatrixFactorization(self,n_components=30, init='nndsvd', beta=1,tol=5e-7, sparseness='components',**kwargs):
        '''
        See documentation for scikit-learn NMF
        '''
        
        minmov=np.min(self)        
        if np.min(self)<0:
            raise ValueError("All values must be positive") 
            
        T,h,w=self.shape
        Y=np.reshape(self,(T,h*w))
        Y=Y-np.percentile(Y,1)
        Y=np.clip(Y,0,np.Inf)
        estimator=NMF(n_components=n_components, init=init, beta=beta,tol=tol, sparseness=sparseness,**kwargs)
        time_components=estimator.fit_transform(Y)
        components_ = estimator.components_        
        space_components=np.reshape(components_,(n_components,h,w))
        
        return space_components,time_components    
        
    
    def IPCA(self, components = 50, batch =1000):

        # Parameters:
        #   components (default 50)
        #     = number of independent components to return
        #   batch (default 1000)
        #     = number of pixels to load into memory simultaneously
        #       in IPCA. More requires more memory but leads to better fit


        # vectorize the images
        num_frames, h, w = np.shape(self);
        frame_size = h * w;
        frame_samples = np.reshape(self, (num_frames, frame_size)).T
        
        # run IPCA to approxiate the SVD
        
        ipca_f = IncrementalPCA(n_components=components, batch_size=batch)
        ipca_f.fit(frame_samples)
        
        # construct the reduced version of the movie vectors using only the 
        # principal component projection
        
        proj_frame_vectors = ipca_f.inverse_transform(ipca_f.transform(frame_samples))
            
        # get the temporal principal components (pixel time series) and 
        # associated singular values
        
        eigenseries = ipca_f.components_.T

        # the rows of eigenseries are approximately orthogonal
        # so we can approximately obtain eigenframes by multiplying the 
        # projected frame matrix by this transpose on the right
        
        eigenframes = np.dot(proj_frame_vectors, eigenseries)

        return eigenseries, eigenframes, proj_frame_vectors    
    
    def IPCA_stICA(self, components = 50, batch = 1000, mu = 1, ICAfun = 'logcosh'):
        # Parameters:
        #   components (default 50)
        #     = number of independent components to return
        #   batch (default 1000)
        #     = number of pixels to load into memory simultaneously
        #       in IPCA. More requires more memory but leads to better fit
        #   mu (default 0.05)
        #     = parameter in range [0,1] for spatiotemporal ICA,
        #       higher mu puts more weight on spatial information
        #   ICAFun (default = 'logcosh')
        #     = cdf to use for ICA entropy maximization    
        #
        # Returns:
        #   ind_frames [components, height, width]
        #     = array of independent component "eigenframes"
    
        eigenseries, eigenframes,_proj = self.IPCA(components, batch)
        # normalize the series
    
        frame_scale = mu / np.max(eigenframes)
        frame_mean = np.mean(eigenframes, axis = 0)
        n_eigenframes = frame_scale * (eigenframes - frame_mean)
    
        series_scale = (1-mu) / np.max(eigenframes)
        series_mean = np.mean(eigenseries, axis = 0)
        n_eigenseries = series_scale * (eigenseries - series_mean)
    
        # build new features from the space/time data
        # and compute ICA on them
    
        eigenstuff = np.concatenate([n_eigenframes, n_eigenseries])
    
        ica = FastICA(n_components=components, fun=ICAfun)
        joint_ics = ica.fit_transform(eigenstuff)
    
        # extract the independent frames
        num_frames, h, w = np.shape(self);
        frame_size = h * w;
        ind_frames = joint_ics[:frame_size, :]
        ind_frames = np.reshape(ind_frames.T, (components, h, w))
        
        return ind_frames  


    def IPCA_denoise(self, components = 50, batch = 1000):
        _, _, clean_vectors = self.IPCA(components, batch)
        self = self.__class__(np.reshape(clean_vectors.T, np.shape(self)),**self.__dict__)
                
        
    def compute_StructuredNMFactorization(self):
        print "to do"
        
   
    def local_correlations(self,eight_neighbours=False):
         # Output:
         #   rho M x N matrix, cross-correlation with adjacent pixel
         # if eight_neighbours=True it will take the diagonal neighbours too

         rho = np.zeros(np.shape(self)[1:3])
         w_mov = (self - np.mean(self, axis = 0))/np.std(self, axis = 0)
 
         rho_h = np.mean(np.multiply(w_mov[:,:-1,:], w_mov[:,1:,:]), axis = 0)
         rho_w = np.mean(np.multiply(w_mov[:,:,:-1], w_mov[:,:,1:,]), axis = 0)
         
         if True:
             rho_d1 = np.mean(np.multiply(w_mov[:,1:,:-1], w_mov[:,:-1,1:,]), axis = 0)
             rho_d2 = np.mean(np.multiply(w_mov[:,:-1,:-1], w_mov[:,1:,1:,]), axis = 0)


         rho[:-1,:] = rho[:-1,:] + rho_h
         rho[1:,:] = rho[1:,:] + rho_h
         rho[:,:-1] = rho[:,:-1] + rho_w
         rho[:,1:] = rho[:,1:] + rho_w
         
         if eight_neighbours:
             rho[:-1,:-1] = rho[:-1,:-1] + rho_d2
             rho[1:,1:] = rho[1:,1:] + rho_d1
             rho[1:,:-1] = rho[1:,:-1] + rho_d1
             rho[:-1,1:] = rho[:-1,1:] + rho_d2
         
         
         if eight_neighbours:
             neighbors = 8 * np.ones(np.shape(self)[1:3])  
             neighbors[0,:] = neighbors[0,:] - 3;
             neighbors[-1,:] = neighbors[-1,:] - 3;
             neighbors[:,0] = neighbors[:,0] - 3;
             neighbors[:,-1] = neighbors[:,-1] - 3;
             neighbors[0,0] = neighbors[0,0] + 1;
             neighbors[-1,-1] = neighbors[-1,-1] + 1;
             neighbors[-1,0] = neighbors[-1,0] + 1;
             neighbors[0,-1] = neighbors[0,-1] + 1;
         else:
             neighbors = 4 * np.ones(np.shape(self)[1:3]) 
             neighbors[0,:] = neighbors[0,:] - 1;
             neighbors[-1,:] = neighbors[-1,:] - 1;
             neighbors[:,0] = neighbors[:,0] - 1;
             neighbors[:,-1] = neighbors[:,-1] - 1;
         
         
         

         rho = np.divide(rho, neighbors)

         return rho
         
    def partition_FOV_KMeans(self,tradeoff_weight=.5,fx=.25,fy=.25,n_clusters=4,max_iter=500):
        """ 
        Partition the FOV in clusters that are grouping pixels close in space and in mutual correlation
                        
        Parameters
        ------------------------------
        tradeoff_weight:between 0 and 1 will weight the contributions of distance and correlation in the overall metric
        fx,fy: downsampling factor to apply to the movie 
        n_clusters,max_iter: KMeans algorithm parameters
        
        Outputs
        -------------------------------
        fovs:array 2D encoding the partitions of the FOV
        mcoef: matric of pairwise correlation coefficients
        distanceMatrix: matrix of picel distances
        
        Example
        
        """
        
        _,h1,w1=self.shape
        self.resize(fx,fy)
        T,h,w=self.shape
        Y=np.reshape(self,(T,h*w))
        mcoef=np.corrcoef(Y.T)
        idxA,idxB =  np.meshgrid(range(w),range(h));
        coordmat=np.vstack((idxA.flatten(),idxB.flatten()))
        distanceMatrix=euclidean_distances(coordmat.T);
        distanceMatrix=distanceMatrix/np.max(distanceMatrix)
        estim=KMeans(n_clusters=n_clusters,max_iter=max_iter);
        kk=estim.fit(tradeoff_weight*mcoef-(1-tradeoff_weight)*distanceMatrix)
        labs=kk.labels_
        fovs=np.reshape(labs,(h,w))
        fovs=cv2.resize(np.uint8(fovs),(w1,h1),1./fx,1./fy,interpolation=cv2.INTER_NEAREST)
        return np.uint8(fovs), mcoef, distanceMatrix
    
    
    def extract_traces_from_masks(self,masks):
        """                        
        Parameters
        ----------------------
        masks: array, 3D with each 2D slice bein a mask (integer or fractional)  
        type_: extracted fluorescence trace, if 'DFF' it will also extract DFF
        Outputs
        ----------------------
        traces: array, 2D of fluorescence traces
        tracesDFF: rray, 2D of DF/F traces
        """
        T,h,w=self.shape
        Y=np.reshape(self,(T,h*w))
        nA,_,_=masks.shape
        A=np.reshape(masks,(nA,h*w))
        pixelsA=np.sum(A,axis=1)
        A=A/pixelsA[:,None] # obtain average over ROI
        traces=trace(np.dot(A,np.transpose(Y)).T,**self.__dict__)               
        return traces
            
    def resize(self,fx=1,fy=1,fz=1,interpolation=cv2.INTER_AREA):  
        """
        resize movies along axis and interpolate or lowpass when necessary
        
        Parameters
        -------------------
        fx,fy,fz:fraction/multiple of dimension (.5 means the image will be half the size)
        interpolation=cv2.INTER_AREA. Set to none if you do not want interpolation or lowpass
        

        """              
        if fx!=1 or fy!=1:
            print "reshaping along x and y"
            t,h,w=self.shape
            newshape=(int(w*fy),int(h*fx))
            mov=[];
            print(newshape)
            for frame in self:                
                mov.append(cv2.resize(frame,newshape,fx=fx,fy=fy,interpolation=interpolation))
            self=movie(np.asarray(mov),**self.__dict__)
        if fz!=1:
            print "reshaping along z"            
            t,h,w=self.shape
            self=np.reshape(self,(t,h*w))            
            mov=cv2.resize(self,(h*w,int(fz*t)),fx=1,fy=fz,interpolation=interpolation)            
#            self=cv2.resize(self,(h*w,int(fz*t)),fx=1,fy=fz,interpolation=interpolation)
            mov=np.reshape(mov,(int(fz*t),h,w))
            self=movie(mov,**self.__dict__)
            self.fr=self.fr*fz  
            
        return self
        
      
    
    
    def play(self,gain=1,fr=None,magnification=2,offset=0):
         """
         Play the movie using opencv
         
         Parameters
         ----------
         gain: adjust  movie brightness
         frate : playing speed if different from original (inter frame interval in seconds)
         """  
         gain*=1.
         maxmov=np.max(self)
         if fr==None:
            fr=self.fr
         for frame in self:
            frame = cv2.resize(frame,None,fx=magnification, fy=magnification, interpolation = cv2.INTER_CUBIC)
            cv2.imshow('frame',(offset+frame)*gain/maxmov)
            if cv2.waitKey(int(1./fr*1000)) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break  
         cv2.destroyAllWindows()       
        

    

        
        


        
        
             
if __name__ == "__main__":
    mov=movie('/Users/agiovann/Dropbox/Preanalyzed Data/ExamplesDataAnalysis/Andrea/PC1/M_FLUO.tif',fr=15.62,start_time=0,meta_data={'zoom':2,'location':[100, 200, 300]})
    mov1=movie('/Users/agiovann/Dropbox/Preanalyzed Data/ExamplesDataAnalysis/Andrea/PC1/M_FLUO.tif',fr=15.62,start_time=0,meta_data={'zoom':2,'location':[100, 200, 300]})    
#    newmov=ts.concatenate([mov,mov1])    
#    mov.save('./test.npz')
#    mov=movie.load('test.npz')
    max_shift=5;
    mov,template,shifts,xcorrs=mov.motion_correct(max_shift_h=max_shift,max_shift_w=max_shift,show_movie=0)
    max_shift=5;
    mov1,template1,shifts1,xcorrs1=mov1.motion_correct(max_shift_h=max_shift,max_shift_w=max_shift,show_movie=0,method='skimage')
    
#    mov=mov.apply_shifts(shifts)    
#    mov=mov.crop(crop_top=max_shift,crop_bottom=max_shift,crop_left=max_shift,crop_right=max_shift)    
#    mov=mov.resize(fx=.25,fy=.25,fz=.2)    
#    mov=mov.computeDFF()      
#    mov=mov-np.min(mov)
#    space_components,time_components=mov.NonnegativeMatrixFactorization();
#    trs=mov.extract_traces_from_masks(1.*(space_components>0.4))
#    trs=trs.computeDFF()