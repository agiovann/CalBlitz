# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 09:02:06 2015

@author: agiovann
"""
#%%
#%load_ext autoreload
#%autoreload 2
import cv2
import copy
import pims
import scipy.ndimage
import warnings
import numpy as np
from sklearn.decomposition import NMF,IncrementalPCA, FastICA
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
import pylab as plt
plt.ion()
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage import label

import sys
import functools
import os.path

#%%

    
#%%
class XMovie(object):
    """ 
    Class representing a movie. Only need to provide the path to the tif file. 

    Example of usage
    
    filename='filename_tiff.tif'
    frameRate=you_frame_rate
    m=XMovie(filename, frameRate=frameRate);
    
    Parameters
    ----------
    filename : name of the tiff file
    frameRate : float, inter frame interval in seconds
    """
    
    def __init__(self, mov , frameRate=None):

        
        if  type(mov) is str:
            filename=mov
            extension = os.path.splitext(filename)[1]

            if extension == '.tif': # load avi file

                self.mov = np.array(pims.open(filename))            
                self.mov = np.swapaxes(self.mov,1,2)            
                self.mov=self.mov[:,:,::-1]

            elif extension == '.avi': # load avi file

                cap = cv2.VideoCapture(filename)
                length = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
                width  = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
                
                self.mov=np.zeros((length, height,width),dtype=np.uint8)
                counter=0
                ret=True
                while True:
                    # Capture frame-by-frame
                    ret, frame = cap.read()
                    if not ret:
                        break
                    self.mov[counter]=frame[:,:,0]
                    counter=counter+1
                    if not counter%100:
                        print counter
                
                # When everything done, release the capture
                cap.release()
                cv2.destroyAllWindows()            
            elif extension == '.npy': # load avi file                             
                self.mov=np.load(filename)
            else:
                raise Exception('Unknown file type_')
                
            self.filename = filename
            
        elif mov.ndim == 3:
             self.mov=mov
        else: 
            raise Exception('You need to specify either matrix or filename')
        
        if not frameRate is None:
            self.frameRate = frameRate
        else:
            raise Exception('You need to specify the frame rate')
    
    ################ STATIC METHODS
    #%%
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
           
   
    #%%
    ################ PUBLIC METHODS            
    
    def append(self,mov):
        """
        Append the movie mov to the object
        
        Parameters
        ---------------------------
        mov: XMovie object
        """        
        self.mov=np.append(self.mov,mov.mov,axis=0)     
    
    def applyShifstToMovie(self, shifts):
        """ 
        Apply precomputed shifts to a movie, using subpixels adjustment (cv2.INTER_CUBIC function)
        
        Parameters
        ------------
        shifts: array of tuples representing x and y shifts for each frame        
        
        """
        t,h,w=self.mov.shape
        for i,frame in enumerate(self.mov):
             if i%100==99:
                 print "Frame %i"%(i+1); 
             sh_x_n, sh_y_n = shifts[i]
             M = np.float32([[1,0,sh_y_n],[0,1,sh_x_n]])                 
             self.mov[i] = cv2.warpAffine(frame,M,(w,h),flags=cv2.INTER_CUBIC)
     

    def extract_traces_from_masks(self,masks,type_=None,window_sec=5,minQuantile=20):
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
        T,h,w=self.mov.shape
        Y=np.reshape(self.mov,(T,h*w))
        nA,_,_=masks.shape
        A=np.reshape(masks,(nA,h*w))
        pixelsA=np.sum(A,axis=1)
        A=A/pixelsA[:,None] # obtain average over ROI
        traces=np.dot(A,np.transpose(Y))
        
        if type_ == 'DFF':
            window=int(window_sec/self.frameRate);            
            assert window <= T, "The window must be shorter than the total length"
            tracesDFF=[]
            for trace in traces:
                traceBL=[np.percentile(trace[i:i+window],minQuantile) for i in xrange(1,len(trace)-window)]
                missing=np.percentile(trace[-window:],minQuantile);
                missing=np.repeat(missing,window+1)
                traceBL=np.concatenate((traceBL,missing))
                tracesDFF.append((trace-traceBL)/traceBL)
            tracesDFF=np.asarray(tracesDFF)
        else:
            tracesDFF=None
            
        return traces.T,tracesDFF.T
        
    def motion_correct(self, max_shift_w=5,max_shift_h=5, show_movie=False,template=None):
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
        
        
        self.mov=np.asarray(self.mov,dtype=np.float32);
        n_frames_,h_i, w_i = self.mov.shape
        
        ms_w = max_shift_w
        ms_h = max_shift_h
        
        if template is None:
            template=np.median(self.mov,axis=0)            
            
        template=template[ms_h:h_i-ms_h,ms_w:w_i-ms_w].astype(np.float32)    
        h,w = template.shape      # template width and height
        
        #if show_movie:
        #    cv2.imshow('template',template/255)
        #    cv2.waitKey(2000) 
        #    cv2.destroyAllWindows()
        
        #% run algorithm, press q to stop it 
        shifts=[];   # store the amount of shift in each frame
        
        for i,frame in enumerate(self.mov):
             if i%100==99:
                 print "Frame %i"%(i+1);
             res = cv2.matchTemplate(frame,template,cv2.TM_CCORR_NORMED)
             avg_corr=np.mean(res);
             top_left = cv2.minMaxLoc(res)[3]
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
                     
             M = np.float32([[1,0,sh_y_n],[0,1,sh_x_n]])
             self.mov[i] = cv2.warpAffine(frame,M,(w_i,h_i),flags=cv2.INTER_CUBIC)

             shifts.append([sh_x_n,sh_y_n,avg_corr]) 
                 
             if show_movie:        
                 fr = cv2.resize(self.mov[i],None,fx=2, fy=2, interpolation = cv2.INTER_CUBIC)
                 cv2.imshow('frame',fr/255.0)
                 if cv2.waitKey(1) & 0xFF == ord('q'):
                     cv2.destroyAllWindows()
                     break  
        cv2.destroyAllWindows()
        return (template,shifts)

             

    def makeSubMov(self,frames):        
        self.mov=self.mov[frames,:,:]

    
    
    def NonnegativeMatrixFactorization(self,n_components=30, init='nndsvd', beta=1,tol=5e-7, sparseness='components'):
        T,h,w=self.mov.shape
        Y=np.reshape(self.mov,(T,h*w))
        Y=Y-np.percentile(Y,1)
        Y=np.clip(Y,0,np.Inf)
        estimator=NMF(n_components=n_components, init=init, beta=beta,tol=tol, sparseness=sparseness)
        time_components=estimator.fit_transform(Y)
        components_ = estimator.components_        
        space_components=np.reshape(components_,(n_components,h,w))
        return space_components,time_components
        
    
    
    
    def plotFrame(self,numFrame):
        plt.imshow(self.mov[numFrame],cmap=plt.cm.Greys_r)
        

    def IPCA(self, components = 50, batch =1000):

        # Parameters:
        #   components (default 50)
        #     = number of independent components to return
        #   batch (default 1000)
        #     = number of pixels to load into memory simultaneously
        #       in IPCA. More requires more memory but leads to better fit


        # vectorize the images
        num_frames, h, w = np.shape(self.mov);
        frame_size = h * w;
        frame_samples = np.reshape(self.mov, (num_frames, frame_size)).T
        
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
    
    def copy(self):
        """ Create copy of the object"""
        return copy.copy(self)    
    
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
        num_frames, h, w = np.shape(self.mov);
        frame_size = h * w;
        ind_frames = joint_ics[:frame_size, :]
        ind_frames = np.reshape(ind_frames.T, (components, h, w))
        
        return ind_frames  


    def IPCA_denoise(self, components = 50, batch = 1000):
        _, _, clean_vectors = self.IPCA(components, batch)
        self.mov = np.reshape(clean_vectors.T, np.shape(self.mov))
                
        
    def compute_StructuredNMFactorization(self):
        print "to do"
        
   
    def local_correlations(self,eight_neighbours=False):
         # Output:
         #   rho M x N matrix, cross-correlation with adjacent pixel
         # if eight_neighbours=True it will take the diagonal neighbours too

         rho = np.zeros(np.shape(self.mov)[1:3])
         w_mov = (self.mov - np.mean(self.mov, axis = 0))/np.std(self.mov, axis = 0)
 
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
             neighbors = 8 * np.ones(np.shape(self.mov)[1:3])  
             neighbors[0,:] = neighbors[0,:] - 3;
             neighbors[-1,:] = neighbors[-1,:] - 3;
             neighbors[:,0] = neighbors[:,0] - 3;
             neighbors[:,-1] = neighbors[:,-1] - 3;
             neighbors[0,0] = neighbors[0,0] + 1;
             neighbors[-1,-1] = neighbors[-1,-1] + 1;
             neighbors[-1,0] = neighbors[-1,0] + 1;
             neighbors[0,-1] = neighbors[0,-1] + 1;
         else:
             neighbors = 4 * np.ones(np.shape(self.mov)[1:3]) 
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
        m1=self.copy()
        _,h1,w1=self.mov.shape
        m1.resize(fx,fy)
        T,h,w=m1.mov.shape
        Y=np.reshape(m1.mov,(T,h*w))
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
       
        
    def playMovie(self,gain=1.0,frate=None,magnification=2,offset=0):
         """
         Play the movie using opencv
         
         Parameters
         ----------
         gain: adjust  movie brightness
         frate : playing speed if different from original (inter frame interval in seconds)
         """       
         maxmov=np.max(self.mov)
         if frate==None:
            frate=self.frameRate
         for frame in self.mov:
            frame = cv2.resize(frame,None,fx=magnification, fy=magnification, interpolation = cv2.INTER_CUBIC)
            cv2.imshow('frame',(offset+frame)*gain/maxmov)
            if cv2.waitKey(int(frate*1000)) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break  
         cv2.destroyAllWindows()       
        
    def crop(self,crop_top=0,crop_bottom=0,crop_left=0,crop_right=0,crop_begin=0,crop_end=0):
        """ Crop movie        
        """
        
        t,h,w=self.mov.shape
        
        self.mov=self.mov[crop_begin:t-crop_end,crop_top:h-crop_bottom,crop_left:w-crop_right]
        
        
    def computeDFF(self,secsWindow=5,quantilMin=8,subtract_minimum=False,squared_F=True):
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
        minmov=np.min(self.mov)
        if subtract_minimum:
            self.mov=self.mov-np.min(self.mov)+.1
            minmov=np.min(self.mov)

        assert(minmov>0),"All pixels must be nonnegative"                       
        numFrames,linePerFrame,pixPerLine=np.shape(self.mov)
        downsampfact=int(secsWindow/self.frameRate);
        elm_missing=int(np.ceil(numFrames*1.0/downsampfact)*downsampfact-numFrames)
        padbefore=np.floor(elm_missing/2.0)
        padafter=np.ceil(elm_missing/2.0)
        print 'Inizial Size Image:' + np.str(np.shape(self.mov)); sys.stdout.flush()
        self.mov=np.pad(self.mov,((padbefore,padafter),(0,0),(0,0)),mode='reflect')
        numFramesNew,linePerFrame,pixPerLine=np.shape(self.mov)
        #% compute baseline quickly
        print "binning data ..."; sys.stdout.flush()
        movBL=np.reshape(self.mov,(downsampfact,int(numFramesNew/downsampfact),linePerFrame,pixPerLine));
        movBL=np.percentile(movBL,quantilMin,axis=0);
        print "interpolating data ..."; sys.stdout.flush()   
        print movBL.shape        
        movBL=scipy.ndimage.zoom(np.array(movBL,dtype=np.float32),[downsampfact ,1, 1],order=0, mode='constant', cval=0.0, prefilter=False)
        
        #% compute DF/F
        if squared_F:
            self.mov=(self.mov-movBL)/np.sqrt(movBL)
        else:
            self.mov=(self.mov-movBL)/movBL
            
        self.mov=self.mov[padbefore:len(movBL)-padafter,:,:]; 
        print 'Final Size Movie:' +  np.str(self.mov.shape)          
        
    
        
        
   
            
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
            t,h,w=self.mov.shape
            newshape=(int(w*fy),int(h*fx))
            mov=[];
            print(newshape)
            for frame in self.mov:                
                mov.append(cv2.resize(frame,newshape,fx=fx,fy=fy,interpolation=interpolation))
            self.mov=np.asarray(mov)
        if fz!=1:
            print "reshaping along z"            
            t,h,w=self.mov.shape
            self.mov=np.reshape(self.mov,(t,h*w))
            self.mov=cv2.resize(self.mov,(h*w,int(fz*t)),fx=1,fy=fz,interpolation=interpolation)
#            self.mov=cv2.resize(self.mov,(h*w,int(fz*t)),fx=1,fy=fz,interpolation=interpolation)
            self.mov=np.reshape(self.mov,(int(fz*t),h,w))
            self.frameRate=self.frameRate/fz