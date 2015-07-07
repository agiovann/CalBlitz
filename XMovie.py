# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 09:02:06 2015

@author: agiovann
"""
#%%
#%load_ext autoreload
#%autoreload 2
import cv2
import pims
import scipy.ndimage
import warnings
import numpy as np
from sklearn.decomposition import IncrementalPCA, FastICA
import pylab as plt
plt.ion()
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage import label

from libtiff import TIFF, TiffFile
import sys
import functools
#%%

    
#%%
class XMovie(object):
    """ 
    Class representing a movie 

    Parameters
    ----------
    filename : name of the tiff file
    frameRate : float, inter frame interval in seconds
    """
    
    def __init__(self, filename=None, mat=None, frameRate=None):
        
        if not filename == None:
            self.mov = np.array(pims.open(filename))
            self.mov = np.swapaxes(self.mov,1,2)
            self.mov=self.mov[:,:,::-1]
            self.filename = filename
        elif np.any(mat):
             self.mov=mat
        else: 
            raise Exception('You need to specify either matrix or filename')
        
        if not frameRate==None:
            self.frameRate = frameRate
        else:
            raise Exception('You need to specify the frame rate')
    
#    def create(self,itemType,itemName,*args,**kwargs):       
#        print "Creating Item %s with name %s, args %r and kwargs %r" % (itemType,itemName,args,kwargs)
#
#    def __getattr__(self,attrName):        
#        try: 
#            return eval('self.mov.'+ attrName)
#        except:          
##            return eval('self.mov.'+ attrName)
#            return functools.partial(self.create,attrName)
#    def create(self,itemType,itemName,*args,**kwargs):       
#        print type(itemType),type(itemName),type(args),type(kwargs)
#
#    def __getattr__(self,attrName):        
#        try: 
#            return eval('self.mov.'+ attrName)
#        except:          
##            return eval('self.mov.'+ attrName)
#            return functools.partial(self.create,attrName)            
        
        
#    def __call__(self,*args,**kw):
#        print args
        
    def motion_correct(self, max_shift=5, show_movie=False,template=None):
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
        
        ms = max_shift
        if template == None:
            template=np.median(self.mov,axis=0)            
            template=template[ms:h_i-ms,ms:w_i-ms].astype(np.float32)
            
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
        
             if (0 < top_left[1] < 2 * ms-1) & (0 < top_left[0] < 2 * ms-1):
                 # if max is internal, check for subpixel shift using gaussian
                 # peak registration
                 log_xm1_y = np.log(res[sh_x-1,sh_y]);             
                 log_xp1_y = np.log(res[sh_x+1,sh_y]);             
                 log_x_ym1 = np.log(res[sh_x,sh_y-1]);             
                 log_x_yp1 = np.log(res[sh_x,sh_y+1]);             
                 four_log_xy = 4*np.log(res[sh_x,sh_y]);
    
                 sh_x_n = -(sh_x - ms + (log_xm1_y - log_xp1_y) / (2 * log_xm1_y - four_log_xy + 2 * log_xp1_y))
                 sh_y_n = -(sh_y - ms + (log_x_ym1 - log_x_yp1) / (2 * log_x_ym1 - four_log_xy + 2 * log_x_yp1))
             else:
                 sh_x_n = -(sh_x - ms)
                 sh_y_n = -(sh_y - ms)
                     
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
#             mc=MotionCorrector(mov=np.float32(self.mov), max_shift=max_shift, show_movie=False, sub_pixel=True)
#             templates,shifts = mc.correct(n_iters)
#             newmov=XMovie(mat=mc.get_motcor_mov(),frameRate=self.frameRate)
#             return newmov,shifts,templates
             

    def makeSubMov(self,frames):        
        mm=self.mov[frames,:,:]
        frate=self.frameRate
        return XMovie(mat=mm,frameRate=frate)
    
        
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
        
        eigenframes = np.dot(proj_frame_samples, eigenseries)

        return eigenseries, eigenframes, proj_frame_vectors        
    
    def IPCA_stICA(self, components = 50, batch = 1000, mu = 0.05, ICAfun = 'logcosh'):
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
    
        eigenseries, eigenframes = self.IPCA(components, batch)
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
    
        ind_frames = joint_ics[:frame_size, :]
        ind_frames = np.reshape(ind_frames.T, (components, h, w))
        
        return ind_frames  

    def IPCA_denoise(self, components = 50, batch = 1000):
        _, _, clean_vectors = self.IPCA(components, batch)
        self.mov = np.reshape(clean_vectors.T, np.shape(self.mov))
                
        
    def compute_StructuredNMFactorization(self):
        print "to do"
        
   
    def local_correlations(self):
         # Output:
         #   rho M x N matrix, cross-correlation with adjacent pixel

         rho = np.zeros(np.shape(self.mov)[1:3])
         w_mov = (self.mov - np.mean(self.mov, axis = 0))/np.std(self.mov, axis = 0)
 
         rho_h = np.mean(np.multiply(w_mov[:,:-1,:], w_mov[:,1:,:]), axis = 0)
         rho_w = np.mean(np.multiply(w_mov[:,:,:-1], w_mov[:,:,1:,]), axis = 0)

         rho[:-1,:] = rho[:-1,:] + rho_h
         rho[1:,:] = rho[1:,:] + rho_h
         rho[:,:-1] = rho[:,:-1] + rho_w
         rho[:,1:] = rho[:,1:] + rho_w

         neighbors = 4 * np.ones(np.shape(self.mov)[1:3])        
         neighbors[0,:] = neighbors[0,:] - 1;
         neighbors[-1,:] = neighbors[-1,:] - 1;
         neighbors[:,0] = neighbors[:,0] - 1;
         neighbors[:,-1] = neighbors[:,-1] - 1;

         rho = np.divide(rho, neighbors)

         return rho
    
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
        
    def crop(self,crop_left,crop_right,crop_top,crop_bottom,crop_begin=0,crop_end=1):
        """ Crop movie        
        """
        self.mov=self.mov[crop_begin:-crop_end,crop_left:-crop_right,crop_top:-crop_bottom]
        
        
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
        
    def extractROIsFromPCAICA(self,spcomps, numSTD=4, gaussiansigmax=2 , gaussiansigmay=2):
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
        #%%
        mask=[];
        for k in xrange(0,numcomps):
            comp=spcomps[k]
            plt.subplot(rowcols,rowcols,k+1)
            comp=gaussian_filter(comp,[gaussiansigmay,gaussiansigmax])
            
            maxc=np.percentile(comp,99);
            minc=np.percentile(comp,1);
            comp=np.sign(maxc-np.abs(minc))*comp;
            q75, q25 = np.percentile(comp, [75 ,25])
            iqr = q75 - q25
            minCompValue=np.median(comp)+numSTD*iqr/1.35;            
            compabs=comp*(comp>minCompValue);
            #height, width = compabs.shape
            labeled, n = label(compabs>0, np.ones((3,3)))
            mask.append(labeled) 
            plt.imshow(labeled)                             
            plt.axis('off')         
        return mask
        
        
    def save_mov(self, filename, shifts=None, zoom=False, zoom_too=False):
         print "to be implemented"
#        if zoom:
#            self.mov = zoom(self.mov, [0.2, 1, 1])
#        tifffile = TIFF.open(filename+'.tif', mode='w')
#        tifffile.write_image(self.mov)
#        tifffile.close()
#        if zoom_too:
#            self.mov = szoom(self.mov, [0.2, 1, 1])
#            tifffile = TIFF.open(filename+'_z_.tif', mode='w')
#            tifffile.write_image(self.mov)
#            tifffile.close()
            
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
            newshape=(int(h*fx),int(w*fy))
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
            self.mov=np.reshape(self.mov,(int(fz*t),w,h))
            self.frameRate=self.frameRate/fz