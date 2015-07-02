# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 21:01:17 2015

@author: agiovann
"""

#%%
import cv2

import scipy.ndimage
import warnings
import numpy as np
#%%
def playMatrix(mov,gain=1.0,frate=.033):
    for frame in mov: 
        cv2.imshow('frame',frame*gain/np.max(frame))
        if cv2.waitKey(int(frate*1000)) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break  
    cv2.destroyAllWindows()        
        
#%%
#def computeDFF(mov,frameRate,maxshift,secsWindow=5,quantilMin=.8,):
#    """ 
#    compute the DFF of the movie
#    
#    """
#    assert(np.min(mov)>0),"All pixels must be nonnegative"
#       
#        
#    numFrames,linePerFrame,pixPerLine=np.shape(mov)
#    downsampfact=int(secsWindow/frameRate);
#    elm_missing=int(np.ceil(numFrames*1.0/downsampfact)*downsampfact-numFrames)
#    padbefore=np.floor(elm_missing/2.0)
#    padafter=np.ceil(elm_missing/2.0)
#    print 'Inizial Size Image:' + np.str(np.shape(mov))
#    mov=np.pad(mov,((padbefore,padafter),(0,0),(0,0)),mode='reflect')
#    numFramesNew,linePerFrame,pixPerLine=np.shape(mov)
#    maxshift=5;
#    #% compute baseline quickly
#    movBL=np.reshape(mov,(downsampfact,int(numFramesNew/downsampfact),linePerFrame,pixPerLine));
#    movBL=np.percentile(movBL,quantilMin,axis=0);
#    movBL=scipy.ndimage.zoom(np.array(movBL,dtype=np.float32),[downsampfact ,1, 1])
#    #%
#    movDFF=(mov-movBL)/np.sqrt(movBL)
#    movDFF=movDFF[padbefore:-padafter,maxshift:-maxshift,; 
#    print 'Final Size Movie:' +  np.str(movDFF.shape)
#    return movDFF

#%%
#filename='temp_mc_subpix_linear.tif'
#movMat=np.array(pims.open(filename))
#movMat=np.swapaxes(movMat,1,2)
#movMat=movMat[:,:,::-1]
#movDFF=computeDFF(movMat,frameRate,5)
##%%
#playMovie(movDFF,frate=frameRate)
##%%
#
#m=XMovie(filename,frameRate)
#m=m.motioncorrect;
#
#
#
#m=m.saveTiff()
#m=m.saveMat()
#
#
#
#    
