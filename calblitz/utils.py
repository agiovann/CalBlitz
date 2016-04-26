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
from pylab import plt
from tempfile import NamedTemporaryFile
from IPython.display import HTML
import calblitz as cb
import numpy as np
from ipyparallel import Client
#%%
def playMatrix(mov,gain=1.0,frate=.033):
    for frame in mov: 
        if gain!=1:
            cv2.imshow('frame',frame*gain)
        else:
            cv2.imshow('frame',frame)
            
        if cv2.waitKey(int(frate*1000)) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break  
    cv2.destroyAllWindows()        
#%% montage
def matrixMontage(spcomps,*args, **kwargs):
    numcomps, width, height=spcomps.shape
    rowcols=int(np.ceil(np.sqrt(numcomps)));           
    for k,comp in enumerate(spcomps):        
        plt.subplot(rowcols,rowcols,k+1)       
        plt.imshow(comp,*args, **kwargs)                             
        plt.axis('off')         
        
#%% CVX OPT
#####    LOOK AT THIS! https://github.com/cvxgrp/cvxpy/blob/master/examples/qcqp.py
if False:
    from cvxopt import matrix, solvers
    A = matrix([ [ .3, -.4,  -.2,  -.4,  1.3 ],
                     [ .6, 1.2, -1.7,   .3,  -.3 ],
                     [-.3,  .0,   .6, -1.2, -2.0 ] ])
    b = matrix([ 1.5, .0, -1.2, -.7, .0])
    m, n = A.size
    I = matrix(0.0, (n,n))
    I[::n+1] = 1.0
    G = matrix([-I, matrix(0.0, (1,n)), I])
    h = matrix(n*[0.0] + [1.0] + n*[0.0])
    dims = {'l': n, 'q': [n+1], 's': []}
    x = solvers.coneqp(A.T*A, -A.T*b, G, h, dims)['x']
    print(x)    
    #%%
    from scipy.signal import lfilter
    dt=0.016;
    t=np.arange(0,10,dt)
    lambda_=dt*1;
    tau=.17;
    sigmaNoise=.1;
    tfilt=np.arange(0,4,dt);
    spikes=np.random.poisson(lam=lambda_,size=t.shape);
    print(np.sum(spikes))
    filtExp=np.exp(-tfilt/tau);
    simTraceCa=lfilter(filtExp,1,spikes);
    simTraceFluo=simTraceCa+np.random.normal(loc=0, scale=sigmaNoise,size=np.shape(simTraceCa));
    plt.plot(t,simTraceCa,'g')
    plt.plot(t,spikes,'r')
    plt.plot(t,simTraceFluo)           
    
      
    #%%
    #trtest=tracesDFF.D_5
    #simTraceFluo=trtest.Data';
    #dt=trtest.frameRate;
    #t=trtest.Time;
    #tau=.21;
    
    #%%
    gam=(1-dt/tau);
    numSamples=np.shape(simTraceFluo)[-1];
    G=np.diag(np.repeat(-gam,numSamples-1),-1) + np.diag(np.repeat(1,numSamples));
    A_2=- np.diag(np.repeat(1,numSamples));
    Aeq1=np.concatenate((G,A_2),axis=1);
    beq=np.hstack((simTraceFluo[0],np.zeros(numSamples-1)));
    A1=np.hstack((np.zeros(numSamples), -np.ones(numSamples)));
    
    
    #%%
    T = np.size(G,0);
    oness=np.ones((T));
    sqthr=np.sqrt(thr);
    #y(y<(mean(y(:)-3*std(y(:)))))=0;
#%%

VIDEO_TAG = """<video controls>
 <source src="data:video/x-m4v;base64,{0}" type="video/mp4">
 Your browser does not support the video tag.
</video>"""

def anim_to_html(anim,fps=20):
    if not hasattr(anim, '_encoded_video'):
        with NamedTemporaryFile(suffix='.mp4') as f:
            anim.save(f.name, fps=fps, extra_args=['-vcodec', 'libx264'])
            video = open(f.name, "rb").read()
        anim._encoded_video = video.encode("base64")
    
    return VIDEO_TAG.format(anim._encoded_video)
    

def display_animation(anim,fps=20):
    plt.close(anim._fig)
    return HTML(anim_to_html(anim,fps=fps))    

#%%
def motion_correct_parallel(file_names,fr,template=None,margins_out=0,max_shift_w=5, max_shift_h=5,remove_blanks=False,apply_smooth=True,backend='single_thread'):
    """motion correct many movies usingthe ipyparallel cluster
    Parameters
    ----------
    file_names: list of strings
        names of he files to be motion corrected
    fr: double
        fr parameters for calcblitz movie 
    margins_out: int
        number of pixels to remove from the borders    
    
    Return
    ------
    base file names of the motion corrected files
    """
    args_in=[];
    for f in file_names:
        args_in.append((f,fr,margins_out,template,max_shift_w, max_shift_h,remove_blanks,apply_smooth))
        
#    try:
    if backend is 'ipyparallel':
        c = Client()   
        dview=c[:]
        file_res = dview.map_sync(process_movie_parallel, args_in)                         
    elif backend is 'single_thread':
        file_res = map(process_movie_parallel, args_in)                         
    else:
        raise Exception('Unknown backend')
#    except:
#        raise
#    finally:
#        if backend is 'ipyparallel':
#            dview.results.clear()       
#            c.purge_results('all')
#            c.purge_everything()
#            c.close()
        
            
    return file_res


    
def process_movie_parallel(arg_in):
#    import calblitz
#    import calblitz.movies
    import ca_source_extraction as cse
    import calblitz as cb
    import numpy as np
    import sys
    

    

    fname,fr,margins_out,template,max_shift_w, max_shift_h,remove_blanks,apply_smooth=arg_in
    
    with open(fname[:-4]+'.stout', "a") as log:
        sys.stdout = log
        
    #    import pdb
    #    pdb.set_trace()
        Yr=cb.load(fname,fr=fr)
        print 'loaded'    
        if apply_smooth:
            print 'applying smoothing'
            Yr=Yr.bilateral_blur_2D(diameter=10,sigmaColor=10000,sigmaSpace=0)
            
        Yr=Yr-np.float32(np.percentile(Yr,1))     # needed to remove baseline
        print 'Remove BL'
        if margins_out!=0:
            Yr=Yr[:,margins_out:-margins_out,margins_out:-margins_out] # borders create troubles
        print 'motion correcting'
        Yr,shifts,xcorrs,template=Yr.motion_correct(max_shift_w=max_shift_w, max_shift_h=max_shift_h,  method='opencv',template=template,remove_blanks=remove_blanks) 
        print 'median computing'        
        template=Yr.bin_median()
        print 'saving'         
        Yr.save(fname[:-3]+'hdf5')        
        print 'saving 2'                 
        np.savez(fname[:-3]+'npz',shifts=shifts,xcorrs=xcorrs,template=template)
        print 'deleting'        
        del Yr
        print 'done!'
        sys.stdout = sys.__stdout__ 
        
    return fname[:-3]        