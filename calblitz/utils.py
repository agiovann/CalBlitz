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
#%%
def playMatrix(mov,gain=1.0,frate=.033):
    for frame in mov: 
        cv2.imshow('frame',frame*gain)
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

