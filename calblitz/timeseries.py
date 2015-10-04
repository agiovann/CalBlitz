# -*- coding: utf-8 -*-
"""
Class representing a time series.

    Example of usage        
    
    Parameters
    ----------
    input_arr: np.ndarray 
    start_time: time beginning movie
    fr: frame rate
    meta_data: dictionary including any custom meta data


author: Andrea Giovannucci
"""
#%%
import os
import warnings
import numpy as np

import pylab as plt
try: 
    plt.ion()
except:
    1;


#%%
class timeseries(np.ndarray):
    """ 
    Class representing a time series.

    Example of usage        
    
    Parameters
    ----------
    input_arr: np.ndarray 
    start_time: time beginning movie
    fr: frame rate
    meta_data: dictionary including any custom meta data

    """
    
    def __new__(cls, input_arr, start_time=None,fr=None,meta_data=None,file_name=None):
        #         
        
        if fr is None or start_time is None:        
            raise Exception('You need to specify both the start time and frame rate')
        
        # case we load movie from file
       
            
        obj = np.asarray(input_arr).view(cls)
        # add the new attribute to the created instance
                
        obj.start_time = start_time*1.
        obj.fr = fr*1.
        if type(file_name) is list:
            obj.file_name = file_name
        else:
            obj.file_name = [file_name];
        
        if type(meta_data) is list:
            obj.meta_data = meta_data
        else:
            obj.meta_data = [meta_data];    
        
        
        return obj
        
    @property
    def time(self):
        return np.linspace(self.start_time,1/self.fr*self.shape[0],self.shape[0])
        
    def __array_prepare__(self, out_arr, context=None):
#        print 'In __array_prepare__:'
#        print '   self is %s' % type(self)
#        print '   out_arr is %s' % type(out_arr)
        inputs=context[1];
        frRef=None;
        startRef=None;
        for inp in inputs:
            if type(inp) is timeseries:
                if frRef is None:
                    frRef=inp.fr
                else:
                    if not (frRef-inp.fr) == 0:
                        raise ValueError('Frame rates of input vectors do not match. You cannot perform operations on time series with different frame rates.') 
                if startRef is None:
                    startRef=inp.start_time
                else:
                    if not (startRef-inp.start_time) == 0:
                        warnings.warn('start_time of input vectors do not match: ignore if this is what desired.',UserWarning) 
        
        # then just call the parent
        return np.ndarray.__array_prepare__(self, out_arr, context)
            
    def __array_finalize__(self, obj):
        # see InfoArray.__array_finalize__ for comments
        if obj is None: return               
        
        self.start_time = getattr(obj, 'start_time', None)
        self.fr = getattr(obj, 'fr', None)
        self.file_name = getattr(obj, 'file_name', None)
        self.meta_data = getattr(obj, 'meta_data', None)
    
#    def __array_wrap__(self, out_arr, context=None):
#        print 'In __array_wrap__:'
#        print '   self is %s' % type(self)
#        print '   arr is %s' % type(out_arr)
#        print context
#        # then just call the parent
#        return np.ndarray.__array_wrap__(self, out_arr, context)    
#    
    
        
    def save(self,file_name):
        extension = os.path.splitext(file_name)[1]
            
        if extension == '.tif': # load avi file
            raise Exception('not implemented')
        elif extension == '.npz':
            np.savez(file_name,input_arr=self, start_time=self.start_time,fr=self.fr,meta_data=self.meta_data,file_name=self.file_name)
        elif extension == 'avi':
            raise Exception('not implemented')
        elif extension == '.mat':
            raise Exception('not implemented')
        else:
            print extension
            raise Exception('Extension Unknown')
    
                

def concatenate(*args,**kwargs):
        """
        Append the movie mov to the object
        
        Parameters
        ---------------------------
        mov: XMovie object
        """                  
        
        obj=[];  
        frRef=None;
        for arg in args:
            for m in arg:
                if issubclass(type(m),timeseries):                    
                    if frRef is None:
                        obj=m;
                        frRef=obj.fr;
                    else:
                        print('added')
                        obj.__dict__['file_name'].extend([ls for ls in m.file_name])
                        obj.__dict__['meta_data'].extend([ls for ls in m.meta_data])
                        if obj.fr != m.fr:
                            raise ValueError('Frame rates of input vectors do not match. You cannot concatenate movies with different frame rates.') 
        print obj
        return obj.__class__(np.concatenate(*args,**kwargs),**obj.__dict__)   
        
        
