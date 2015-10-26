# CalBlitz
**Blazing fast** calcium imaging analysis toolbox

# PLEASE BE ADVISED THAT FOR THE NEXT FEW DAYS THE CODE IS UNDER RECONSTRUCTION. THIS MESSAGE WILL DISAPPEAR WHEN STABLE AGAIN. 

## Synopsis
This projects implements a set of essential methods required in the calcium imaging movies analysis pipeline. **Fast and scalable algorithms** are implemented for motion correction, movie manipulation and roi segmentation. It is assumed that movies are collected with the scanimage data acquisition software and stored in *.tif* format. Find below a schematic of the calcium imaging pipeline:

![Alt text](images/CaImagingPipeline.png?raw=true "calcium imaging pipeline")

## Code Example

```python
#%% add CalBlitz folder to python directory
path_to_CalBlitz_folder='/home/ubuntu/SOFTWARE/CalBlitz'
path_to_CalBlitz_folder='C:/Users/agiovann/Documents/SOFTWARE/CalBlitz/CalBlitz'

import sys
sys.path
sys.path.append(path_to_CalBlitz_folder)
#% add required packages
import calblitz as cb
import time
import pylab as pl
import numpy as np
#% cdset basic ipython functionalities
#try: 
#    pl.ion()
#    %load_ext autoreload
#    %autoreload 2
#except:
#    print "Probably not a Ipython interactive environment" 


#%% define movie
filename='M_FLUO.tif'
frameRate=15.25; # in Hz
start_time=0;

#%%
filename_py=filename[:-4]+'.npz'
filename_mc=filename[:-4]+'_mc.npz'
filename_analysis=filename[:-4]+'_analysis.npz'    
filename_traces=filename[:-4]+'_traces.npz'    

#%% load movie
m=cb.movie(filename, fr=frameRate,start_time=start_time);

#%% example plot a frame
pl.imshow(m[100],cmap=pl.cm.Greys_r)

#%% example play movie
m.play(fr=20,gain=1.0,magnification=1)

#%%
m.save(filename_py)

#%%
m=cb.movie.load(filename_py); 


#%% concatenate movies (it will add to the original movie)
# you have to create another movie new_mov=XMovie(...)
# cb.concatenate([m,new_mov])

#%% motion correct run 3 times


templates=[];
shifts=[];
corrs=[]
max_shift_w=5;
max_shift_h=5;
num_iter=3; # numer of times motion correction is executed
template=None # here you can use your own template (best representation of the FOV)
print np.min(m)
for j in range(0,num_iter):
    m,template_used,shift,corrs=m.motion_correct(max_shift_w=max_shift_w,max_shift_h=max_shift_h,template=None,show_movie=False);
    templates.append(template_used)
    shift=np.asarray(shift)
    shifts.append(shift)
    corrs.append(corrs)
    
pl.plot(np.asarray(shifts).reshape((j+1)*shift.shape[0],shift.shape[1]))

#%%
cb.matrixMontage(np.asarray(templates),cmap=pl.cm.gray,vmin=0,vmax=1000)

#%% apply shifts to original movie in order to minimize smoothing

m=cb.movie.load(filename_py); 
m=m.crop(crop_top=0,crop_bottom=1,crop_left=0,crop_right=0,crop_begin=0,crop_end=0);
totalShifts=np.sum(np.asarray(shifts),axis=0)[:,0:2].tolist()
m=m.apply_shifts(totalShifts)
    
   

#%% plot movie median
minBrightness=20;
maxBrightness=500;
pl.imshow(np.median(m,axis=0),cmap=pl.cm.Greys_r,vmin=minBrightness,vmax=maxBrightness)

#%% save motion corrected movie inpython format along with the results. This takes some time now but will save  a lot later...
np.savez(filename_mc,templates=templates,shifts=shifts,max_shift_h=max_shift_h,max_shift_w=max_shift_w)

#%% RELOAD MOTION CORRECTED MOVIE
m=cb.movie.load(filename_py); 
shifts=np.load(filename_mc)['shifts']
totalShifts=np.sum(np.asarray(shifts),axis=0)[:,0:2].tolist()
m=m.apply_shifts(totalShifts) 
max_h,max_w= np.max(totalShifts,axis=0)
min_h,min_w= np.min(totalShifts,axis=0)
m=m.crop(crop_top=max_h,crop_bottom=-min_h+1,crop_left=max_w,crop_right=-min_w,crop_begin=0,crop_end=0)


#%% resize to increase SNR and have better convergence of segmentation algorithms
resizeMovie=True
if resizeMovie:
    fx=.5; # downsample a factor of four along x axis
    fy=.5;
    fz=.2; # downsample  a factor of 5 across time dimension
    m=m.resize(fx=fx,fy=fy,fz=fz)
else:
    fx,fy,fz=1,1,1

#%% compute delta f over sqrt(f) (DF/sqrt(F)) movie
initTime=time.time()
m=m.computeDFF(secsWindow=10,quantilMin=50)
print 'elapsed time:' + str(time.time()-initTime) 



#%% compute spatial components via NMF
initTime=time.time()
space_spcomps,time_comps=(m-np.min(m)).NonnegativeMatrixFactorization(n_components=20,beta=1,tol=5e-7);
print 'elapsed time:' + str(time.time()-initTime) 
cb.matrixMontage(np.asarray(space_spcomps),cmap=pl.cm.gray) # visualize components

#%% compute spatial components via ICA PCA
from scipy.stats import mode
initTime=time.time()
spcomps=m.IPCA_stICA(components=20,mu=.5,batch=10000);
print 'elapsed time:' + str(time.time()-initTime) 
cb.matrixMontage(spcomps,cmap=pl.cm.gray) # visualize components
 
#%% extract ROIs from spatial components 
#_masks,masks_grouped=m.extractROIsFromPCAICA(spcomps, numSTD=6, gaussiansigmax=2 , gaussiansigmay=2)
_masks,_=m.extractROIsFromPCAICA(spcomps, numSTD=10.0, gaussiansigmax=0 , gaussiansigmay=0)
cb.matrixMontage(np.asarray(_masks),cmap=pl.cm.gray)

#%%  extract single ROIs from each mask
minPixels=10;
maxPixels=2500;
masks_tmp=[];
for mask in _masks:
    numPixels=np.sum(np.array(mask));        
    if (numPixels>minPixels and numPixels<maxPixels):
        print numPixels
        masks_tmp.append(mask>0)
        
masks_tmp=np.asarray(masks_tmp,dtype=np.float16)

# reshape dendrites if required(if the movie was resized)
if fx != 1 or fy !=1:
    mdend=cb.movie(np.asarray(masks_tmp,dtype=np.float32), fr=1,start_time=0);
    mdend=mdend.resize(fx=1/fx,fy=1/fy)
    all_masks=mdend;
else:
    all_masks=masks_tmp              

all_masksForPlot=[kk*(ii+1)*1.0 for ii,kk in enumerate(all_masks)]
pl.imshow(np.max(np.asarray(all_masksForPlot,dtype=np.float16),axis=0))


#%% extract DF/F from orginal movie, needs to reload the motion corrected movie
m=cb.movie.load(filename_py); 
m=m.crop(crop_top=0,crop_bottom=1,crop_left=0,crop_right=0,crop_begin=0,crop_end=0);
shifts=np.load(filename_mc)['shifts']
totalShifts=np.sum(np.asarray(shifts),axis=0)[:,0:2].tolist()
m=m.apply_shifts(totalShifts) 
max_h,max_w= np.max(totalShifts,axis=0)
min_h,min_w= np.min(totalShifts,axis=0)
m=m.crop(crop_top=max_h,crop_bottom=-min_h+1,crop_left=max_w,crop_right=-min_w,crop_begin=0,crop_end=0)
minPercentileRemove=.1;
# remove an estimation of what a Dark patch is, you should provide a better estimate
F0=np.percentile(m,minPercentileRemove)
m=m-F0; 
traces = m.extract_traces_from_masks(all_masks)
#,type='DFF',window_sec=15,minQuantile=8
traces=traces.computeDFF(window_sec=1,minQuantile=8);
traces.plot()

```


## Motivation

Recent advances in calcium imaging acquisition techniques are creating datasets of the order of Terabytes/week. Memory and computationally efficient algorithms are required to analyze in reasonable amount of time terabytes of data.  

## Installation

###Prerequisites

install anaconda python distribution, then in your terminal type

```
conda install scipy 
conda install matplotlib
conda install PIL 
conda install ipython 
conda install pip
pip install pims
conda install scikit­-learn #(or pip install scikit-learn)
conda install opencv #(this will not work on windows)
conda install CVXOPT

```
For opencv windows installation check [here]( 
http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_setup/py_setup_in_windows/py_setup_in_windows.html)

If you have problems installing opencv remember to match your architecture (32/64 bits) and to make sure that you have the required libraries installed

### Install the package

clone the git package 

git clone https://github.com/agiovann/CalBlitz.git

or download the zipped version 
 
 

Add the CalBlitz folder to your Python path (or call the script from within the library). We suggest to use spyder to run the example code in *ExamplePipeline.py*. Each [code cell](https://pythonhosted.org/spyder/editor.html#how-to-define-a-code-cell) is a unit that should be run and the result inspected. This package is supposed to be used interactively, like in [MATLAB](http://www.mathworks.com).   

## API Reference

TODO 

## Tests

Open the ExamplePipeline.py file and run cell by cell

## Contributors

Andrea Giovannucci, Ben Deverett, Chad Giusti


## License

This program is free software; you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation; either version 2 of the License, or (at your option) any later version.

This program is distributed WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not, see <http://www.gnu.org/licenses/>.

## Troubleshooting

1. Depending on terminal program used anaconda may not be in default path. In this case, add anaconda to bin to path: 

```
export PATH=//anaconda/bin:$PATH
```

2. Error: No packages found in current osx­64 channels matching: pims
 install pip: conda install pip
­ use pip to install pims: pip install pims
­ if pims causes kernel crash then use 

 ```
 pip install pims ­­--upgrade
 ```
 
3. If you get another compile time error installing pims, install the following 

[Microsoft C++ ompiler for Python](https://www.microsoft.com/en-us/download/confirmation.aspx?id=44266)

by typing 

``` 
msiexec /i <path to downloaded MSI File> ALLUSERS=1 
```


­
