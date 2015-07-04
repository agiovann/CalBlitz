# CalBlitz
Blazing fast calcium imaging analysis toolbox

## Synopsis
This projects implements a set of essential methods required in the calcium imaging movies analysis pipeline. Fast and scalable algorithms are implemented for motion correction, movie manipulation and roi segmentation. It is assumed that movies are collecetd with the scanimage data acquisition software and stored in tif format

## Code Example

```
#%%
from XMovie import XMovie
from pylab import plt
import numpy as np

#%% define movie
filename='your_file.tif'
m=XMovie(filename, frameRate=.033);

#%% motion correct run 1 time
template,shift=m.motion_correct(max_shift=max_shift,template=None,show_movie=False);

#%% compute delta f over f DF/F
m.computeDFF(secsWindow=5,quantilMin=20,subtract_minimum=True)

#%% compute spatial components via ICA PCA
spcomps=m.IPCA_stICA(components=50);

#%% extract ROIs from spatial components 
masks=m.extractROIsFromPCAICA(spcomps, numSTD=8, gaussiansigmax=2 , gaussiansigmay=2)

#%%  extract single ROIs from each mask
allMasks=[np.array(mm==ll) for mm in masks  for ll in xrange(np.max(mm)) if 2000>np.sum(np.array(mm==ll))>10   ]
allMasks=np.asarray(allMasks,dtype=np.float16)

#%% example plot a frame
plt.imshow(m.mov[100],cmap=plt.cm.Greys_r)

#%% example play movie
m.playMovie(frate=.1,gain=6.0,magnification=1)

```


## Motivation

Recent advances in calcium imaging acquisition techniques are creating datasets of the order of Terabytes/week. Memory and computationally efficient algorithms are required to analyze in reasonable amount of time terabytes of data.  

## Installation

requirements

anaconda python

conda install scipy 
conda install matplotlib
conda install PIL 
conda install ipython 
conda install pims
conda install scikit­learn
conda install opencv (for windows installation check (http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_setup/py_setup_in_windows/py_setup_in_windows.html)



Provide code examples and explanations of how to get the project.

## API Reference

TODO CREATE API REFERENCE

## Tests

Open the ExamplePipeline.py file and run cell by cell

## Contributors

Andrea Giovannucci 
Ben Deverett
Chad Giusti


## License

## Troubleshooting

Depending on terminal program used anaconda may not be in default path
in this case, add anaconda to bin to path: export PATH=//anaconda/bin:$PATH

Error: No packages found in current osx­64 channels matching: pims


­ install pip: conda install pip
­ use pip to install pims: pip install pims
­ if pims causes kernel crash then use pip install pims ­­upgrade


­ remove from motion_correction import MotionCorrector
