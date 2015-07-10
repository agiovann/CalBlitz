# CalBlitz
**Blazing fast** calcium imaging analysis toolbox

## Synopsis
This projects implements a set of essential methods required in the calcium imaging movies analysis pipeline. **Fast and scalable algorithms** are implemented for motion correction, movie manipulation and roi segmentation. It is assumed that movies are collected with the scanimage data acquisition software and stored in *.tif* format. Find below a schematic of the calcium imaging pipeline:

![Alt text](images/CaImagingPipeline.png?raw=true "calcium imaging pipeline")

## Code Example

```python
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