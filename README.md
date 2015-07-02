# CalBlitz
Blazing fast calcium imaging analysis toolbox

## Synopsis
This projects implements a set of essential methods required in the calcium imaging movies analysis pipeline. Fast and scalable algorithms are implemented for motion correction, movie manipulation and roi segmentation. It is assumed that movies are ccollecetd with the scanimage data acquisition software


## Code Example

load movie in memory 

m=XMovie('movie_name.tif', frameRate=.033);

correct movie 

template,shift=m.motion_correct(max_shift=max_shift,template=None,show_movie=False);

create DFF version of the movie 

m=m.computeDFF(secsWindow=5,quantilMin=8,subtract_minimum=True)

Perform iterative PCA ICA with 50 spatial components extracted

spcomps=m.IPCA_stICA(components=50);

Get regions of interest from spatial components

masks=m.extractROIsFromPCAICA(spcomps, numSTD=5, gaussiansigmax=2 , gaussiansigmay=2)


## Motivation

Recent advances in calcium imaging acquisition techniques are creating datasets of the order of Terabytes/week. Memory and computationally algorithms are required to 

## Installation

requirements

anaconda python

conda install scipy 
conda install matplotlib
conda install PIL 
conda install ipython 
conda install pims
conda install opencv


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

