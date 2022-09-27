import numpy as np
import imageio
import os
from numpy import linalg
import pyexr
import Imath
import sys

#Checks if given image is a 1 degree rotation around z-axis

def read_image(file):
    img = None
    if file.endswith(".exr"):
        img = pyexr.read(file)
    else:
        img = imageio.v2.imread(file)
    if len(img.shape) == 2:
        img = img[..., None]
    return img

test = read_image(sys.argv[1])
expected = read_image(sys.argv[2])

#Create a grid of coordinates, where each entry states the coordinate of the cells-center. This should match most opengl-implementations
grid = np.meshgrid(np.linspace(-1+1/test.shape[0],1+1/test.shape[0],test.shape[0],endpoint=False),np.linspace(-1+1/test.shape[1],1+1/test.shape[1],test.shape[1],endpoint=False))
#Elevation in a spherical equidistant coordinate system ranging from 0 to 1
elev = np.sqrt(np.square(grid[0]) + np.square(grid[1]))
#We are only interested in pixels which don't have more that 90 degree elevation
mask = elev < 1
expected = expected * mask[:,:,np.newaxis]
test = test * mask[:,:,np.newaxis]
maxdiff = np.max(np.abs(expected - test))/np.max(expected)

#As the implementation currently uses half floats with a mantissa of 10 this should result in an error of about 2^(-10)=0.0009765625
print(maxdiff)
if not maxdiff < 2**(-10)*2:
    raise Exception(maxdiff, "higher than expected", 2^(-10)*2)
