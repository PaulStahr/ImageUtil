import numpy as np
import imageio
import glob, sys
from enum import Enum
import matplotlib.pyplot as plt

class CmdMode(Enum):
    UNDEF = 0
    INPUT = 1
    HELP = 2

class Projection(Enum):
    SPHERICAL_EQUIDISTANT_RADIAL = 0
    SPHERICAL_EQUIDISTANT_DEGREE = 1
    SPHERICAL_EQUISOLID_RADIAL = 2
    SPHERICAL_EQUISOLID_DEGREE = 3
    SPHERICAL_EQUIRECTANGULAR_RADIAL = 4
    SPHERICAL_EQUIRECTANGULAR_DEGREE = 5

def process_frame(filenames, proj, expression, logging):
    for i in range(len(filenames)):
        filename = filenames[i]
        img = imageio.imread(filename)
        if expression is not None:
            img = eval(expression)
        xv, yv = np.mgrid[-img.shape[0]/2:img.shape[0]/2,-img.shape[1]/2:img.shape[1]/2]
        xv /= img.shape[0]/2
        yv /= img.shape[1]/2
        if proj == Projection.SPHERICAL_EQUIDISTANT_RADIAL or proj == Projection.SPHERICAL_EQUIDISTANT_DEGREE:
            arc = np.sqrt(xv**2+yv**2)*np.pi/2
            stamp=np.sin(arc)/arc
            stamp=np.where(arc<np.pi/2,stamp,0)
            stamp*=np.pi**2/(img.shape[0]*img.shape[1])
            if proj == Projection.SPHERICAL_EQUIDISTANT_DEGREE:
                stamp *= (180/np.pi)**2
            img=np.ones_like(img)
            print(img.reshape(-1,img.shape[-1]).T.dot(stamp.flatten()))

proj = None
expression = None
logging = 0
i = 1
cmdMode = None
inputs=[]
while i < len(sys.argv):
    arg = sys.argv[i]
    if arg == "--expression":
        expression = compile(sys.argv[i + 1], '<string>', 'eval')
        i += 2
    elif arg == "--input":
        cmdMode = CmdMode.INPUT
        i += 1
    elif arg == "--logging":
        logging = int(sys.argv[i + 1])
        i += 2
    elif arg == "--help":
        print("projections:")
        print("speqdr","SPHERICAL_EQUIDISTANT_RADIAL")
        print("speqdd","SPHERICAL_EQUIDISTANT_DEGREE")
        i += 1
    elif arg == "--projection":
        value = sys.argv[i + 1]
        if value == "speqdr":
            proj = Projection.SPHERICAL_EQUIDISTANT_RADIAL
        if value == "speqdd":
            proj = Projection.SPHERICAL_EQUIDISTANT_DEGREE
        i += 2
    elif cmdMode == CmdMode.INPUT:
        inputs+=glob.glob(arg)
        i += 1
    else:
        raise Exception('Invalid input', arg)
process_frame(inputs, proj, expression, logging)
