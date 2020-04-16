import cv2
import numpy as np
import imageio
import os
import matplotlib.pyplot as plt
import glob, sys
from collections import deque
import csv
import ntpath
from PIL import Image, ImageDraw
import multiprocessing
from joblib import Parallel, delayed
from numpy import linalg as LA
from enum import Enum
import OpenEXR, array, Imath

class Mode(Enum):
    AVERAGE = 0
    AVERAGE_NORMALIZED = 1
    VARIANCE = 2
    VARIANCE_NORMALIZED = 3
    VARIANCE_ARC = 4

def get_index(arg,name):
    try:
        return arg.index(name)
    except:
        return -1

def process_frame(filenames, cap, mode, expression, offset, logging):
    if len(filenames) == 0:
        return None
    if logging < 0:
        print(filenames)
    added = None
    accepted = 0
    for file in filenames:
        img = imageio.imread(file)
        if expression is not None:
            img = eval(expression)
        if (img == 0).all():
            if logging < 1:
                print("complete black frame " + file)
        elif not np.isfinite(img).all():
            if logging < 1:
                print("not finite values in " + file)
        else:
            if np.max(np.abs(img)) > cap:
                print("value cap exceeded in " + file)
            else:
                img = img.astype(float)
                if mode == Mode.AVERAGE_NORMALIZED or mode == Mode.VARIANCE_ARC:
                    #if np.min(np.abs(LA.norm(img, axis=2))) == 0:
                    #    plt.imshow(LA.norm(img, axis=2))
                    #    plt.show()
                    img /= LA.norm(img, axis=2)[...,None]
                    #plt.imshow(img*0.5 + 0.5)
                    #plt.show()
                if offset is not None:
                    img -= offset
                accepted += 1
                if mode == Mode.AVERAGE or mode == Mode.AVERAGE_NORMALIZED:
                    to_add=img
                elif mode == Mode.VARIANCE or mode == Mode.VARIANCE_NORMALIZED:
                    to_add=img ** 2
                elif mode == Mode.VARIANCE_ARC:
                    to_add=2*np.arcsin(LA.norm(img, axis=2) / 2)
                if added is not None:
                    added += to_add
                else:
                    added = to_add
    return (added, accepted)

def process_frames(filenames, cap, mode, expression, offset, logging):
    num_cores = multiprocessing.cpu_count()
    factor = (len(filenames) + num_cores - 1) // num_cores
    image_list=Parallel(n_jobs=num_cores)(delayed(process_frame)(filenames[i * factor:min((i + 1) * factor,len(filenames))], cap, mode, expression, offset, logging) for i in range(num_cores))
    accepted = 0
    added = None
    for img in image_list:
        if img is not None and img[0] is not None:
            if added is not None:
                added+=img[0]
                accepted += img[1]
            else:
                added=img[0]
                accepted += img[1]
    return (added, accepted)

imageio.plugins.freeimage.download()
filenames = []
verbose_idx = get_index(sys.argv, "logging")
logging = 0
if verbose_idx != -1:
    logging=int(sys.argv[verbose_idx +1])
if logging < 1:
    print(sys.argv)

output_idx = get_index(sys.argv, "output")
if output_idx == -1:
    print("no output specified")
    exit()

cap_idx = get_index(sys.argv, "cap")
cap = np.inf
if cap_idx != -1:
    cap = int(sys.argv[cap_idx + 1])
divide = True
divide_idx = get_index(sys.argv, "divide")
if divide_idx != -1:
    divide = sys.argv[divide_idx + 1].lower() in ("true", "yes", "1")
mode_idx = get_index(sys.argv, "mode")
premode = None
mode = Mode.AVERAGE
if mode_idx != -1:
    value = sys.argv[mode_idx + 1]
    if value == "average":
        premode = None
        mode = Mode.AVERAGE
    elif value == "averagem":
        premode = None
        mode = Mode.AVERAGE_NORMALIZED
    elif value == "variance":
        premode = Mode.AVERAGE
        mode = Mode.VARIANCE
    elif value == "variancem":
        premode = Mode.AVERAGE_NORMALIZED
        mode = Mode.VARIANCE_NORMALIZED
    elif value == "variancea":
        premode = Mode.AVERAGE_NORMALIZED
        mode = Mode.VARIANCE_ARC
    else:
        print("Value ",value," not known")
        
expression_index = get_index(sys.argv, "expression")
if expression_index != -1:
    expression = compile(sys.argv[expression_index + 1], '<string>', 'eval')

input_idx = get_index(sys.argv, "input")
for arg in sys.argv[input_idx + 1:]:
    filenames = filenames + glob.glob(arg)
filenames.sort()

show = get_index(sys.argv, "showplot") != -1

if logging < 0:
    print(filenames)
preimage = None
if premode is not None:
    preimage, accepted=process_frames(filenames, cap, premode, expression, None, logging)
    print("accepted",accepted)
    preimage /= accepted
    if show:
        plt.imshow(preimage)
        plt.show()
    print("preimage ", preimage.shape);
image = None
if mode is not None:
    print("second pass")
    image, accepted = process_frames(filenames, cap, mode, expression, preimage, logging)
    if divide:
        image /= accepted
    if mode == Mode.VARIANCE_NORMALIZED or mode == Mode.AVERAGE_NORMALIZED:
        image = LA.norm(image, axis=2)
else:
    image = preimage
print("image ", image.shape)
print("accepted",accepted,"of",len(filenames))


coutput_idx = get_index(sys.argv, "coutput")
if coutput_idx != -1:
    coutput = sys.argv[coutput_idx + 1]
    print(coutput , accepted)
    file = open(coutput,"w")
    file.write(str(accepted))
    file.close()
output_file = sys.argv[output_idx + 1]

#if output_file.endswith('.exr'):
#    HEADER = OpenEXR.Header(*image.shape[0:2])
#    float_chan = Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))
#    HEADER['channels'] = dict([(c, float_chan) for c in "RGB"])
#    exr = OpenEXR.OutputFile(output_file, HEADER)
#    r = array.array('f', (image[:,:,0].flatten()).astype(np.float32)).tostring()
#    g = array.array('f', (image[:,:,1].flatten()).astype(np.float32)).tostring()
#    b = array.array('f', (image[:,:,2].flatten()).astype(np.float32)).tostring()
#    exr.writePixels({'R': r, 'G': g, 'B': b})
#else:
image = np.where(np.isnan(image), np.zeros_like(image), image)
imageio.imwrite(sys.argv[output_idx + 1],(image).astype(np.float32))
image=imageio.imread(sys.argv[output_idx + 1])
print(np.min(image), np.max(image))
if show:
    plt.imshow(image / 255)
    plt.show()

#imageio.imwrite(sys.argv[output_idx + 1],image.astype(np.uint8))
