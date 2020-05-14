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

def process_frame(filenames, mode, criteria, expression, offset, logging):
    if len(filenames) == 0:
        return None
    if logging < 0:
        print(filenames)
    added = None
    accepted = []
    for file in filenames:
        img = imageio.imread(file)
        if criteria is not None:
            if not eval(criteria):
                continue
        if expression is not None:
            img = eval(expression)
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
        accepted.append(file)
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

def process_frames(filenames, mode, criteria, expression, offset, logging):
    num_cores = multiprocessing.cpu_count()
    factor = (len(filenames) + num_cores - 1) // num_cores
    image_list=Parallel(n_jobs=num_cores)(delayed(process_frame)(filenames[i * factor:min((i + 1) * factor,len(filenames))], mode, criteria, expression, offset, logging) for i in range(num_cores))
    accepted = []
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
expression = None
if expression_index != -1:
    expression = compile(sys.argv[expression_index + 1], '<string>', 'eval')

criteria_index = get_index(sys.argv, "criteria")
criteria = None
if criteria_index != -1:
    criteria = compile(sys.argv[criteria_index + 1], '<string>', 'eval')

input_idx = get_index(sys.argv, "input")
for arg in sys.argv[input_idx + 1:]:
    filenames = filenames + glob.glob(arg)
filenames.sort()

show = get_index(sys.argv, "showplot") != -1

if logging < 0:
    print(filenames)
preimage = None
if premode is not None:
    preimage, accepted=process_frames(filenames, premode, criteria, expression, None, logging)
    print("accepted",accepted)
    preimage /= len(accepted)
    if show:
        plt.imshow(preimage)
        plt.show()
    print("preimage ", preimage.shape);
image = None
if mode is not None:
    print("second pass")
    image, accepted = process_frames(filenames, mode, criteria, expression, preimage, logging)
    if mode == Mode.VARIANCE_NORMALIZED or mode == Mode.AVERAGE_NORMALIZED:
        image = LA.norm(image, axis=2)
else:
    image = preimage
print("accepted",len(accepted),"of",len(filenames))

coutput_idx = get_index(sys.argv, "coutput")
if coutput_idx != -1:
    coutput = sys.argv[coutput_idx + 1]
    print(coutput , len(accepted))
    file = open(coutput,"w")
    file.write(str(len(accepted)))
    file.close()

noutput_idx = get_index(sys.argv, "noutput")
if noutput_idx != -1:
    noutput = sys.argv[noutput_idx + 1]
    print(noutput , len(accepted))
    file = open(noutput,"w")
    for file in accepted:
        file.write(os.path.splitext(file)[0])
    file.close()

moutput_idx = get_index(sys.argv, "moutput")
if moutput_idx != -1:
    moutput = sys.argv[moutput_idx + 1]
    print(moutput , len(accepted))
    indices = []
    for afile in accepted:
        indices.append(int(os.path.splitext(ntpath.basename(afile))[0]))
    indices = np.asarray(indices) + 1
    indices=np.sort(indices)
    file = open(moutput,"w")
    for idx in indices:
        file.write(str(idx) + '\n')
    file.close()

if image is not None:
    doutput_idx = get_index(sys.argv, "doutput")
    image = np.where(np.isnan(image), np.zeros_like(image), image)
    if doutput_idx != -1:
        doutput_file = sys.argv[doutput_idx + 1]
        if len(accepted) != 0:
            divided = image / len(accepted)
            imageio.imwrite(doutput_file,(divided).astype(np.float32))

    output_idx = get_index(sys.argv, "output")
    if output_idx != -1:
        output_file = sys.argv[output_idx + 1]
        imageio.imwrite(output_file,(image).astype(np.float32))
        #image=imageio.imread(output_file)
        #print(np.min(image), np.max(image))
        #if show:
        #    plt.imshow(image / 255)
        #    plt.show()

        #imageio.imwrite(sys.argv[output_idx + 1],image.astype(np.uint8))
