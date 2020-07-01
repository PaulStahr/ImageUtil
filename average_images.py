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
#import OpenEXR, array, Imath

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

def process_frame(filenames, mode, criteria, expression, extract, offset, logging):
    if len(filenames) == 0:
        return None
    if logging < 0:
        print(filenames)
    added = None
    accepted = []
    extracted = []
    for file in filenames:
        img = imageio.imread(file)
        if criteria is not None:
            if not eval(criteria):
                continue
        if expression is not None:
            img = eval(expression)
        img = img.astype(float)
        if mode == Mode.VARIANCE_ARC and img.shape[2] != 3:
            raise Exception("Shape of image is completely wrong", img.shape)
        if mode == Mode.AVERAGE_NORMALIZED or mode == Mode.VARIANCE_ARC:
            div = LA.norm(img, axis=2)[...,None]
            if np.all(div == 0):
                raise Exception("Warning image ",str(file)," is completely zero")
            img /= div
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
        extracted.append(to_add[extract])
    return (added, accepted, extracted)

def process_frames(filenames, mode, criteria, expression, extract, offset, logging):
    num_cores = multiprocessing.cpu_count()
    factor = (len(filenames) + num_cores - 1) // num_cores
    image_list=Parallel(n_jobs=num_cores)(delayed(process_frame)(filenames[i * factor:min((i + 1) * factor,len(filenames))], mode, criteria, expression, extract, offset, logging) for i in range(num_cores))
    accepted = []
    extracted = []
    added = None
    for img in image_list:
        if img is not None and img[0] is not None:
            if added is not None:
                added+=img[0]
            else:
                added=img[0]
            accepted += img[1]
            extracted += img[2]
    return (added, accepted, extracted)

def read_numbers(filename):
    with open(filename,'r') as f:
         return np.asarray([int(x) for x in f])

def create_parent_directory(filename):
    dirname=os.path.dirname(filename)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

imageio.plugins.freeimage.download()
filenames = []
divide = True
premode = None
mode = Mode.AVERAGE
expression = None
criteria = None
show = False
logging = 0
coutput = None
noutput = None
moutput = None
preout = None
prein = None
eoutput = None
output = None
doutput = None
soutput = None
extract = []

i = 1
while i <len(sys.argv):
    arg = sys.argv[i]
    if arg == "mode":
        value = sys.argv[i + 1]
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
            raise Exception("Value ",value," not known")
        i += 1
    elif arg == "framelist" or arg == "mframelist":
        framelist = sys.argv[i + 1]
        numbers = None
        if framelist == "stdin":
            numbers = np.asarray([int(line) for line in sys.stdin],dtype=int)
        else:
            numbers = read_numbers(framelist)
        if arg == "mframelist":
            numbers = numbers - 1
        prefix = sys.argv[i + 2]
        suffix = sys.argv[i + 3]
        i += 3
        filenames = np.core.defchararray.add(np.core.defchararray.add(prefix,(read_numbers(framelist)-1).astype(str)),suffix)
    elif arg == "input":
        for arg in sys.argv[i + 1:]:
            filenames = filenames + glob.glob(arg)
        filenames.sort()
        break
    elif arg == "criteria":
        criteria = compile(sys.argv[i + 1], '<string>', 'eval')
        i += 1
    elif arg == "expression":
        expression = compile(sys.argv[i + 1], '<string>', 'eval')
        i += 1
    elif arg == "showplot":
        show = True
    elif arg == "logging":
        logging=int(sys.argv[i + 1])
        i += 1
    elif arg == "coutput":
        coutput = sys.argv[i + 1]
        i += 1
    elif arg == "noutput":
        noutput = sys.argv[i + 1]
        i += 1
    elif arg == "moutput":
        moutput = sys.argv[i + 1]
        i += 1
    elif arg == "doutput":
        doutput = sys.argv[i + 1]
        i += 1
    elif arg == "output":
        output = sys.argv[i + 1]
        i += 1
    elif arg == "eoutput":
        eoutput = sys.argv[i + 1]
        i += 1
    elif arg == "preout":
        preout = sys.argv[i + 1]
        i += 1
    elif arg == "soutput":
        soutput = sys.argv[i + 1]
        i += 1
    elif arg == "prein":
        prein = sys.argv[i + 1]
        i += 1
    elif arg == "extract":
        extract.append(np.fromstring(sys.argv[i + 1], sep=',', dtype=int))
        i += 1
    else:
        raise Exception("Unknown argument " + arg)
    i += 1
print("extract:",extract)
extract=(*np.asarray(extract).T,)
if len(extract) == 0:
    extract = ([],[])
print("extract:",extract)
if logging < 1:
    print(sys.argv)
if logging < 0:
    print(filenames)
preimage = None
if prein is not None:
    preimage = imageio.imread(prein)
elif premode is not None:
    preimage, accepted, extracted=process_frames(filenames, premode, criteria, expression, extract, None, logging)
    preimage /= len(accepted)
    #TODO: maybe normalize again for some measures?
    if show:
        plt.imshow(preimage)
        plt.show()
    if preout != None:
        imageio.imwrite(preout, preimage.astype(np.float32))
if eoutput is not None or coutput is not None or noutput is not None or moutput is not None or output is not None or doutput is not None:
    image = None
    if mode is not None:
        print("second pass")
        image, accepted, extracted = process_frames(filenames, mode, criteria, expression, extract, preimage, logging)
        if mode == Mode.VARIANCE_NORMALIZED or mode == Mode.AVERAGE_NORMALIZED:
            image = LA.norm(image, axis=2)
    else:
        image = preimage
    print("accepted",len(accepted),"of",len(filenames))

    if eoutput is not None:
        create_parent_directory(eoutput)
        np.savetxt(eoutput, extracted, delimiter=' ',fmt='%i')

    if coutput is not None:
        print(coutput , len(accepted))
        create_parent_directory(coutput)
        file = open(coutput,"w")
        file.write(str(len(accepted)))
        file.close()

    if noutput is not None:
        print(noutput , len(accepted))
        file = open(noutput,"w")
        for file in accepted:
            create_parent_directory(os.path.splitext(file)[0])
            file.write(os.path.splitext(file)[0])
        file.close()

    if moutput is not None:
        print(moutput , len(accepted))
        indices = []
        for afile in accepted:
            indices.append(int(os.path.splitext(ntpath.basename(afile))[0]))
        indices = np.asarray(indices) + 1
        indices=np.sort(indices)
        create_parent_directory(moutput)
        np.savetxt(moutput, indices, delimiter='\n',fmt='%i')
    #image = np.where(np.isnan(image), np.zeros_like(image), image)
    if image is not None:
        if doutput is not None:
            divided = image / len(accepted)
            create_parent_directory(doutput)
            imageio.imwrite(doutput,divided.astype(np.float32))
        if output is not None:
            create_parent_directory(output)
            imageio.imwrite(output,image.astype(np.float32))
        if soutput is not None:
            create_parent_directory(soutput)
            file = open(soutput, "w")
            file.write (str(np.sum(image)))
            file.close()
