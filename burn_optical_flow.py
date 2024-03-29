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
from matplotlib import cm
import pyexr
import OpenEXR
import Imath

def read_numbers(filename):
    with open(filename, 'r') as f:
        return np.asarray([int(x) for x in f])


def create_parent_directory(filename):
    dirname = os.path.dirname(filename)
    if not os.path.exists(dirname):
        try:
            os.makedirs(dirname)
        except FileExistsError:
            pass
        except:
            raise


def highres(colmap, data):
    dataf = (data * colmap.N)
    datai = dataf.astype(int)
    mod = (dataf - datai)[:, :, None]
    return colmap(datai) * (1 - mod) + colmap(datai + 1) * mod


class ProcessingOptions:
    def __init__(self):
        self.imagefolder = ""
        self.flowfolder = ""
        self.imageout = None
        self.arrowout = None
        self.tableout = None
        self.absout2 = None
        self.absout3 = None
        self.cabsout2 = None
        self.cmax = None
        self.rows = 11
        self.transform = np.asarray([[1, 0, 0], [0, 1, 0]])
        self.anchor_minimum = False


def read_image(file):
    img = None
    if file.endswith(".exr"):
        img = pyexr.read(file)
    else:
        img = imageio.imread(file)
    if len(img.shape) == 2:
        img = img[..., None]
    return img

def write_fimage(filename, img):
    if filename.endswith(".exr"):
        if len(img.shape) == 2 or img.shape[2] == 1:
            header = OpenEXR.Header(*img.shape[0:2])
            header['channels'] = {'Y': Imath.Channel(Imath.PixelType(OpenEXR.FLOAT))}
            out = OpenEXR.OutputFile(filename, header)
            out.writePixels({'Y': img})
            out.close()
        else:
            pyexr.write(filename, img)
    else:
        imageio.imwrite(filename, img)

def process_frame(imagefile, flowfile, po):
    basename = ntpath.basename(imagefile)
    img = None
    if imagefile is not None:
        img = read_image(imagefile)
    name = ntpath.splitext(basename)[0];
    flow = read_image(flowfile)
    img2 = np.zeros((*flow.shape[0:2], 3), dtype=np.uint8)
    shape = np.asarray(img2.shape)
    cv2.circle(img2, (shape[1] // 2, shape[0] // 2), shape[0] // 2, (255, 255, 255))
    scalar = np.asarray(shape, dtype=float)[0:2] / np.asarray((po.rows, po.rows), dtype=float)
    if not np.isfinite(flow).all():
        raise Exception("error \"" + flowfile + "\" contains illegal values")
    abs2 = np.sqrt(np.sum(flow[:, :, 0:2] ** 2, axis=2))
    img2[:, :, 0] = (abs2 < 0.01) * 255
    img2[:, :, 1] = np.logical_and(0.00001 < abs2, abs2 < 0.001) * 255
    img2[:, :, 2] = (abs2 < 0.0001) * 255
    arrows = np.zeros((po.rows * po.rows, 4))
    offset = (0, 0)
    if po.anchor_minimum:
        argmin = np.unravel_index(np.argmin(abs2 + 10000*(abs2 == 0)), abs2.shape)
        offset = np.mod(argmin, abs2.shape / np.full(2, po.rows))
    for x in range(po.rows):
        for y in range(po.rows):
            startpoint = scalar * np.asarray((x, y), dtype=float) + offset
            index = startpoint.astype(int)
            fl = np.dot(po.transform, np.asarray((*flow[(index[1], index[0])][0:2], 1)))
            endpoint = startpoint + fl
            arrows[x * po.rows + y, :] = (*(startpoint / shape[0:2]), *fl)
            startpoint = tuple(startpoint.astype(int))
            endpoint = tuple(endpoint.astype(int))
            if img is not None:
                img = cv2.arrowedLine(img, startpoint, endpoint, (0, 0, 0), 2)
            img2 = cv2.arrowedLine(img2, startpoint, endpoint, (255, 255, 255), 2)
    if po.imageout is not None:
        filename = po.imageout + "/" + basename
        create_parent_directory(filename)
        cv2.imwrite(filename, img)
    if po.arrowout is not None:
        filename = po.arrowout + "/" + name + ".png"
        create_parent_directory(filename)
        cv2.imwrite(filename, img2)
    if po.tableout is not None:
        filename = po.tableout + "/" + name + ".csv"
        create_parent_directory(filename)
        np.savetxt(filename, arrows)
    if po.absout2 is not None:
        filename = po.absout2 + "/" + name + ".exr"
        create_parent_directory(filename)
        write_fimage(filename, np.sqrt(np.sum(flow[:, :, 0:2] ** 2, axis=2)))
    if po.absout3 is not None:
        filename = po.absout3 + "/" + name + ".exr"
        create_parent_directory(filename)
        write_fimage(filename, np.sqrt(np.sum(flow ** 2, axis=2)))
    if po.cabsout2 is not None:
        if po.cmax is None:
            abs2 /= np.max(abs2)
        else:
            abs2 /= po.cmax
        filename = po.cabsout2 + "/" + name + ".png"
        create_parent_directory(filename)
        colmap = cm.gnuplot
        plot = highres(colmap, abs2)
        imageio.imwrite(filename, (plot[:, :, 0:3] * 0xFF).astype(np.uint8))
        # imageio.imwrite(filename, (plot[:,:,0:3]*0xFFFF).astype(np.uint16), 'PNG-FI')


imagenames = []
flownames = []

i = 1
po = ProcessingOptions()
while i < len(sys.argv):
    arg = sys.argv[i]
    if arg == "mframelist" or arg == "framelist":
        framelist = sys.argv[i + 1]
        imgprefix = sys.argv[i + 2]
        imgsuffix = sys.argv[i + 3]
        flowprefix = sys.argv[i + 4]
        flowsuffix = sys.argv[i + 5]
        i += 5
        numbers = read_numbers(framelist)
        if arg == "mframelist":
            numbers -= 1
        imagenames = np.core.defchararray.add(np.core.defchararray.add(imgprefix, numbers.astype(str)), imgsuffix)
        flownames = np.core.defchararray.add(np.core.defchararray.add(flowprefix, numbers.astype(str)), flowsuffix)
    elif arg == "input":
        for arg in sys.argv[i + 1:]:
            filenames = filenames + glob.glob(arg)
        filenames.sort()
        break
    elif arg == "single":
        imagenames = [sys.argv[i + 1]]
        flownames = [sys.argv[i + 2]]
        i += 2
    elif arg == "rows":
        rows = int(sys.argv[i + 1])
        i += 1
    elif arg == "absout2":
        po.absout2 = sys.argv[i + 1]
        i += 1
    elif arg == "cabsout2":
        po.cabsout2 = sys.argv[i + 1]
        i += 1
    elif arg == "cmax":
        po.cmax = float(eval(sys.argv[i + 1]))
        i += 1
    elif arg == "absout3":
        po.absout3 = sys.argv[i + 1]
        i += 1
    elif arg == "imgout":
        po.imageout = sys.argv[i + 1]
        i += 1
    elif arg == "arrowout":
        po.arrowout = sys.argv[i + 1]
        i += 1
    elif arg == "tableout":
        po.tableout = sys.argv[i + 1]
        i += 1
    elif arg == "transform":
        po.transform = np.asarray([float(eval(x)) for x in sys.argv[i + 1:i + 7]]).reshape(2, 3)
        i += 6
    elif arg == "anchormin":
        po.anchor_minimum = True
    elif arg == "help":
        print(sys.argv[
                  0] + " <Input Filenames> <ImageInputFolder> <FlowInputFolder> <ImageArrowOutput> <ArrowOutput> <TextOutput>")
    else:
        raise Exception("Unknown argument", arg)
    i += 1
imageio.plugins.freeimage.download()
num_cores = multiprocessing.cpu_count()
Parallel(n_jobs=num_cores)(
    delayed(process_frame)(imagenames[i] if imagenames is not None else None, flownames[i], po) for i in
    range(len(flownames)))
