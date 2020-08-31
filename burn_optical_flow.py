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
import util
from matplotlib import cm
import pyexr


def read_numbers(filename):
    with open(filename, 'r') as f:
        return np.asarray([int(x) for x in f])


def create_parent_directory(filename):
    dirname = os.path.dirname(filename)
    if not os.path.exists(dirname):
        os.makedirs(dirname)


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


def process_frame(imagefile, flowfile, po):
    basename = ntpath.basename(imagefile)
    img = None
    if imagefile is not None:
        img = imageio.imread(imagefile)
    name = ntpath.splitext(basename)[0];
    flow = imageio.imread(flowfile)
    img2 = np.zeros((*flow.shape[0:2], 3), dtype=np.uint8)
    cv2.circle(img2, (img2.shape[1] // 2, img2.shape[0] // 2), img2.shape[0] // 2, (255, 255, 255))
    scalar = np.asarray(img2.shape, dtype=float)[0:2] / np.asarray((po.rows, po.rows), dtype=float)
    if not np.isfinite(flow).all():
        raise ("error \"" + flowfile + "\" contains illegal values")
    img2[:, :, 0] = (flow[:, :, 0] ** 2 + flow[:, :, 1] ** 2 < 0.1) * 255
    img2[:, :, 1] = (flow[:, :, 0] ** 2 + flow[:, :, 1] ** 2 < 0.01) * 255
    arrows = np.zeros((po.rows * po.rows, 4))
    for x in range(po.rows):
        for y in range(po.rows):
            startpoint = scalar * np.asarray((x, y), dtype=float)
            index = startpoint.astype(int)
            fl = np.dot(po.transform, np.asarray((*flow[(index[1], index[0])][0:2], 1)))
            endpoint = startpoint + fl
            arrows[x * po.rows + y, :] = (*(startpoint / np.asarray(img2.shape[0:2])), *fl)
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
        # filename = po.absout2 + "/" + name + ".tif"
        filename = po.absout2 + "/" + name + ".exr"
        create_parent_directory(filename)
        pyexr.write(filename, np.sqrt(np.sum(flow[:, :, 0:2] ** 2, axis=2)))
        # imageio.imwrite(filename, np.sqrt(np.sum(flow[:, :, 0:2] ** 2, axis=2)))
    if po.absout3 is not None:
        # filename = po.absout3 + "/" + name + ".tif"
        filename = po.absout3 + "/" + name + ".exr"
        create_parent_directory(filename)
        pyexr.write(filename, np.sqrt(np.sum(flow ** 2, axis=2)))
        # imageio.imwrite(filename, np.sqrt(np.sum(flow ** 2, axis=2)))
    if po.cabsout2 is not None:
        # plt.imsave (po.cabsout2 + "/" + basename + ".png", np.sqrt(np.sum(flow[:,:,0:2] ** 2, axis=2)), cmap=cm.gnuplot)
        abs2 = np.sqrt(np.sum(flow[:, :, 0:2] ** 2, axis=2))
        if po.cmax is None:
            abs2 /= np.max(abs2)
        else:
            abs2 /= po.cmax
        # print(cm.gnuplot(abs2).astype(np.uint16).shape)
        # print(np.max(np.max(cm.gnuplot(abs2)[:,:,0:3], axis=0),axis=0))
        filename = po.cabsout2 + "/" + name + ".png"
        create_parent_directory(filename)
        colmap = cm.gnuplot

        # Workaround to create plots with higher color-resolution
        # colmap.N=512
        abs2f = (abs2 * colmap.N)
        abs2i = abs2f.astype(int)
        mod = (abs2f - abs2i)[:, :, None]
        plot = colmap(abs2i) * (1 - mod) + colmap(abs2i + 1) * mod
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
        po.cmax = float(sys.argv[i + 1])
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
        po.transform = np.asarray([int(x) for x in sys.argv[i + 1:i + 7]]).reshape(2, 3)
        i += 6
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
