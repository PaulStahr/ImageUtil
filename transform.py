import glob, sys
import numpy as np
from joblib import Parallel, delayed
from enum import Enum
import multiprocessing
import imageio
import os
from matplotlib import cm
import pyexr


class CmdMode(Enum):
    UNDEF = 0
    INPUT = 1
    SCALARINPUT = 2
    OUTPUT = 3
    FUNCTION = 4


def create_parent_directory(filename):
    dirname = os.path.dirname(filename)
    if not os.path.exists(dirname):
        os.makedirs(dirname)


# def set_resolution(colormap, N):
#    colors=cm.gnuplot(np.linspace(0,1,cm.gnuplot.N))
#    {'red':np.asarray((np.linspace(0,1,len(colors)),colors[:,0],colors[:,0])).T,
#    'green':np.asarray((np.linspace(0,1,len(colors)),colors[:,0],colors[:,0])).T,
#    'blue':np.asarray((np.linspace(0,1,len(colors)),colors[:,0],colors[:,0])).T}
#    return cm.LinearSegmentedColormap(colom.ap.name, colors, N)

class Opts():
    def __init__(self):
        self.low = 0
        self.high = 1


def highres(colmap, data):
    dataf = (data * colmap.N)
    datai = dataf.astype(int)
    mod = (dataf - datai)[:, :, None]
    return colmap(datai) * (1 - mod) + colmap(datai + 1) * mod


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


def process_frame(filenames, scalfilenames, scalarfolder, outputs, opts, logging):
    if scalfilenames is not None and len(scalfilenames) != len(filenames):
        raise Exception("Different lengths in filename and scalfilenames", len(scalfilenames), len(filenames))
    for i in range(len(filenames)):
        if scalfilenames is not None and len(scalfilenames[i]) != len(filenames[i]):
            raise Exception("Different lengths in filename and scalfilenames", len(scalfilenames[i]), len(filenames[i]))
        for j in range(len(filenames[i])):
            filename = filenames[i][j]
            print(filename)
            base = os.path.splitext(os.path.basename(filename))[0]
            img = read_image(filname)
            args = {'np': np, 'img': img, 'cm': cm, 'highres': highres, 'opts': opts}
            if scalarfolder is not None:
                with open(scalarfolder + '/' + base + ".txt") as file:
                    args['scal'] = float(file.readline())
            if scalfilenames is not None:
                with open(scalfilenames[i][j], "r") as file:
                    args['scal'] = float(file.readline())
            for output in outputs:
                out = eval(output[0], args)
                try:
                    filename = output[1] + '/' + base + output[2]
                    create_parent_directory(filename)
                    write_fimage(filename, out)
                except Exception as ex:
                    print("Can't write image", ex)


def process_frames(inputs, scalarinputs, scalarfolder, outputs, opts, logging):
    njobs = len(inputs)
    num_cores = multiprocessing.cpu_count()
    factor = (njobs + num_cores - 1)
    Parallel(n_jobs=num_cores)(delayed(process_frame)(inputs[i * factor:min((i + 1) * factor, njobs)],
                                                      None if scalarinputs is None else scalarinputs[
                                                                                        i * factor:min((i + 1) * factor,
                                                                                                       njobs)],
                                                      scalarfolder, outputs, opts, logging) for i in range(num_cores))


inputs = []
scalarinputs = []
outputs = []
scalarfolder = None
scalebarout = None
opts = Opts()
i = 1
cmdMode = CmdMode.UNDEF
logging = 0
while i < len(sys.argv):
    arg = sys.argv[i]
    if arg == "--input":
        cmdMode = CmdMode.INPUT
    elif arg == "--scalarinput":
        cmdMode = CmdMode.SCALARINPUT
    elif arg == "--output":
        outputs.append((compile(sys.argv[i + 1], '<string>', 'eval'), sys.argv[i + 2], sys.argv[i + 3]))
        cmdMode = CmdMode.UNDEF
        i += 3
    elif arg == "--alphamask":
        opts.alphamask = imageio.imread(sys.argv[i + 1])
        i += 1
    elif arg == "--cmap":
        cmdMode = CmdMode.UNDEF
    elif arg == "--scalarfolder":
        scalarfolder = sys.argv[i + 1]
        cmdMode = CmdMode.UNDEF
        i += 1
    elif arg == "--scalebarout":
        scalebarout = (sys.argv[i + 1], sys.argv[i + 2])
        cmdMode = CmdMode.UNDEF
        i += 2
    elif arg == "--range":
        opts.low = float(sys.argv[i + 1])
        opts.high = float(sys.argv[i + 2])
        cmdMode = CmdMode.UNDEF
        i += 2
    elif cmdMode == CmdMode.INPUT:
        inputs.append(glob.glob(arg))
    elif cmdMode == CmdMode.SCALARINPUT:
        scalarinputs.append(glob.glob(arg))
    elif arg == "--help":
        print("--input <input files>")
        print("--output <formular> <folder> <filetype>")
    else:
        raise Exception('Invalid input', arg)
    i += 1
if len(scalarinputs) == 0:
    for i in range(len(inputs)):
        inputs[i].sort()
else:
    for i in range(len(inputs)):
        inputs[i].sort()
        scalarinputs[i].sort()

if scalebarout is not None:
    colormap = eval(compile(scalebarout[0], '<string>', 'eval'), {'cm': cm})
    result = (np.asarray([colormap(np.linspace(0, 1, colormap.N))]) * 0xFF).astype(np.uint8)
    print(result.shape)
    write_fimage(scalebarout[1], result)
process_frames(inputs, None if len(scalarinputs) == 0 else scalarinputs, scalarfolder, outputs, opts, logging)
