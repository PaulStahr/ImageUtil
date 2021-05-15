import glob
import sys
import numpy as np
from joblib import Parallel, delayed
from enum import Enum
import multiprocessing
import imageio
import os
from matplotlib import cm
import pyexr
import png
import colorsys


class CmdMode(Enum):
    UNDEF = 0
    INPUT = 1
    SCALARINPUT = 2
    OUTPUT = 3
    FUNCTION = 4


class Transformation(Enum):
    EQUIDISTANT = 0
    EQUIDISTANT_HALF = 1

def divlim(divident, divisor):
    return np.divide(divident, divisor, np.ones_like(divident), where=np.logical_or(divident!=0,divisor!=0))


def highdensity(img, p, transformation = None):
    permutation = np.argsort(img.flatten())
    cumsum = None
    if transformation == None:
        cumsum = np.cumsum(img[permutation])
    elif transformation == "spherical-equidistant":
        x = None
        y = None
        if len(img.shape) == 2:
            x, y = np.mgrid[0:img.shape[0], 0:img.shape[1]]
        elif len(img.shape) == 3:
            x, y, z = np.mgrid[0:img.shape[0], 0:img.shape[1], 0:img.shape[2]]
        else:
            raise Exception("Unsopported image-dimension", len(img.shape))
        r = np.sqrt(((x*2-img.shape[0])/img.shape[0])**2+((y*2-img.shape[1])/img.shape[1])**2)*np.pi * 0.5
        cumsum = np.cumsum((img * divlim(np.sin(r), r)).flatten()[permutation])
    else:
        raise Exception("Unknown transformation", transformation)
    idx = np.searchsorted(cumsum, cumsum[-1] * (1-p))
    return img > img.flatten()[permutation][idx]


def create_parent_directory(filename):
    dirname = os.path.dirname(filename)
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def write_numbers(filename, list):
    file = open(filename, "w")
    for elem in list:
        file.write(str(elem) + '\n')
    file.close()


def cmyk_to_rgb(c, m, y, k, cmyk_scale, rgb_scale=255):
    r = rgb_scale * (1.0 - c / float(cmyk_scale)) * (1.0 - k / float(cmyk_scale))
    g = rgb_scale * (1.0 - m / float(cmyk_scale)) * (1.0 - k / float(cmyk_scale))
    b = rgb_scale * (1.0 - y / float(cmyk_scale)) * (1.0 - k / float(cmyk_scale))
    return r, g, b


def cmyk_to_rgb2(c,m,y,k):
    r = rgb_scale*(1.0-(c+k)/float(cmyk_scale))
    g = rgb_scale*(1.0-(m+k)/float(cmyk_scale))
    b = rgb_scale*(1.0-(y+k)/float(cmyk_scale))
    return r,g,b


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
        self.alphamask = None
        self.transformation = None


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
    elif len(img.shape) == 3 and img.shape[2] == 2 and filename.endswith(".png"):
        w = png.Writer(img.shape[1], img.shape[0], alpha=True, greyscale=True)
        with open(filename, 'wb') as f:
            w.write(f, img[:, :, 0:2].reshape((img.shape[0], -1)))
    else:
        imageio.imwrite(filename, img)


def equi2cart(x, y):
    radius = np.sqrt(x ** 2 + y ** 2)
    radian = radius * np.pi
    sin = np.sin(radian)
    return np.asarray((divlim(sin * x, radius), divlim(sin * y, radius), np.cos(radian)))


def cart2equi(x, y, z):
    length = np.sqrt(x * x + y * y);
    length = np.arctan2(length, z) / (length * np.pi);
    return np.asarray((x * length, y * length))


def process_frame(filenames, scalfilenames, scalarfolder, outputs, distoutputs, opts, logging):
    if scalfilenames is not None and len(scalfilenames) != len(filenames):
        raise Exception("Different lengths in filename and scalfilenames", len(filenames), len(scalfilenames))
    for i in range(len(filenames)):
        if scalfilenames is not None and len(scalfilenames[i]) != len(filenames[i]):
            raise Exception("Different lengths in filename and scalfilenames", len(filenames[i]), len(scalfilenames[i]))
        for j in range(len(filenames[i])):
            filename = filenames[i][j]
            print(filename)
            base = os.path.splitext(os.path.basename(filename))[0]
            img = read_image(filename)
            x, y, z = np.mgrid[0:img.shape[0], 0:img.shape[1], 0:img.shape[2]]
            ds = None
            if opts.transformation == Transformation.EQUIDISTANT or opts.transformation == Transformation.EQUIDISTANT_HALF:
                tmp = np.sqrt(((x*2-img.shape[0])/img.shape[0])**2+((y*2-img.shape[1])/img.shape[1])**2) * np.pi
                if opts.transformation == Transformation.EQUIDISTANT_HALF:
                    tmp /= 2
                ds = divlim(np.sin(tmp),tmp)
            args = {'np': np, 'ds': ds, 'equi2cart': equi2cart, 'cart2equi': cart2equi, 'img': img, 'cm': cm, 'highres': highres, 'opts': opts, 'min': np.min(img), 'max': np.max(img), 'highdensity': highdensity, 'x': x, 'y': y, 'z': z, 'xf': x/img.shape[0], 'yf': y/img.shape[1], 'zf':z/img.shape[2], 'rf': np.sqrt(((x*2-img.shape[0])/img.shape[0])**2+((y*2-img.shape[1])/img.shape[1])**2), 'divlim': divlim, 'colorsys': colorsys, 'cmyk_to_rgb': cmyk_to_rgb, 'cmyk_to_rgb2': cmyk_to_rgb2}
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
            for output in distoutputs:
                out = eval(output[0], args)
                filename = output[1] + '/' + base + output[2]
                create_parent_directory(filename)
                if len(out.shape) == 2:
                    np.savetxt(filename, out, delimiter=' ', fmt='%f')
                else:
                    write_numbers(filename, out)


def process_frames(inputs, scalarinputs, scalarfolder, outputs, distoutputs, opts, logging):
    njobs = len(inputs)
    if njobs == 0:
        return
    num_cores = multiprocessing.cpu_count()
    factor = (njobs + num_cores - 1)
    Parallel(n_jobs=num_cores)(delayed(process_frame)(inputs[i * factor:min((i + 1) * factor, njobs)],
                                                      None if scalarinputs is None else scalarinputs[
                                                                                        i * factor:min((i + 1) * factor,
                                                                                                       njobs)],
                                                      scalarfolder, outputs, distoutputs, opts, logging) for i in range(num_cores))


inputs = []
scalarinputs = []
outputs = []
distoutputs = []
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
    elif arg == "--distoutput":
        distoutputs.append((compile(sys.argv[i + 1], '<string>', 'eval'), sys.argv[i + 2], sys.argv[i + 3]))
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
    elif arg == "--transformation":
        if sys.argv[i + 1] == "equidistant-half":
            opts.transformation = Transformation.EQUIDISTANT_HALF
        elif sys.argv[i + 1] == "equidistant":
            opts.transformation = Transformation.EQUIDISTANT
        else:
            raise Exception('Unknown transformation', sys.argv[i + 1])
        i += 1
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
    create_parent_directory(scalebarout[1])
    write_fimage(scalebarout[1], result)
process_frames(inputs, None if len(scalarinputs) == 0 else scalarinputs, scalarfolder, outputs, distoutputs, opts, logging)
