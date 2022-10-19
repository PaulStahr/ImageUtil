import numpy as np
import imageio
import os
import glob
import sys
import ntpath
from numpy import linalg
from enum import Enum
import pyexr
import OpenEXR
import Imath
import time


class Transformation(Enum):
    EQUIDISTANT = 0
    EQUIDISTANT_HALF = 1
    CYLINDRICAL_EQUIDISTANT = 2
    PROJECT = 3
    PERSPECTIVE = 4


class Opts:
    def __init__(self):
        self.extract = []
        self.sextract = []
        self.display_process = -1
        self.alphamask = None
        self.transformation = None
        self.check = None
        self.criteria = None


class Mode(Enum):
    AVERAGE = 0
    AVERAGE_NORMALIZED = 1
    VARIANCE = 2
    VARIANCE_NORMALIZED = 3
    VARIANCE_ARC = 4


def get_index(arg, name):
    try:
        return arg.index(name)
    except:
        return -1


def read_image(file):
    img = None
    header = None
    if file.endswith(".exr"):
        f = pyexr.open(file)
        header = f.input_file.header()
        img = f.get('default', pyexr.FLOAT)
        #img = pyexr.read(file)
    elif file.endswith(".png"):
        import cv2
        img = cv2.cvtColor(cv2.imread(file),cv2.COLOR_BGR2RGB)
    else:
        img = imageio.v2.imread(file)
    if len(img.shape) == 2:
        img = img[..., None]
    return img, header


def read_ranges(filename):
    result = [np.zeros(0,dtype=int)]
    with open(filename, 'r') as f:
        for line in f:
            sp = line.split()
            result.append(np.arange(int(sp[0]), int(sp[1])))
    return np.concatenate(result,dtype=int)


def printProgressBar (iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█', printEnd="\r"):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()


def divlim(divident, divisor):
    return np.divide(divident, divisor, np.ones_like(divident), where=np.logical_or(divident!=0,divisor!=0))


def conditional_frequency(x,y,axis=None):
    return np.sum(x*y,axis)/np.sum(y,axis)


def inrange(data, low, high, lowclosed = True, highclosed = False):
    lbound = low < data if lowclosed else low <= data
    hbound = data < high if highclosed else data <= high
    return np.logical_and(lbound, hbound)


def process_frame(core, filenames, mode, expression, opts, offset, logging):
    if len(filenames) == 0:
        return None
    if logging < 0:
        print(filenames)
    added = None
    accepted = []
    accepted_total = 0
    extracted = []
    for ifile in range(len(filenames)):
        file = filenames[ifile]
        try:
            img, header = read_image(file)
            if opts.criteria is not None:
                if not eval(opts.criteria):
                    continue
            if opts.check is not None:
                chk = eval(opts.check)
                if chk is not None:
                    raise chk
            if header is not None and header['num_total'] is not None:
                accepted_total += int(header['num_total'])
            else:
                accepted_total += 1
            if expression is not None:
                ldict={'img':img}
                exec(expression,globals(),ldict)
                img=ldict['img']
            ds = None
            shape = img.shape
            if opts.transformation == Transformation.EQUIDISTANT or opts.transformation == Transformation.EQUIDISTANT_HALF:
                pts = (np.linspace(-1+1/shape[0],1+1/shape[0],shape[0],endpoint=False), np.linspace(-1+1/shape[1],1+1/shape[1],shape[1],endpoint=False))
                tmp = np.sqrt(np.square(pts[0])[:,np.newaxis,np.newaxis]+np.square(pts[1])[np.newaxis,:,np.newaxis]) * np.pi
                if opts.transformation == Transformation.EQUIDISTANT_HALF:
                    tmp /= 2
                ds = divlim(np.sin(tmp),tmp)
            elif opts.transformation == Transformation.CYLINDRICAL_EQUIDISTANT:
                y = np.linspace((-1+1/shape[0]) * np.pi * 0.5,(1+1/shape[0]) * np.pi * 0.5,shape[0],endpoint=False)
                ds = np.cos(y)[:,np.newaxis,np.newaxis]
            if mode == Mode.VARIANCE_ARC and img.shape[2] != 3:
                raise Exception("Shape of image is completely wrong", img.shape)
            if mode == Mode.AVERAGE_NORMALIZED or mode == Mode.VARIANCE_ARC:
                div = linalg.norm(img, axis=2)[..., np.newaxis]
                if np.all(div == 0):
                    raise Exception("Warning image ", str(file), " is completely zero")
                img = img.astype(float,copy=False)
                img /= div
            if offset is not None:
                img = img.astype(float,copy=False)
                img -= offset
            accepted.append(file)
            if mode == Mode.AVERAGE or mode == Mode.AVERAGE_NORMALIZED:
                to_add = img
            elif mode == Mode.VARIANCE or mode == Mode.VARIANCE_NORMALIZED:
                img = img.astype(float,copy=False)
                to_add = np.square(img)
            elif mode == Mode.VARIANCE_ARC:
                img = img.astype(float,copy=False)
                to_add = 2 * np.arcsin(linalg.norm(img, axis=2) / 2)
            if added is not None:
                added += to_add
            else:
                added = to_add.astype(float,copy=False)
            for sext in opts.sextract:
                ldict={'img':img,'ds':ds,'header':header,'file':file,'filename':os.path.splitext(os.path.basename(file))[0]}
                exec(sext,globals(),ldict)
                extracted.append(ldict['res'])
            if len(opts.sextract) == 0:
                extracted.append(to_add[opts.extract])
            if opts.display_process == core:
                printProgressBar(ifile, len(filenames), prefix='Progress:', suffix='Complete', length=50)
        except ValueError as err:
            raise Exception("Couldn't process " + file, err)
    return added, accepted, accepted_total, extracted

def process_frames(filenames, mode, expression, opts, offset, logging):
    if offset is not None and (mode == Mode.VARIANCE_ARC or mode == Mode.VARIANCE_NORMALIZED):
        div = linalg.norm(offset, axis=2)[..., None]
        if np.any(div != 0):
            offset /= div
    image_list = None
    if len(filenames) < 10:
        image_list = [process_frame(0, filenames, mode, expression, opts, offset, logging)]
    else:
        import multiprocessing
        from joblib import Parallel, delayed
        num_cores = max(1,min(multiprocessing.cpu_count(), len(filenames) // 10))
        factor = (len(filenames) + num_cores - 1) // num_cores
        image_list = Parallel(n_jobs=num_cores)(
            delayed(process_frame)(core, filenames[core * factor:min((core + 1) * factor, len(filenames))], mode,
                                   expression, opts, offset, logging) for core in range(num_cores))
    accepted = []
    extracted = []
    accepted_total = 0
    added = None
    for img in image_list:
        if img is not None and img[0] is not None:
            if added is not None:
                added += img[0]
            else:
                added = img[0]
            accepted += img[1]
            accepted_total += img[2]
            extracted += img[3]
    return added, accepted, accepted_total, extracted


def read_numbers(filename):
    with open(filename, 'r') as f:
        return np.asarray([int(x) for x in f])


def create_parent_directory(filename):
    dirname = os.path.dirname(filename)
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def write_fimage(filename, img, extra_header={}):
    if filename.endswith(".exr"):
        if len(img.shape) == 2 or img.shape[2] == 1:
            header = OpenEXR.Header(*img.shape[0:2])
            header['channels'] = {'Y': Imath.Channel(Imath.PixelType(OpenEXR.FLOAT))}
            for key in extra_header:
                header[key] = extra_header[key]
            #header.insert(extra_header)
            out = OpenEXR.OutputFile(filename, header)
            out.writePixels({'Y': img})
            out.close()
        else:
            pyexr.write(filename, img, extra_headers=extra_header)
    else:
        imageio.imwrite(filename, img)


def to_exr_value(value):
    if value > np.iinfo(np.int32).max:
        return str(value).encode()
    return value


def main():
    imageio.plugins.freeimage.download()
    filenames = []
    divide = True
    premode = None
    mode = Mode.AVERAGE
    expression = None
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
    fallback = None
    precheck = False
    opts = Opts()
    time_start = time.time()
    i = 1
    while i < len(sys.argv):
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
                raise Exception("Value ", value, " not known")
            i += 1
        elif arg == "framelist" or arg == "mframelist" or arg == "rangelist":
            framelist = sys.argv[i + 1]
            numbers = None
            if framelist == "stdin":
                numbers = np.asarray([int(line) for line in sys.stdin], dtype=int)
            else:
                if arg == "rangelist":
                    numbers = read_ranges(framelist)
                else:
                    numbers = read_numbers(framelist)
            if arg == "mframelist":
                numbers = numbers - 1
            filename = sys.argv[i + 2].split("{}")
            if len(filename) == 1:
                filenames = filename
            elif len(filename) == 2:
                filenames = np.core.defchararray.add(
                    np.core.defchararray.add(filename[0], numbers.astype(str)), filename[1])
            else:
                raise Exception("only one placeholder is supported")
            i += 3
        elif arg == "input" or arg == "-i":
            for arg in sys.argv[i + 1:]:
                filenames = filenames + glob.glob(arg)
            filenames.sort()
            break
        elif arg == "criteria":
            opts.criteria = compile(sys.argv[i + 1], '<string>', 'eval')
            i += 1
        elif arg == "check":
            opts.check = compile(sys.argv[i + 1], '<string>', 'eval')
            i += 1
        elif arg == "alphamask":
            opts.alphamask, alphaheader = read_image(sys.argv[i + 1])
            i += 1
        elif arg == "expression":
            expression = compile(sys.argv[i + 1], '<string>', 'exec')
            i += 1
        elif arg == "showplot":
            show = True
        elif arg == "logging":
            logging = int(sys.argv[i + 1])
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
            eoutput = (sys.argv[i + 1],sys.argv[i + 2].split())
            i += 2
        elif arg == "preout":
            preout = sys.argv[i + 1]
            i += 1
        elif arg == "soutput":
            soutput = sys.argv[i + 1]
            i += 1
        elif arg == "prein":
            prein = sys.argv[i + 1]
            i += 1
        elif arg == "transformation":
            if sys.argv[i + 1] == "equidistant-half":
                opts.transformation = Transformation.EQUIDISTANT_HALF
            elif sys.argv[i + 1] == "equidistant":
                opts.transformation = Transformation.EQUIDISTANT
            elif sys.argv[i + 1] == "cylindrical_equidistant":
                opts.transformation = Transformation.CYLINDRICAL_EQUIDISTANT
            else:
                raise Exception('Unknown transformation', sys.argv[i + 1])
            i += 1
        elif arg == "progress":
            opts.display_process = int(sys.argv[i + 1])
            i += 1
        elif arg == "extract":
            opts.extract.append(np.fromstring(sys.argv[i + 1], sep=',', dtype=int))
            i += 1
        elif arg == "sextract":
            opts.sextract.append(compile(sys.argv[i + 1], '<string>', 'exec'))
            i += 1
        elif arg == "precheck":
            precheck = True
        elif arg == "fallback":
            fallback = (*[int(a) for a in sys.argv[i+1:i+4]], sys.argv[i + 4])
            i += 4
        else:
            raise Exception("Unknown argument " + arg)
        i += 1
    opts.extract = (*np.asarray(opts.extract).T,)
    if len(opts.extract) == 0:
        opts.extract = ([], [])
    if logging < 0:
        print(sys.argv)
    if logging < -1:
        print(filenames)
    if precheck:
        from os import path
        successful = True
        for file in filenames:
            if not path.isfile(file):
                successful = False
                print("Precheck failed, file " + file + " doesn't exist")
        if not successful:
            raise Exception("Files incomplete")
    preimage = None
    if prein is not None:
        preimage, preheader = read_image(prein)
    elif premode is not None:
        preimage, accepted, accepted_total, extracted = process_frames(filenames, premode, expression, opts, None, logging)
        preimage /= len(accepted)
        if show:
            import matplotlib.pyplot as plt
            plt.imshow(preimage)
            plt.show()
        if preout is not None:
            write_fimage(preout, preimage.astype(np.float32), {'num': len(accepted), 'num_total': to_exr_value(accepted_total)})
    if eoutput is not None or coutput is not None or noutput is not None or moutput is not None or output is not None or doutput is not None:
        image = None
        if mode is not None:
            if logging < 0:
                print("second pass")
            image, accepted, accepted_total, extracted = process_frames(filenames, mode, expression, opts, preimage, logging)
            if mode == Mode.VARIANCE_NORMALIZED or mode == Mode.AVERAGE_NORMALIZED:
                image = linalg.norm(image, axis=2)
        else:
            image = preimage
        print("accepted", len(accepted), "of", len(filenames), "\t", (time.time() - time_start) / max(len(accepted),1))

        if eoutput is not None:
            if eoutput[0] == "stdout":
                print(extracted)
            else:
                create_parent_directory(eoutput[0])
                if len(extracted)==1:
                    extracted = extracted[0]
                try:
                    np.savetxt(eoutput[0], [x.flatten() for x in extracted], delimiter=' ', fmt='%f', header=' '.join(eoutput[1]), comments='')
                except:
                    np.savetxt(eoutput[0], [x.flatten() for x in extracted], delimiter=' ', fmt='%s', header=' '.join(eoutput[1]), comments='')

        if coutput is not None:
            print(coutput, len(accepted))
            create_parent_directory(coutput)
            file = open(coutput, "w")
            file.write(str(len(accepted)))
            file.close()

        if noutput is not None:
            print(noutput, len(accepted))
            file = open(noutput, "w")
            for file in accepted:
                create_parent_directory(file)
                file.write(os.path.splitext(file)[0])
            file.close()

        if moutput is not None:
            print(moutput, len(accepted))
            indices = []
            for afile in accepted:
                indices.append(int(os.path.splitext(ntpath.basename(afile))[0]))
            indices = np.asarray(indices) + 1
            indices = np.sort(indices)
            create_parent_directory(moutput)
            np.savetxt(moutput, indices, delimiter='\n', fmt='%i')
        # image = np.where(np.isnan(image), np.zeros_like(image), image)
        if image is not None:
            if logging < -2:
                print("Result has shape " + str(image.shape))
            if doutput is not None:
                divided = image / len(accepted)
                create_parent_directory(doutput)
                write_fimage(doutput, divided.astype(np.float32, copy=False),{'num': len(accepted), 'num_total': to_exr_value(accepted_total)})
            if output is not None:
                create_parent_directory(output)
                write_fimage(output, image.astype(np.float32),{'num': len(accepted), 'num_total': to_exr_value(accepted_total)})
            if soutput is not None:
                create_parent_directory(soutput)
                file = open(soutput, "w")
                file.write(str(np.sum(image)))
                file.close()
        elif fallback is not None:
            x, y, z = np.ogrid[0:fallback[0], 0:fallback[1], 0:fallback[2]]
            img = eval(fallback[3])
            if logging < -2:
                print("Fallback image has shape " + img.shape)
            if doutput is not None:
                create_parent_directory(doutput)
                write_fimage(doutput, img.astype(np.float32),{'num': len(accepted), 'num_total': to_exr_value(accepted_total)})
            if output is not None:
                create_parent_directory(output)
                write_fimage(output, img.astype(np.float32),{'num': len(accepted), 'num_total': to_exr_value(accepted_total)})
        else:
            print("Don't write image")
main()
