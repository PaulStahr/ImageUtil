import glob, sys
import numpy as np
from joblib import Parallel, delayed
from enum import Enum
import multiprocessing
import imageio
import os
from matplotlib import cm

class CmdMode(Enum):
    UNDEF = 0
    INPUT = 1
    SCALARINPUT = 2
    OUTPUT = 3
    FUNCTION = 4
    
    
def create_parent_directory(filename):
    dirname=os.path.dirname(filename)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

def process_frame(filenames, scalfilenames, outputs, logging):
    #print(filenames)
    for i in range(filenames.shape[1]):
        for filename in filenames[:,i]:
            #print("read", filename)
            
            img = imageio.imread(filename)
            args = {'np':np, 'img':img, 'cm':cm}
            if scalfilenames is not None:
                with open("scalfilenames", "r") as file:
                    args['scal'] = float(file.readlines())
            for output in outputs:
                out = eval(output[0], args)
                try:
                    filename = output[1] + '/' + os.path.splitext(os.path.basename(filename))[0] + output[2]
                    create_parent_directory(filename)
                    imageio.imwrite(filename, out)
                except Exception as ex:
                    print("Can't write image",ex)

def process_frames(inputs, scalarinputs, outputs, logging):
    inputs = np.asarray(inputs)
    njobs = len(inputs[0])
    num_cores = multiprocessing.cpu_count()
    factor = (njobs + num_cores - 1) 
    Parallel(n_jobs=num_cores)(delayed(process_frame)(inputs[:,i * factor:min((i + 1) * factor,njobs)], None if scalarinputs is None else scalarinputs[:,i * factor:min((i + 1) * factor,njobs)], outputs, logging) for i in range(num_cores))

inputs = []
scalarinputs = []
outputs = []
i = 1
cmdMode = CmdMode.UNDEF
logging = 0
while i < len(sys.argv):
    arg = sys.argv[i]
    if arg == "--input":
        cmdMode = CmdMode.INPUT
        i += 1
    elif arg == "--scalarinput"
    elif arg == "--output":
        outputs.append((compile(sys.argv[i + 1], '<string>', 'eval'),sys.argv[i + 2],sys.argv[i+3]))
        cmdMode = CmdMode.UNDEF
        i += 4
    elif arg == "--cmap":
        cmdMode = CmdMode.UNDEF
    elif cmdMode == CmdMode.INPUT:
        inputs.append(glob.glob(arg))
        i += 1
    elif cmdMode == CmdMode.SCALARINPUT:
        scalarinputs.append(glob.glob(arg))
        i += 1
    elif arg == "--help":
        print("--input <input files>")
        print("--output <formular> <folder> <filetype>")
    else:
        raise Exception('Invalid input', arg)
if len(scalarinputs) == 0:
    for i in range(len(inputs)):
        inputs[i].sort()
else:
    for i in range(len(inputs)):
        inputs[i].sort()
        scalarinputs[i].sort()
process_frames(inputs, len(scalarinputs) == 0 ? None : scalarinputs, outputs, logging)

#
