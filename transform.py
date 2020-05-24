import glob, sys
import numpy as np
from joblib import Parallel, delayed
from enum import Enum
import multiprocessing
import imageio
import os

class CmdMode(Enum):
    UNDEF = 0
    INPUT = 1
    OUTPUT = 2
    FUNCTION = 3
    
def process_frame(filenames, outputs, logging):
    #print(filenames)
    for i in range(filenames.shape[1]):
        for filename in filenames[:,i]:
            #print("read", filename)
            img = imageio.imread(filename)
            #print("images", len(img), 'out', outputs)
            for output in outputs:
                out = eval(output[0], {'np':np, 'img':img})
                print("shape", out.shape)
                #print(out)
                imageio.imwrite(output[1] + '/' + os.path.basename(filename), out)
    
    
def process_frames(inputs, outputs, logging):
    inputs = np.asarray(inputs)
    njobs = len(inputs[0])
    num_cores = multiprocessing.cpu_count()
    factor = (njobs + num_cores - 1) 
    Parallel(n_jobs=num_cores)(delayed(process_frame)(inputs[:,i * factor:min((i + 1) * factor,njobs)], outputs, logging) for i in range(num_cores))

inputs = []
outputs = []
i = 1
cmdMode = CmdMode.UNDEF
logging = 0
while i < len(sys.argv):
    arg = sys.argv[i]
    if (arg == "--input"):
        cmdMode = CmdMode.INPUT
        i += 1
    elif(arg == "--output"):
        outputs.append((compile(sys.argv[i + 1], '<string>', 'eval'),sys.argv[i + 2]))
        cmdMode = CmdMode.UNDEF
        i += 3
    elif cmdMode == CmdMode.INPUT:
        inputs.append(glob.glob(arg))
        i += 1
    else:
        raise Exception('Invalid input', arg)
for i in range(len(inputs)):
    inputs[i].sort()
#print("inputs", inputs)
process_frames(inputs, outputs, logging)
