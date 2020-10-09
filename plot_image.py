import imageio
import matplotlib.pyplot as plt
import sys
import numpy as np
import pyexr

def read_image(file):
    img = None
    if file.endswith(".exr"):
        img = pyexr.read(file)
    else:
        img = imageio.imread(file)
    if len(img.shape) == 2:
        img = img[..., None]
    if img.shape[2] == 1:
        img = img[:,:,0]
    return img


img = read_image(sys.argv[1])
plt.imshow(img)
if len(sys.argv) < 3:
    plt.show()
else:
    if True:
         img /= np.max(img)
    plt.imsave(sys.argv[2], img, cmap=plt.cm.jet)
