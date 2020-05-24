import imageio
import matplotlib.pyplot as plt
import sys
import numpy as np

img = imageio.imread(sys.argv[1])
plt.imshow(img)
if True:
    img /= np.max(img)
plt.imsave(sys.argv[2], img, cmap=plt.cm.jet)
