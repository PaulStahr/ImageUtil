import matplotlib.pyplot as plt
import numpy as np
import imageio
import sys

im0 = imageio.imread(sys.argv[1])
im1 = imageio.imread(sys.argv[2])

ax = plt.gca()
fig = plt.gcf()
implot = ax.imshow(im0)

def onclick(event):
    if event.xdata != None and event.ydata != None and event.button == 2:
        print(event.xdata, event.ydata, *im1[int(round(event.ydata)), int(round(event.xdata))])
cid = fig.canvas.mpl_connect('button_press_event', onclick)

plt.show()
