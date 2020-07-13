import matplotlib.pyplot as plt
import numpy as np
import imageio
import sys

img0 = imageio.imread(sys.argv[1])
if len(sys.argv) > 2:
    img1 = imageio.imread(sys.argv[2])
else:
    img1 = img0
ax = plt.gca()
fig = plt.gcf()
implot = ax.imshow(img0)

def onclick(event):
    if event.xdata != None and event.ydata != None and event.button == 2:
        print(event.xdata, event.ydata, img1[int(round(event.ydata)), int(round(event.xdata))])
cid = fig.canvas.mpl_connect('button_press_event', onclick)

plt.show()
