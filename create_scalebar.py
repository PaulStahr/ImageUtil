import matplotlib.pyplot as plt
import matplotlib as mpl
import sys
from matplotlib import cm

if sys.argv[1] != 'h' and sys.argv[1] != 'v':
    raise Exception('Direction has to be either horiontal or vertical')

fig, ax = plt.subplots(figsize=(6, 1) if sys.argv[1] == 'h' else (1, 6))
fig.subplots_adjust(bottom=0.5)

#cmap = mpl.cm.cool
cmap = cm.gnuplot
norm = mpl.colors.Normalize(vmin=float(sys.argv[2]), vmax=float(sys.argv[3]))

fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
             cax=ax, orientation=('horizontal' if sys.argv[1] == 'h' else 'vertical'), label=sys.argv[4])

if len(sys.argv) < 6:
    plt.show()
else:
    plt.savefig(sys.argv[5])