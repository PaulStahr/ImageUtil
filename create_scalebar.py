import matplotlib.pyplot as plt
import matplotlib as mpl
import sys
from matplotlib import cm

if len(sys.argv) < 2 or (sys.argv[1] != 'h' and sys.argv[1] != 'v'):
    raise Exception("Usage: <v/h> <cmap> <from> <to> <scale> <title> (<output>)")
else:
    mpl.rcParams['savefig.pad_inches'] = 0
    fig, ax = plt.subplots(figsize=(6, 1) if sys.argv[1] == 'h' else (2, 6))
    # fig.subplots_adjust(bottom=0.5)
    fig.subplots_adjust(left=0)
    # cmap = mpl.cm.cool
    cmap = None
    if sys.argv[2] == "gnuplot":
        cmap = cm.gnuplot
    elif sys.argv[2] == "gray":
        cmap = cm.gray
    else:
        raise Exception("Unknown cmap", sys.argv[2])
    norm = None
    if sys.argv[5] == "norm":
        norm = mpl.colors.Normalize(vmin=float(eval(sys.argv[3])), vmax=float(eval(sys.argv[4])))
    elif sys.argv[5] == "log":
        norm = mpl.colors.LogNorm(vmin=float(eval(sys.argv[3])), vmax=float(eval(sys.argv[4])))
    fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                 cax=ax, orientation=('horizontal' if sys.argv[1] == 'h' else 'vertical'), label=sys.argv[6], fraction=0.046, pad=0.04)
    plt.tight_layout()
    if len(sys.argv) < 8:
        plt.show()
    else:
        plt.savefig(sys.argv[7])
