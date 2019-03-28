import matplotlib
# Required for mac
matplotlib.use('TkAgg')
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
newparams = {'figure.figsize': (8.0, 4.0), 'axes.grid': True,
             'lines.markersize': 8, 'lines.linewidth': 2,
             'font.size': 12}


def plot_solution(x, t, U, txt='Solution', azim=75):
    fig = plt.figure()
    plt.clf()
    ax = fig.gca(projection='3d')
    T, X = np.meshgrid(t, x)
    ax.plot_wireframe(T, X, U)
    ax.plot_surface(T, X, U, cmap=cm.coolwarm)
    ax.view_init(azim=azim)
    plt.xlabel('t')
    plt.ylabel('x')
    plt.title(txt)
    plt.show()
