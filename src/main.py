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


from functions import forward_euler, backward_euler
from classes import PorousMediumEquation

if __name__ == "__main__":

    def Barenblatt(x, t, m, n):
        alpha = n / ((n * (m - 1)) + 2)
        beta = alpha / n
        k = beta * (m - 1) / (2 * m)
        s = 0.5 - (k * (np.abs(x)**2)) / (t**(2 * beta))
        return (1 / (t**alpha)) * ((np.maximum(s, 0))**(1 / (m - 1)))

    def initial(x):
        return Barenblatt(x, t=0.01, m=1.2, n=1)

    def boundary1(t):
        return (t-t) + 0

    def boundary2(t):
        return (t-t) + 0

    porous_forward = PorousMediumEquation(m=1.2, f=initial, g1=boundary1, g2=boundary2, M=90, N=200, T_low=0.01, T_high=2)
    porous_backward = PorousMediumEquation(m=1.2, f=initial, g1=boundary1, g2=boundary2, M=200, N=200, T_low=0.01, T_high=2)
    # porous_backward.add_impulse(index=54)

    x1, t1, U1, h1, k1 = porous_forward.forward_euler()
    x2, t2, U2, h2, k2 = porous_backward.backward_euler()

    fig = plt.figure()
    plt.clf()
    ax = fig.gca(projection='3d')
    X, T = np.meshgrid(t1, x1)
    ax.plot_wireframe(T, X, U1)
    ax.plot_surface(T, X, U1, cmap=cm.coolwarm)
    ax.view_init(azim=75)
    plt.xlabel('x')
    plt.ylabel('t')
    plt.title('Forward-Euler')
    plt.show()

    fig = plt.figure()
    plt.clf()
    ax = fig.gca(projection='3d')
    X, T = np.meshgrid(t2, x2)
    ax.plot_wireframe(T, X, U2)
    ax.plot_surface(T, X, U2, cmap=cm.coolwarm)
    ax.view_init(azim=75)
    plt.xlabel('x')
    plt.ylabel('t')
    plt.title('Backward-Euler')
    plt.show()
