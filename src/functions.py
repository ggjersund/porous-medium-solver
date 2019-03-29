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

def plot_convergence(x=np.array([]), e=np.array([]), labels=np.array([]), xlabel='x', ylabel='y', txt='Convergence plot'):
    plt.figure()
    for i in range(len(x)):
        plt.loglog(x[i], e[i], label=labels[i])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(txt)
    plt.legend()
    plt.show()

def forward_convergence_in_space(N):

    h_vector = np.zeros(10)
    L1 = np.zeros(10)
    L2 = np.zeros(10)
    Linf = np.zeros(10)

    for i in range(0, 10):
        object.change_M(30 + (20 * i))

        x, t, U, h, k = object.forward_euler()

        X, T = np.meshgrid(x, t)
        u = np.transpose(analytic(X, T))

        h_vector[i] = h

        # Calculate error norms
        L1[i] = np.sum(np.abs(U - u)) * h * k
        L2[i] = np.sqrt(np.sum(abs(U - u)**2) * h * k)
        Linf[i] = np.amax(abs(U - u))

        #print("Iteration:", i+1, "M:", self.M, "N:", self.N, "U max", np.amax(U), "u max", np.amax(u), "L1:", L1[i], "L2:", L2[i], "Linf:", Linf[i])

    return h_vector, L1, L2, Linf
