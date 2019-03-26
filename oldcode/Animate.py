#from functions import *
import matplotlib; matplotlib.use("TkAgg")
#import parameters as pr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.cm as cm


def animasjonsfunk(x, y, N):  # INPUTS x_list and y_list!
    fig = plt.figure('Planets', figsize=(8, 8))
    ax = plt.axes(xlim = (-np.max(x), np.max(x)), ylim = (-np.max(y), np.max(y)))
    circle = plt.Circle((0, 0),  0.00464913034*10, color='y')
    # circ = fig.patch(circle)
    ax.add_artist(circle)
    line, = ax.plot([], [], 'og', lw=2, label='Planet')
    traj, = ax.plot([], [], '-k', lw = 2, label = 'Trajectory')
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
    def init():
        line.set_data([x[0]], [y[0]])
        time_text.set_text('')
        traj.set_data([x[0]], [y[0]])
        return line, time_text, circle, traj


    def animate(i):
        line.set_data(x[i], y[i])
        if i<2000:

            traj.set_data(x[:i], y[:i])
        else:
            traj.set_data(x[i-2000:i], y[i-2000:i])
        # -------------------------
        # time_text.set_text('Tid: {0:.3E}s'.format(i * timeStep))
        return line, circle, traj# -----------------------
        # return line, time_text  # -------------------------

    # Set up formatting for the movie files
    # Writer = animation.writers['ffmpeg']
    # writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)

    # animation.writers()


    plt.legend()
    plt.grid()
    anim = animation.FuncAnimation(fig, animate, init_func=init, repeat=True, frames=N, interval=1, blit=False)
    # anim.save('N{}dx{}Xo{}.gif'.format(N, dx, x0/(N * dx)), writer= "imagemagick")
    plt.show()


def animasjonsfunk2(x1, y1, x2, y2, N):  # INPUTS x_list and y_list!
    fig = plt.figure('Planets', figsize=(8, 8))
    ax = plt.axes(xlim = (-np.max(2*x2), np.max(2*x2)), ylim = (-np.max(2*x2), np.max(2*x2)))
    circle = plt.Circle((0, 0),  0.00464913034*10, color='y')
    # circ = fig.patch(circle)
    ax.add_artist(circle)

    line, = ax.plot([], [], 'og', lw=2, label='Earth')
    line2, = ax.plot([], [], 'or', lw=1.5, label='Mars')
    traj, = ax.plot([], [], '-g', lw = 2, label = 'Trajectory')
    traj2, = ax.plot([], [], '-r', lw = 2, label = 'Trajectory')
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
    def init():
        line.set_data([x1[0]], [y1[0]])
        line2.set_data([x2[0]], [y2[0]])
        time_text.set_text('')
        traj.set_data([x1[0]], [y1[0]])
        traj2.set_data([x2[0]], [y2[0]])


        return line, time_text, circle, traj, line2, traj2


    def animate(i):
        line.set_data(x1[i], y1[i])
        traj.set_data(x1[:i], y1[:i])
        line2.set_data(x2[i], y2[i])
        traj2.set_data(x2[:i], y2[:i])
        # -------------------------
        # time_text.set_text('Tid: {0:.3E}s'.format(i * timeStep))
        return line, circle, traj, line2, traj2# -----------------------
        # return line, time_text  # -------------------------

    # Set up formatting for the movie files
    # Writer = animation.writers['ffmpeg']
    # writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)

    # animation.writers()


    plt.legend()
    anim = animation.FuncAnimation(fig, animate, init_func=init, repeat=True, frames=N, interval=100, blit=False)
    # anim.save('N{}dx{}Xo{}.gif'.format(N, dx, x0/(N * dx)), writer= "imagemagick")
    plt.show()