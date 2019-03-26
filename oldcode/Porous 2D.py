import numpy as np
import matplotlib; matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from scipy import signal
from scipy.sparse import diags
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import fsolve
from matplotlib import cm
from scipy import sparse
from scipy.sparse.linalg import spsolve
from matplotlib import animation
from matplotlib.animation import FuncAnimation
import mpl_toolkits.mplot3d.axes3d as p3
import time
import matplotlib as mpl


m=1
g=m+1
T=4
X=6


n=2
a=n/(n*(g-1)+2)
b=1/(n*(g-1)+2)


def Barenblatt(x,y,t):
    return (1/t**a)*((0.5-((g-1)/(2*g))*b*(x**2+y**2)/(t**(2*b)))**(1/(g-1)))



def func(x,y):
    return np.exp(-(x**2+y**2))

def func2(x,y):
    return np.exp(-(x**4+y**4))

def A_laplace(M):
    M2 = M**2
    A = -4*np.eye(M2)
    for i in range(M2-1):
        A[i,i+1] = 1
        A[i+1,i] = 1
    for i in range(M-1,M2-1,M):
        A[i,i+1] = 0
        A[i+1,i] = 0
    for i in range(M2-M):
        A[i,i+M] = 1
        A[i+M,i] = 1
    return A

def solution(M,N,ui,xrange,trange):
    def gs(x):
        return (x ** 6)

    def gn(x):
        return (x ** 6) + 1

    def gw(y):
        return (y ** 7)

    def ge(y):
        return 1 + (y ** 7)

    # Source function
    def f(x, y):
        return (30 * (x ** 4)) + (42 * (y ** (5)))

    # The grid
    x = np.linspace(0, 1, M + 1)
    y = np.linspace(0, 1, M + 1)
    h = xrange / M
    k = trange / N
    p = k/(h**2)

    # Inner grid
    xi = x[1:-1]
    yi = y[1:-1]
    Xi, Yi = np.meshgrid(xi, yi)
    Mi = M - 1
    Mi2 = Mi ** 2

    A = A_laplace(M + 1)
    #As = sparse.csr_matrix(A)
    #b = np.zeros(Mi2)
    def F(u):
        return u - p * (np.dot(A, u ** g)) - ui #- p*b**g
    U=fsolve(F,ui)

    # Solve the linear system
    #Ui = spsolve(As, b)

    # Make an (M+1)x(M+1) array to store the solution,
    # including boundary
    #U = np.zeros((M + 1, M + 1))

    # Reshape the solution vector, and insert into the solution matrix
    #U[1:-1, 1:-1] = np.reshape(sol, (Mi, Mi))

    # include the boundaries
    #U[0, :] = 0#gs(x)
    #U[M, :] = 0#gn(x)
    #U[:, 0] = 0#gw(y)
    #U[:, M] = 0#ge(y)
    U = np.reshape(U,(M+1,M+1))

    return U

M=80
N=120
xrange=16
trange=5
t_start=0.01

t = np.linspace(t_start, trange+t_start, N + 1)
x = np.linspace(-xrange/2, xrange/2, M + 1)
y = np.linspace(-xrange/2, xrange/2, M + 1)
X,Y=np.meshgrid(x,y)
#u0=func(X,Y).flatten()
#print(len(u0))
#u=solution(M,N,u0)
#print(len(u[0]))
u=np.empty(shape=(N+1,M+1,M+1))
u[0]=np.array(Barenblatt(X,Y,t_start))
for i in range(len(u[0])):
    for j in range(len(u[0][i])):
        if u[0][i][j]<0:
            u[0][i][j]=0

start_time=time.time()
for i in range(N):
    print("Iteration",i+1)
    u[i+1]=solution(M,N,u[i].flatten(),xrange,trange)
    print("It has taken",time.time()-start_time,"seconds.")



# Attaching 3D axis to the figure
fig = plt.figure()
#ax = p3.Axes3D(fig)

def new_frame(i):
    ax = p3.Axes3D(fig)
    ax.set_xlim3d([-xrange / 2, xrange / 2])
    ax.set_xlabel(r'$x$')

    ax.set_ylim3d([-xrange / 2, xrange / 2])
    ax.set_ylabel(r'$y$')

    ax.set_zlim3d([0.0, trange])
    ax.set_zlabel(r'$u(x,t)$')

    ax.set_title('Porous medium equation')
    surf = ax.plot_surface(X, Y, u[i], cmap=cm.seismic)
    fake2Dline = mpl.lines.Line2D([0], [0], linestyle="none", c='b', marker='o')
    ax.legend([fake2Dline], [r"$t=$"+str(np.round(t[i],2))], numpoints=1)
    return surf

#Setting the data
data = u

""" 
# Setting the axes properties
ax.set_xlim3d([-xrange/2, xrange/2])
ax.set_xlabel('X')

ax.set_ylim3d([-xrange/2, xrange/2])
ax.set_ylabel('Y')

ax.set_zlim3d([0.0, 1.0])
ax.set_zlabel('Z')

ax.set_title('Porous medium equation')
"""
# Creating the Animation object
PME_ani = animation.FuncAnimation(fig, new_frame, N+1, interval=50,repeat=True, blit=False)

plt.show()

PME_ani.save('PME_finer.mp4')

print("The total time was",time.time()-start_time,"seconds.")