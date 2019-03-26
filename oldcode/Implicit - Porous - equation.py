import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.sparse import diags
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import fsolve
from matplotlib import cm


def d(x,a):
    return signal.unit_impulse(len(x), a)

def molli(x):
    f=np.zeros(len(x))
    y=np.where(abs(x)<1)
    for i in y:
        f[i]=np.exp((-1/(1-abs(x[i])**2)))
    return f

def molli2(x):
    f = np.ones(len(x))
    y = np.where(abs(x) < 1)
    for i in y:
        f[i] -= np.exp((-1 / (1 - abs(x[i]) ** 2)))
    return f

m=3
g=m+1
T=4
X=6
M=200
N=200#int(np.ceil(2*M**2*T/(X**2*(m+1)))) #Is this CFL condition right
h=X/M
k=T/N
x=np.linspace(-X/2,X/2,M+1)
t=np.linspace(0,T,N+1)
p=k/(h**2)

U=np.zeros(shape=[N+1,M+1])
U[0]=molli(x)#d(x,int(np.ceil(0.5*M)))
test_start=0.5*np.ones(M-1)


b=np.zeros(M-1)
b[0]=0
b[-1]=0

A=diags([1,-2,1],[-1,0,1],shape=(M-1,M-1)).toarray()


for i in range(N):
    def F(u):
        return u - p * (np.dot(A, u ** g)) - U[i][1:-1] - p*b**g
    U[i+1][1:-1]=fsolve(F,U[i][1:-1])
    U[i+1][0]=b[0]
    U[i+1][-1]=b[-1]

x,t=np.meshgrid(x,t)

fig=plt.figure()
ax=fig.add_subplot(111,projection="3d")
surf=ax.plot_surface(x,t,U,cmap=cm.seismic)
ax.view_init(azim=55)
plt.show()


s=np.sum(U[-1])*h
s1=np.sum(U[0])*h
print(s1)
print(s)
print("Total endring i masse,",(s1-s))

