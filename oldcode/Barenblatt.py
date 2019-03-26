import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.sparse import diags
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import fsolve
from matplotlib import cm


g=1.3
n=1
a=n/(n*(g-1)+2)
b=1/(n*(g-1)+2)


def Barenblatt(x,t):
    return (1/t**a)*((0.5-((g-1)/(2*g))*b*(x**2)/(t**(2*b)))**(1/(g-1)))


def F(x):
    return x**(m+1)

def G(x):
    return np.dot(A,F(x))

def d(x,a):
    return signal.unit_impulse(len(x), a)





m=g-1
T=2
X=16
M=300
N=600#int(np.ceil(4*M**2*T/(X**2))*(max(m,1))*3**m) #Is this CFL condition right=
print(N)
h=X/M
k=T/N
x=np.linspace(-X/2,X/2,M+1)
y=np.linspace(-X/2,X/2,M+1)
t=np.linspace(0.01,T+0.01,N+1)
p=k/(h**2)




x,t=np.meshgrid(x,t)

u_ext=np.array(Barenblatt(x,t))

for i in range(len(u_ext)):
    for j in range(len(u_ext[i])):
        if u_ext[i][j]<0:
            u_ext[i][j]=0


A=diags([1,-2,1],[-1,0,1],shape=(M-1,M-1)).toarray()
#A[0][1]+=1/h
#A[-1][-2]+=1/h
U=np.zeros(shape=[N+1,M+1])
U[0]=u_ext[0]#Barenblatt(x,0.5)
for i in range(len(U)):
    for j in range(len(U[i])):
        if U[i][j]<0:
            U[i][j]=0

b=np.zeros(M-1)
b[0]=0
b[-1]=0
""" 
for j in range(N):
    U[j+1][1:-1]=U[j][1:-1]+p*G(U[j][1:-1])+p*F(b)
    U[j+1][0]=b[0]
    U[j+1][-1]=b[-1]
"""

for i in range(N):
    def F(u):
        return u - p * (np.dot(A, u ** g)) - U[i][1:-1] - p*b**g
    U[i+1][1:-1]=fsolve(F,U[i][1:-1])
    U[i+1][0]=b[0]
    U[i+1][-1]=b[-1]


fig=plt.figure()
ax=fig.add_subplot(111,projection="3d")
surf=ax.plot_surface(x,t,u_ext,cmap=cm.seismic)
plt.title("Barenblatt")
ax.view_init(azim=55)
plt.show()



fig=plt.figure()
ax=fig.add_subplot(111,projection="3d")
surf=ax.plot_surface(x,t,U,cmap=cm.seismic)
plt.title("Num")
ax.view_init(azim=55)
plt.show()

fig=plt.figure()
ax=fig.add_subplot(111,projection="3d")
surf=ax.plot_surface(x,t,u_ext-U,cmap=cm.seismic)
plt.title("error")
ax.view_init(azim=55)
plt.show()



print(np.amax(abs(U-u_ext)))



