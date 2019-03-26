import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import fsolve
from matplotlib import cm
from scipy.integrate import solve_ivp, odeint, LSODA
from scipy import signal



def f(x):
    return np.sin(np.pi*x)

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

def delta_app(x,a):
    return molli(x/a)/a

def g(x):
    return molli(x / (5*10**(-1))) / (5*10**(-1))


m=0.3
T=4
X=6
M=200
N=int(np.ceil(4*M**2*T/(X**2))*(max(m,1))*1**m) #Is this CFL condition right=
print(N)
h=X/M
k=T/N
x=np.linspace(-X/2,X/2,M+1)
t=np.linspace(0,T,N+1)
p=k/(h**2)

def d(x,a):
    return signal.unit_impulse(len(x), a)


A=diags([1,-2,1],[-1,0,1],shape=(M-1,M-1)).toarray()
#A[0][1]+=1/h
#A[-1][-2]+=1/h
U=np.zeros(shape=[N+1,M+1])
#U[0]=molli2(x)#3*(np.ones(M+1)-d(x,int(np.ceil(0.5*M)))-0.75*d(x,int(np.ceil(0.6*M))))

b=np.zeros(M-1)
b[0]=0
b[-1]=0

#If we want impulses
for i in range(1,10):
    if i<5:
        U[0]+=i*d(x,int(np.ceil(0.05*i*M)))
    else:
        U[0] += (10-i) * d(x, int(np.ceil(0.1 * i * M)))




def F(x):
    return x**(m+1)

def G(x):
    return np.dot(A,F(x))

for j in range(N):
    U[j+1][1:-1]=U[j][1:-1]+p*G(U[j][1:-1])+p*F(b)
    U[j+1][0]=b[0]
    U[j+1][-1]=b[-1]

print(U[-1][0])
#plt.plot(x,U[-1])
#plt.title("Sjå her")
#plt.show()

#print(G(np.array([2 for i in range(M+1)])))


#u=solve_ivp(A,[0,T],U[0],method="LSODA",t_eval=t)#,#t_eval=t,method='BDF')



x,t=np.meshgrid(x,t)

fig=plt.figure()
ax=fig.add_subplot(111,projection="3d")
surf=ax.plot_surface(x,t,U,cmap=cm.seismic)
ax.view_init(azim=55)
plt.show()

y=np.linspace(-X/2,X/2,M+1)

plt.plot(y,U[0])
plt.show()
#plt.plot(y,U[-2])
#plt.show()
plt.plot(y,U[20])
plt.show()
plt.plot(y,U[60])
plt.show()
plt.plot(y,U[-1])
plt.show()

s=np.sum(U[-1])*h
s1=np.sum(U[0])*h
print(s1)
print(s)
print("Total endring i masse,",(s1-s))