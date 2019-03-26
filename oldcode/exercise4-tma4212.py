import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from scipy.sparse import diags


def f(x):
    return np.sin(x)#np.exp(-64*(x-0.5)**2)*np.sin(np.pi*32*x)



def u(x,t):
    return np.sin(x-t)



x=np.linspace(0,3,4000)
t=np.linspace(0,2,4000)
x,t=np.meshgrid(x,t)

h=1/200
k=1/200
M=int(3/h)
r=k/h
N=int(2/k)
y=np.linspace(0,3,M+1)
T=np.linspace(0,2,N+1)

h1=1/800
k1=1/800
N_ex=int(2/k1)
M_ex=int(3/h1)
yex=np.linspace(0,3,M_ex)

fig=plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf=ax.plot_surface(x,t,u(x,t),cmap=cm.seismic)
ax.view_init(azim=30)
plt.xlabel(r"x-axis")
plt.ylabel(r"t-axis")
plt.show()
""" 
for i in T:
    plt.plot(y,u(y,i),label=r"$u(x,t)$")
    plt.grid()
    plt.xlabel(r"$x$-axis")
    plt.ylabel(r"$u$")
    #plt.legend()
    plt.show()
"""
""" 
plt.plot(y,f(y))
plt.show()
"""
#Lax-Wend
A=diags([r/2+r**2/2,1-r**2,-r/2+r**2/2],[-1,0,1],shape=(M+1,M+1)).toarray()
U1=np.zeros(shape=[N+1,M+1])
U1[0]=f(y)
b=np.zeros(M+1)
b[0]=1

Aex=diags([r/2+r**2/2,1-r**2,-r/2+r**2/2],[-1,0,1],shape=(M_ex,M_ex)).toarray()
""" 
Uex=np.zeros(shape=[N_ex,M_ex])
Uex[0]=f(yex)
Udif=np.zeros(shape=[N,M])
Udif=Uex[0][::int(h/h1)]-U1[0]
print(len(Udif))
print(len(U1))
for i in range(N_ex-1):
    Uex[i+1]=np.dot(Aex,Uex[i])
"""
for i in range(N):
    U1[i+1]=np.dot(A,U1[i])+(r/2+r**2/2)*f(-(i+1)*k)*b
    #U1[i+1][0]=f(-(i+1)*k)
    #U=Uex[i*int(k/k1)][::int(h/h1)]
    #print(len(U[i*int(k/k1)]))
    #Udif[i+1]=U-U1[i] #i*int(k/k1)



    #U1[i+1][0]=f(-(i+1)*k)

#Wend
AC = diags([(1 - r) / (1 + r), 1, 0], [-1, 0, 1], shape=(M+1, M+1)).toarray()
BC = diags([1, (1 - r) / (1 + r), 0], [-1, 0, 1], shape=(M+1, M+1)).toarray()

U = np.zeros(shape=[N+1, M+1])
U[0] = f(y)

for i in range(N):
    b = np.dot(BC, U[i])
    U[i + 1] = np.linalg.solve(AC, b)
    U[i + 1][0] = f(-(i+1) * k)

#Making a grid
y,T=np.meshgrid(y,T)

#plotting Lax-wend
fig=plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf=ax.plot_surface(y,T,U1,cmap=cm.seismic)
ax.view_init(azim=30)
plt.title(r"Lax-Wend")
plt.xlabel(r"x-axis")
plt.ylabel(r"t-axis")
plt.show()
"""
#Plotting analytic
fig=plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf=ax.plot_surface(y,T,u(y,T),cmap=cm.seismic)
ax.view_init(azim=30)
plt.xlabel(r"x-axis")
plt.ylabel(r"t-axis")
plt.show()
"""
#Plotting error Lax
fig=plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf=ax.plot_surface(y,T,U1-u(y,T),cmap=cm.seismic)
ax.view_init(azim=30)
plt.title(r"Lax-Wend - Error")
plt.xlabel(r"x-axis")
plt.ylabel(r"t-axis")
plt.show()





#y,T=np.meshgrid(y,T)
#plotting Wend
fig=plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf=ax.plot_surface(y,T,U,cmap=cm.seismic)
plt.title(r"Wend")
ax.view_init(azim=30)
plt.xlabel(r"x-axis")
plt.ylabel(r"t-axis")
plt.show()

#plotting error Wend
fig=plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf=ax.plot_surface(y,T,U-u(y,T),cmap=cm.seismic)
ax.view_init(azim=30)
plt.title(r"Wend - Error")
plt.xlabel(r"x-axis")
plt.ylabel(r"t-axis")
plt.show()


#some testing at a spesific time

"""

z=np.linspace(0,3,M)

d=U1[0]-u(z,T[0])

plt.plot(z,d)
plt.show()
plt.plot(z,U1[200])
plt.show()
plt.plot(z,u(z,T[200]))
plt.show()
"""