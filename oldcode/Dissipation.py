import numpy as np
import matplotlib.pyplot as plt


r=1
p=(1-r)/(1+r)
x=np.linspace(-np.pi,np.pi,1000)
print(p)
def ksi(x):
    a=(p+np.cos(x))**2+np.sin(x)**2
    b=p**2*np.sin(x)**2+(p*np.cos(x)+1)**2

    return np.sqrt((p+np.cos(x))**2+np.sin(x)**2)/(np.sqrt(p**2*np.sin(x)**2+(p*np.cos(x)+1)**2))

plt.plot(x,ksi(x))
plt.show()


def phi(x):
    im=p**2*np.sin(x)-np.sin(x)
    re=2*p+p**2*np.cos(x)+np.cos(x)
    t=np.arctan2(-im,re)
    return t/(r*x)

def psi(x):
    im=(1+r)**2*np.sin(x)-(1-r)**2*np.sin(x)
    re=2*(1-r)*(1+r)+((1+r)**2+(1-r)**2)*np.cos(x)
    t=np.arctan2(im,re)
    return t/(r*x)

plt.plot(x,phi(x)-psi(x))
plt.show()

plt.plot(x,phi(x),label=r"Helge")
#plt.plot(x,psi(x),label=r"Thomas")
plt.legend()
plt.show()

