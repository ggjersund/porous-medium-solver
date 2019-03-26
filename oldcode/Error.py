import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.sparse import diags
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import fsolve
from matplotlib import cm
import time
import scipy.integrate as integrate


g=2
n=1
a=n/(n*(g-1)+2)
b=1/(n*(g-1)+2)
m=g-1
T=2
X=16



def Barenblatt(x,t):
    return (1/t**a)*((0.5-((g-1)/(2*g))*b*(x**2)/(t**(2*b)))**(1/(g-1)))


def F(x):
    return x**(m+1)



def d(x,a):
    return signal.unit_impulse(len(x), a)


error=np.array([])
h_list=np.array([])
error2=np.array([])
h_list2=np.array([])
error3=np.array([])
h_list3=np.array([])


def error_analysis_expl(i,metnum):
    M=50*i
    N=int(np.ceil(2*M**2*T/(X**2))*3**m) #Is this CFL condition right=
    #print(N)
    h=X/M
    k=T/N
    x=np.linspace(-X/2,X/2,M+1)
    y=np.linspace(-X/2,X/2,M+1)
    t=np.linspace(0.5,T+0.5,N+1)
    p=k/(h**2)

    def G(x):
        return np.dot(A, F(x))

    x,t=np.meshgrid(x,t)

    u_ext=np.array(Barenblatt(x,t))

    for i in range(len(u_ext)):
        for j in range(len(u_ext[i])):
            if u_ext[i][j]<0:
                u_ext[i][j]=0

    A = diags([1, -2, 1], [-1, 0, 1], shape=(M - 1, M - 1)).toarray()
    #A[0][1]+=1/h
    #A[-1][-2]+=1/h
    U=np.zeros(shape=[N+1,M+1])
    U[0]=u_ext[0]

    b=np.zeros(M-1)
    b[0]=0
    b[-1]=0

    for j in range(N):
        U[j+1][1:-1]=U[j][1:-1]+p*G(U[j][1:-1])+p*F(b)
        U[j+1][0]=b[0]
        U[j+1][-1]=b[-1]

    if metnum == 1:
        met = r"Error in $||.||_{L^1}$"
        # REMEMBER SQUARE ROOT HERE
        e = np.sum(abs(U - u_ext)) * h * k  # np.amax(abs(U-u_ext))
    elif metnum == 2:
        met = r"Error in $||.||_{L^2}$"
        e = np.sum(abs(U - u_ext)**2) * h * k  # np.amax(abs(U-u_ext))
    else:
        e=np.amax(abs(U-u_ext))
        met=r"Error in $||.||_{L^\infty}$"

    #e=np.sum(abs(U-u_ext)**1)*h**2*k#np.amax(abs(U-u_ext))
    titel = "Convergence plot for the error for explicit method"
    return e,h,titel,met


def error_analysis_impl(i,metnum):

    M=50*i
    N=M#**2
    h=X/M
    k=T/N
    x=np.linspace(-X/2,X/2,M+1)
    y=np.linspace(-X/2,X/2,M+1)
    t=np.linspace(0.5,T+0.5,N+1)
    p=k/(h**2)


    x,t=np.meshgrid(x,t)

    u_ext=np.array(Barenblatt(x,t))

    for i in range(len(u_ext)):
        for j in range(len(u_ext[i])):
            if u_ext[i][j]<0:
                u_ext[i][j]=0

    A = diags([1, -2, 1], [-1, 0, 1], shape=(M - 1, M - 1)).toarray()
    #A[0][1]+=1/h
    #A[-1][-2]+=1/h
    U=np.zeros(shape=[N+1,M+1])
    U[0]=u_ext[0]

    b=np.zeros(M-1)
    b[0]=0
    b[-1]=0

    for i in range(N):
        def F(u):
            return u - p * (np.dot(A, u ** g)) - U[i][1:-1] - p * b ** g
        U[i + 1][1:-1] = fsolve(F, U[i][1:-1])
        U[i + 1][0] = b[0]
        U[i + 1][-1] = b[-1]

    if metnum == 1:
        met = r"Error in $||.||_{L^1}$"
        e = np.sum(abs(U - u_ext)) * h * k  # np.amax(abs(U-u_ext))
    elif metnum == 2:
        met = r"Error in $||.||_{L^2}$"
        e = np.sum(abs(U - u_ext)**2) * h * k  # np.amax(abs(U-u_ext))
    else:
        e=np.amax(abs(U-u_ext))
        met=r"Error in $||.||_{L^\infty}$"
    titel="Convergence plot for the error for implicit method"
    return e,h,titel,met


start_t=time.time()
for i in range(2,11):
    print("iteration ", i)
    e,h,titel,met1=error_analysis_expl(i,1)
    e2, h2, titel, met2 = error_analysis_expl(i, 2)
    e3, h3, titel, met3 = error_analysis_expl(i, 3)
    error =np.append(error,e)
    h_list =np.append(h_list,h)
    error2 = np.append(error2, e2)
    h_list2 = np.append(h_list2, h2)
    error3 = np.append(error3, e3)
    h_list3 = np.append(h_list3, h3)
    print("It took ",time.time()-start_t,"seconds to finish this iteration from start.")

plt.loglog(h_list,error,label=met1)
plt.loglog(h_list2,error2,label=met2)
plt.loglog(h_list3,error3,label=met3)
#plt.loglog(h_list,h_list**(2),"r--")
plt.xlabel(r"$h$")
plt.ylabel(r"$||e||$")
plt.title(titel)
plt.legend()
plt.grid()
plt.show()

z=np.around(np.polyfit(np.log(h_list), np.log(error), 1)[0],2)
print(r"The degree of the L1 method is", z)
z=np.around(np.polyfit(np.log(h_list2), np.log(error2), 1)[0],2)
print(r"The degree of the L2 method is", z)
z=np.around(np.polyfit(np.log(h_list2), np.log(error3), 1)[0],2)
print(r"The degree of the Max method is", z)
