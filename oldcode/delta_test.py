import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

M=999

u=(M)*signal.unit_impulse(M+1,int((M+1)/2))
x=np.linspace(-1,1,M+1)

plt.plot(x,u)
plt.show()

s=np.sum(u)/M
print(s)