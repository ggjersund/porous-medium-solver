import numpy as np

from functions import plot_solution
from classes import PorousMediumEquation


if __name__ == "__main__":

    def Barenblatt(x, t, m, n):
        alpha = n / ((n * (m - 1)) + 2)
        beta = alpha / n
        k = beta * (m - 1) / (2 * m)
        s = 0.5 - (k * (np.abs(x)**2)) / (t**(2 * beta))
        return (1 / (t**alpha)) * ((np.maximum(s, 0))**(1 / (m - 1)))

    def molli2(x):
        f = np.ones(len(x))
        y = np.where(abs(x) < 1)
        for i in y:
            f[i] -= np.exp((-1 / (1 - abs(x[i]) ** 2)))
        return f

    def initial(x):
        return molli2(x)
        #return Barenblatt(x, t=0.01, m=1.2, n=1)

    def boundary1(t):
        return (t-t) + 1

    def boundary2(t):
        return (t-t) + 1

    porous_forward = PorousMediumEquation(m=1.2, f=initial, g1=boundary1, g2=boundary2, M=90, N=1800, T_low=0.01, T_high=2)
    porous_backward = PorousMediumEquation(m=1.2, f=initial, g1=boundary1, g2=boundary2, M=20, N=20, T_low=0.01, T_high=2)
    #porous_forward.add_impulse(index=54, ratio=0.75)

    x1, t1, U1, h1, k1 = porous_forward.forward_euler()
    x2, t2, U2, h2, k2 = porous_backward.backward_euler()

    plot_solution(x1, t1, U1, txt='Forward-Euler', azim=-20)
    plot_solution(x2, t2, U2, txt='Backward-Euler', azim=-20)
