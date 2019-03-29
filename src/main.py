import numpy as np

from functions import plot_solution, plot_convergence
from classes import PorousMediumEquation


if __name__ == "__main__":

    def barenblatt(x, t, m, n):
        alpha = n / ((n * (m - 1)) + 2)
        beta = alpha / n
        k = beta * (m - 1) / (2 * m)
        s = 0.5 - ((k * (np.abs(x)**2)) / (t**(2 * beta)))
        return (1 / (t**alpha)) * ((np.maximum(s, 0))**(1 / (m - 1)))

    def molli1(x):
        f = np.zeros(len(x))
        y = np.where(abs(x) < 1)
        for i in y:
            f[i] = np.exp((-1 / (1 - abs(x[i]) ** 2)))
        return f

    def molli2(x):
        f = np.ones(len(x))
        y = np.where(abs(x) < 1)
        for i in y:
            f[i] -= np.exp((-1 / (1 - abs(x[i]) ** 2)))
        return f

    def initial(x):
        #return molli1(x)
        #return molli2(x)
        return barenblatt(x, t=0.01, m=2, n=1)

    def analytic(x, t):
        return barenblatt(x, t, m=2, n=1)

    def boundary1(t):
        return (t-t) + 0

    def boundary2(t):
        return (t-t) + 0



    porous_forward = PorousMediumEquation(m=1.3, f=initial, g1=boundary1, g2=boundary2, M=90, N=5000, T_low=0.01, T_high=2)
    porous_backward = PorousMediumEquation(m=2, f=initial, g1=boundary1, g2=boundary2, M=200, N=400, T_low=0.01, T_high=2, X_low=-8, X_high=8)
    #porous_forward.add_impulse(index=54, ratio=0.75)

    #x1, t1, U1, h1, k1 = porous_forward.forward_euler()
    x2, t2, U2, h2, k2 = porous_backward.backward_euler()

    #plot_solution(x1, t1, U1, txt='Forward-Euler', azim=-30)
    plot_solution(x2, t2, U2, txt='Backward-Euler', azim=-30)

    u = np.zeros((len(x2), len(t2)))
    for i in range(len(x2)):
        for j in range(len(t2)):
            u[i, j] = barenblatt(x2[i], t2[j], m=2, n=1)
    print(u.shape)
    print(U2.shape)
    plot_solution(x2, t2, u, txt='Analytic', azim=-30)

    """
    h_vector = np.zeros(10)
    L1 = np.zeros(10)
    L2 = np.zeros(10)
    Linf = np.zeros(10)

    for i in range(0, 10):
        M = 30 + (20 * i)
        porous_forward = PorousMediumEquation(m=1.3, f=initial, g1=boundary1, g2=boundary2, M=M, N=10000, T_low=0.01, T_high=2)
        x, t, U, h, k = porous_forward.forward_euler()

        X, T = np.meshgrid(x, t)
        u = np.transpose(analytic(X, T))

        h_vector[i] = h
        L1[i] = np.sum(np.abs(U - u)) * h * k
        L2[i] = np.sqrt(np.sum(abs(U - u)**2) * h * k)
        Linf[i] = np.amax(abs(U - u))

        plot_solution(x, t, u, txt='Forward-Euler', azim=-30)

        print("Iteration:", i+1, "M:", M, "N:", 10000, "L1:", L1[i], "L2:", L2[i], "Linf:", Linf[i])

    plot_convergence(
        x=np.array([h_vector, h_vector, h_vector]),
        e=np.array([L1, L2, Linf]),
        labels=np.array(['L1', 'L2', 'Linf']),
        xlabel='h',
        ylabel='e',
        txt='Convergence plot'
    )
    """
