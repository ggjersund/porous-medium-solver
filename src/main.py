import numpy as np

from functions import plot_solution, plot_convergence
from classes import PorousMediumEquation


if __name__ == "__main__":

    def barenblatt(x, t, g, n):
        m = g
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
        return barenblatt(x, t=0.01, g=2, n=1)

    def analytic(x, t):
        return barenblatt(x, t, g=2, n=1)

    def boundary1(t):
        return (t-t) + 0

    def boundary2(t):
        return (t-t) + 0

    porous_forward = PorousMediumEquation(m=1, f=initial, g1=boundary1, g2=boundary2, M=90, N=5000, T_low=0.01, T_high=2, X_low=-8, X_high=8)
    porous_backward = PorousMediumEquation(m=1, f=initial, g1=boundary1, g2=boundary2, M=200, N=900, T_low=0.01, T_high=2, X_low=-8, X_high=8)
    #porous_forward.add_impulse(index=54, ratio=0.75)

    x1, t1, U1, h1, k1 = porous_forward.forward_euler()
    x2, t2, U2, h2, k2 = porous_backward.backward_euler()

    X1, T1 = np.meshgrid(x1, t1)
    X2, T2 = np.meshgrid(x2, t2)
    u1 = np.transpose(analytic(X1, T1))
    u2 = np.transpose(analytic(X2, T2))

    plot_solution(x1, t1, U1, txt='Forward-Euler solution', azim=-30)
    plot_solution(x2, t2, U2, txt='Backward-Euler solution', azim=-30)

    plot_solution(x1, t1, U1 - u1, txt='Forward-Euler error', azim=-30)
    plot_solution(x2, t2, U2 - u2, txt='Backward-Euler error', azim=-30)

    """
    Forward-Euler convergence plots
    """
    h_vector, L1, L2, Linf = porous_forward.forward_euler_convergence_space(analytic)
    plot_convergence(
        x=np.array([h_vector, h_vector, h_vector]),
        e=np.array([L1, L2, Linf]),
        labels=np.array(['L1', 'L2', 'Linf']),
        xlabel='h',
        ylabel='e',
        txt='Forward-Euler space convergence'
    )
    k_vector, L1, L2, Linf = porous_forward.forward_euler_convergence_time(analytic)
    plot_convergence(
        x=np.array([k_vector, k_vector, k_vector]),
        e=np.array([L1, L2, Linf]),
        labels=np.array(['L1', 'L2', 'Linf']),
        xlabel='k',
        ylabel='e',
        txt='Forward-Euler time convergence'
    )

    """
    Backward-Euler convergence plots
    """
    h_vector, L1, L2, Linf = porous_backward.backward_euler_convergence_space(analytic)
    plot_convergence(
        x=np.array([h_vector, h_vector, h_vector]),
        e=np.array([L1, L2, Linf]),
        labels=np.array(['L1', 'L2', 'Linf']),
        xlabel='h',
        ylabel='e',
        txt='Backward-Euler space convergence'
    )
    k_vector, L1, L2, Linf = porous_backward.backward_euler_convergence_time(analytic)
    plot_convergence(
        x=np.array([k_vector, k_vector, k_vector]),
        e=np.array([L1, L2, Linf]),
        labels=np.array(['L1', 'L2', 'Linf']),
        xlabel='k',
        ylabel='e',
        txt='Backward-Euler time convergence'
    )
