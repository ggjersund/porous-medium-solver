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

    def initial1(x):
        #return molli1(x)
        #return molli2(x)
        return barenblatt(x, t=0.01, g=2, n=1)

    def initial2(x):
        #return molli1(x)
        #return molli2(x)
        return barenblatt(x, t=0.5, g=2, n=1)

    def initial3(x):
        return molli2(x)

    def analytic(x, t):
        return barenblatt(x, t, g=2, n=1)

    def boundary1(t):
        return (t-t) + 0

    def boundary2(t):
        return (t-t) + 0

    def boundary11(t):
        return (t-t) + 1

    def boundary22(t):
        return (t-t) + 1

    def boundary111(t):
        return (t-t) - 1/8

    def boundary222(t):
        return (t-t) + 1/8

    """
    Dirichlet boundary conditions
    """
    porous_forward = PorousMediumEquation(m=1, f=initial1, g1=boundary1, g2=boundary2, M=90, N=5000, T_low=0.01, T_high=2.01, X_low=-8, X_high=8)
    porous_backward = PorousMediumEquation(m=1, f=initial1, g1=boundary1, g2=boundary2, M=200, N=400, T_low=0.01, T_high=2.01, X_low=-8, X_high=8)

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
    Dirichlet with mollification as initial condition
    """
    porous_forward = PorousMediumEquation(m=1, f=initial3, g1=boundary11, g2=boundary22, M=90, N=5000, T_low=0.01, T_high=2.01, X_low=-8, X_high=8)
    porous_backward = PorousMediumEquation(m=1, f=initial3, g1=boundary11, g2=boundary22, M=200, N=400, T_low=0.01, T_high=2.01, X_low=-8, X_high=8)

    x1, t1, U1, h1, k1 = porous_forward.forward_euler()
    x2, t2, U2, h2, k2 = porous_backward.backward_euler()

    plot_solution(x1, t1, U1, txt='Forward-Euler solution', azim=-30)
    plot_solution(x2, t2, U2, txt='Backward-Euler solution', azim=-30)

    """
    Von-Neumann boundary conditions
    """
    porous_forward = PorousMediumEquation(m=1, f=initial1, g1=boundary1, g2=boundary2, M=90, N=20000, T_low=0.01, T_high=2.01, X_low=-2, X_high=2)
    x2, t2, U2, h2, k2 = porous_forward.forward_euler(neumann=True)
    plot_solution(x2, t2, U2, txt='Forward-Euler with Neumann boundary conditions', azim=-30)

    porous_backward = PorousMediumEquation(m=1, f=initial1, g1=boundary1, g2=boundary2, M=100, N=700, T_low=0.01, T_high=2.01, X_low=-2, X_high=2)
    x2, t2, U2, h2, k2 = porous_backward.backward_euler(neumann=True)
    plot_solution(x2, t2, U2, txt='Backward-Euler with Neumann boundary conditions', azim=-30)

    """
    Von-Neumann boundary conditions with impulses
    """
    porous_forward = PorousMediumEquation(m=1, f=initial3, g1=boundary111, g2=boundary222, M=90, N=20000, T_low=0.01, T_high=2.01, X_low=-2, X_high=2)
    porous_forward.add_impulse(index=81, ratio=0.72)
    porous_forward.add_impulse(index=79, ratio=0.35)
    porous_forward.add_impulse(index=84, ratio=0.65)
    x2, t2, U2, h2, k2 = porous_forward.forward_euler(neumann=True)
    plot_solution(x2, t2, U2, txt='Forward-Euler with Neumann boundary conditions', azim=-30)

    porous_backward = PorousMediumEquation(m=1, f=initial3, g1=boundary111, g2=boundary222, M=100, N=700, T_low=0.01, T_high=2.01, X_low=-2, X_high=2)
    porous_backward.add_impulse(index=81, ratio=0.72)
    porous_backward.add_impulse(index=79, ratio=0.35)
    porous_backward.add_impulse(index=84, ratio=0.65)
    x2, t2, U2, h2, k2 = porous_backward.backward_euler(neumann=True)
    plot_solution(x2, t2, U2, txt='Backward-Euler with Neumann boundary conditions', azim=-30)

    """
    Forward-Euler convergence plots
    """
    porous_forward = PorousMediumEquation(m=1, f=initial2, g1=boundary1, g2=boundary2, M=90, N=5000, T_low=0.5, T_high=2.5, X_low=-8, X_high=8)

    h_vector, L1, L2, Linf = porous_forward.forward_euler_convergence_space(analytic)
    plot_convergence(
        x=np.array([h_vector]),
        e=np.array([L1]),
        labels=np.array([r"$||e||_{L^1}$"]),
        xlabel=r"$h$",
        ylabel=r"$||e||$",
        txt='Forward-Euler space convergence'
    )

    """
    Backward-Euler convergence plots
    """
    porous_backward = PorousMediumEquation(m=1, f=initial2, g1=boundary1, g2=boundary2, M=200, N=900, T_low=0.5, T_high=2.5, X_low=-8, X_high=8)

    h_vector, L1, L2, Linf = porous_backward.backward_euler_convergence_space(analytic)
    plot_convergence(
        x=np.array([h_vector]),
        e=np.array([L1]),
        labels=np.array([r"$||e||_{L^1}$"]),
        xlabel=r"$h$",
        ylabel=r"$||e||$",
        txt='Backward-Euler space convergence'
    )

    k_vector, L1, L2, Linf = porous_backward.backward_euler_convergence_time(analytic)
    plot_convergence(
        x=np.array([k_vector]),
        e=np.array([L1]),
        labels=np.array([r"$||e||_{L^1}$"]),
        xlabel=r"$k$",
        ylabel=r"$||e||$",
        txt='Backward-Euler time convergence'
    )
