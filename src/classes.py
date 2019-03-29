import numpy as np
from scipy import signal
from scipy.optimize import fsolve


class PorousMediumEquation(object):

    def __init__(self, m=1, f=0, g1=0, g2=0, M=10, N=10, T_low=0, T_high=2, X_low=-3, X_high=3):

        # Set m
        self.m = m

        # Initial condition
        self.f = f

        # Boundary conditions
        self.g1 = g1
        self.g2 = g2

        # Set axis limits
        self.X_low = X_low
        self.X_high = X_high
        self.T_low = T_low
        self.T_high = T_high

        # Set gridsize
        self.M = M
        self.N = N

        # Create grid
        self.x = np.linspace(self.X_low, self.X_high, self.M+1)
        self.t = np.linspace(self.T_low, self.T_high, self.N+1)

        # Create solution grid
        self.U = np.zeros((self.M+1, self.N+1))
        self.impulses = np.zeros(self.M+1)

        # Set stepsize
        self.h = (self.X_high - self.X_low) / self.M
        self.k = (self.T_high - self.T_low) / self.N
        self.r = (1/(self.m+1)) * (self.k / self.h**2)

    def __str__(self):
        return "Porous Medium Equation Object"

    @staticmethod
    def tridiag(a, b, c, N):
        # Create tri-diagonal matrix
        e = np.ones(N)
        A = a*np.diag(e[1:], -1) + b*np.diag(e) + c*np.diag(e[1:], 1)
        return A

    def add_impulse(self, index, ratio=1/4):
        # Adds discrete delta function at given index
        # Note that Forward-Euler can have convergence problems for impulses
        if index < (self.M + 1):
            self.impulses[:] += signal.unit_impulse(self.M+1, index) * ratio

    def forward_euler(self):

        # Reset solution grid
        self.U = np.zeros((self.M+1, self.N+1))

        # Check CFL condition (more strict that usual 1/2)
        if ((int(np.ceil(4*self.M**2*(self.T_high - self.T_low)/((self.X_high - self.X_low)**2)) * np.amax(self.f(self.x))** self.m)) >= self.N):
            raise ValueError("CFL condition not satisfied for Forward-Euler method.")

        # Set initial conditions
        self.U[:, 0] += self.impulses[:]
        self.U[:, 0] += self.f(self.x)

        # Generate A matrix and b vector
        A = self.tridiag(1, -2, 1, self.M-1)
        b = np.zeros(self.M-1)

        for n in range(self.N):
            b[0] = self.g1(self.t[n])
            b[-1] = self.g2(self.t[n])
            self.U[1:-1, n+1] = self.U[1:-1, n] + (self.r * np.dot(A, self.U[1:-1, n]**(self.m+1))) + (self.r * b**(self.m+1))
            self.U[0, n+1] = b[0]
            self.U[-1, n+1] = b[-1]

        return self.x, self.t, self.U, self.h, self.k

    def backward_euler(self):

        # Reset solution grid
        self.U = np.zeros((self.M+1, self.N+1))

        # Set initial conditions
        self.U[:, 0] += self.impulses[:]
        self.U[:, 0] += self.f(self.x)

        # Generate A matrix and b vector
        A = self.tridiag(1, -2, 1, self.M-1)
        b = np.zeros(self.M-1)

        for n in range(self.N):
            b[0] = self.g1(self.t[n])
            b[-1] = self.g2(self.t[n])

            def F(u):
                g = self.m + 1
                return u - (self.r * np.dot(A, u ** g)) - self.U[1:-1, n] - (self.r * b ** g)

            self.U[1:-1, n+1] = fsolve(F, self.U[1:-1, n])
            self.U[0, n+1] = b[0]
            self.U[-1, n+1] = b[-1]

        return self.x, self.t, self.U, self.h, self.k

    def forward_euler_convergence_space(self):
        for i in range(0, 8):
            pass

    def forward_euler_convergence_time(self):
        for i in range(0, 8):
            pass

    def backward_euler_convergence_space(self):
        for i in range(0, 8):
            pass

    def backward_euler_convergence_time(self):
        for i in range(0, 8):
            pass
