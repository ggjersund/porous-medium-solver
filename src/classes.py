import numpy as np
import time
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
        self.r = self.k / self.h**2

    def __str__(self):
        return "Porous Medium Equation Object"

    @staticmethod
    def tridiag(a, b, c, N):
        e = np.ones(N)
        A = a*np.diag(e[1:], -1) + b*np.diag(e) + c*np.diag(e[1:], 1)
        return A

    def change_N(self, N):
        self.N = N
        self.t = np.linspace(self.T_low, self.T_high, self.N+1)
        self.U = np.zeros((self.M+1, self.N+1))
        self.k = (self.T_high - self.T_low) / self.N
        self.r = self.k / self.h**2

    def change_M(self, M):
        self.M = M
        self.x = np.linspace(self.X_low, self.X_high, self.M+1)
        self.U = np.zeros((self.M+1, self.N+1))
        self.impulses = np.zeros(self.M+1)
        self.h = (self.X_high - self.X_low) / self.M
        self.r = self.k / self.h**2

    def add_impulse(self, index, ratio=1/4):
        # Adds discrete delta function at given index
        # Note that Forward-Euler can have convergence problems for impulses
        if index < (self.M + 1):
            self.impulses[:] += signal.unit_impulse(self.M+1, index) * ratio

    def forward_euler(self):

        # Reset solution grid
        self.U = np.zeros((self.M+1, self.N+1))

        # Check CFL condition (more strict that usual 1/2)
        if ((int(np.ceil(4*self.M**2*(self.T_high - self.T_low)/((self.X_high - self.X_low)**2)) * np.amax(self.f(self.x))**(self.m-1))) >= self.N):
            raise ValueError("CFL condition not satisfied for Forward-Euler method.")

        # Set initial conditions
        self.U[:, 0] += self.impulses[:]
        self.U[:, 0] += self.f(self.x)

        # Generate A matrix and b vector
        A = self.tridiag(1, -2, 1, self.M-1)
        b = np.zeros(self.M-1)

        start_t = time.time()
        print("Forward-Euler calculation starting")

        for n in range(self.N):
            b[0] = self.g1(self.t[n])
            b[-1] = self.g2(self.t[n])
            self.U[1:-1, n+1] = self.U[1:-1, n] + (self.r * np.dot(A, self.U[1:-1, n]**(self.m+1))) + (self.r * b**(self.m+1))
            self.U[0, n+1] = b[0]
            self.U[-1, n+1] = b[-1]

        print("Forward-Euler calculation used t =", time.time() - start_t, "seconds for N:", self.N, "and M:", self.M)

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

        start_t = time.time()
        print("Backward-Euler calculation starting")

        for n in range(self.N):
            b[0] = self.g1(self.t[n])
            b[-1] = self.g2(self.t[n])

            def F(u):
                g = self.m + 1
                return u - (self.r * np.dot(A, u ** g)) - self.U[1:-1, n] - (self.r * b ** g)

            self.U[1:-1, n+1] = fsolve(F, self.U[1:-1, n])
            self.U[0, n+1] = b[0]
            self.U[-1, n+1] = b[-1]

        print("Backward-Euler calculation used t =", time.time() - start_t, "seconds for N:", self.N, "and M:", self.M)

        return self.x, self.t, self.U, self.h, self.k

    def forward_euler_convergence_space(self, analytic):
        iterations = 6
        old_N = self.N
        old_M = self.M
        self.change_N(30000)

        h_vector = np.zeros(iterations)
        L1 = np.zeros(iterations)
        L2 = np.zeros(iterations)
        Linf = np.zeros(iterations)

        start_t = time.time()
        print("Forward-Euler Convergence in space calculation starting at t = 0")

        for i in range(0, iterations):
            self.change_M(30 + (30 * i))

            x, t, U, h, k = self.forward_euler()

            X, T = np.meshgrid(x, t)
            u = np.transpose(analytic(X, T))

            h_vector[i] = h
            L1[i] = np.sum(np.abs(U - u)) * h * k
            L2[i] = np.sqrt(np.sum(abs(U - u)**2) * h * k)
            Linf[i] = np.amax(abs(U - u))

            print("Iteration:", i+1, "\tM:", self.M, "N:", self.N, "L1:", L1[i], "L2:", L2[i], "Linf:", Linf[i])

        print("Order estimation for Forward-Euler (L1):", np.polyfit(np.log(h_vector), np.log(L1), 1)[0])
        print("Order estimation for Forward-Euler (L2):", np.polyfit(np.log(h_vector), np.log(L2), 1)[0])
        print("Order estimation for Forward-Euler (Linf):", np.polyfit(np.log(h_vector), np.log(Linf), 1)[0])
        print("Forward-Euler Convergence in space calculation used t =", time.time() - start_t, "seconds")

        # Change back
        self.change_M(old_M)
        self.change_N(old_N)

        return h_vector, L1, L2, Linf

    def forward_euler_convergence_time(self, analytic):
        old_N = self.N
        old_M = self.M
        self.change_M(100)

        k_vector = np.zeros(10)
        L1 = np.zeros(10)
        L2 = np.zeros(10)
        Linf = np.zeros(10)

        start_t = time.time()
        print("Forward-Euler Convergence in time calculation starting at t = 0")

        for i in range(0, 10):
            self.change_N(5000 + (400 * i))

            x, t, U, h, k = self.forward_euler()

            X, T = np.meshgrid(x, t)
            u = np.transpose(analytic(X, T))

            k_vector[i] = k
            L1[i] = np.sum(np.abs(U - u)) * h * k
            L2[i] = np.sqrt(np.sum(abs(U - u)**2) * h * k)
            Linf[i] = np.amax(abs(U - u))

            print("Iteration:", i+1, "\tM:", self.M, "N:", self.N, "L1:", L1[i], "L2:", L2[i], "Linf:", Linf[i])

        print("Order estimation for Forward-Euler (L1):", np.polyfit(np.log(k_vector), np.log(L1), 1)[0])
        print("Order estimation for Forward-Euler (L2):", np.polyfit(np.log(k_vector), np.log(L2), 1)[0])
        print("Order estimation for Forward-Euler (Linf):", np.polyfit(np.log(k_vector), np.log(Linf), 1)[0])
        print("Forward-Euler Convergence in time calculation used t =", time.time() - start_t, "seconds")

        # Change back
        self.change_M(old_M)
        self.change_N(old_N)

        return k_vector, L1, L2, Linf

    def backward_euler_convergence_space(self, analytic):
        iterations = 6
        old_N = self.N
        old_M = self.M
        self.change_N(4000)

        h_vector = np.zeros(iterations)
        L1 = np.zeros(iterations)
        L2 = np.zeros(iterations)
        Linf = np.zeros(iterations)

        start_t = time.time()
        print("Backward-Euler Convergence in space calculation starting at t = 0")

        for i in range(0, iterations):
            self.change_M(25 + (25 * i))

            x, t, U, h, k = self.backward_euler()

            X, T = np.meshgrid(x, t)
            u = np.transpose(analytic(X, T))

            h_vector[i] = h
            L1[i] = np.sum(np.abs(U - u)) * h * k
            L2[i] = np.sqrt(np.sum(abs(U - u)**2) * h * k)
            Linf[i] = np.amax(abs(U - u))

            print("Iteration:", i+1, "\tM:", self.M, "N:", self.N, "L1:", L1[i], "L2:", L2[i], "Linf:", Linf[i])

        print("Order estimation for Backward Euler (L1):", np.polyfit(np.log(h_vector), np.log(L1), 1)[0])
        print("Order estimation for Backward Euler (L2):", np.polyfit(np.log(h_vector), np.log(L2), 1)[0])
        print("Order estimation for Backward Euler (Linf):", np.polyfit(np.log(h_vector), np.log(Linf), 1)[0])
        print("Backward-Euler Convergence in space calculation used t =", time.time() - start_t, "seconds")

        # Change back
        self.change_M(old_M)
        self.change_N(old_N)

        return h_vector, L1, L2, Linf

    def backward_euler_convergence_time(self, analytic):
        iterations = 6
        old_N = self.N
        old_M = self.M
        self.change_M(300)

        k_vector = np.zeros(iterations)
        L1 = np.zeros(iterations)
        L2 = np.zeros(iterations)
        Linf = np.zeros(iterations)

        start_t = time.time()
        print("Backward-Euler Convergence in time calculation starting at t = 0")

        for i in range(0, iterations):
            self.change_N(50 + (50 * i))

            x, t, U, h, k = self.backward_euler()

            X, T = np.meshgrid(x, t)
            u = np.transpose(analytic(X, T))

            k_vector[i] = k
            L1[i] = np.sum(np.abs(U - u)) * h * k
            L2[i] = np.sqrt(np.sum(abs(U - u)**2) * h * k)
            Linf[i] = np.amax(abs(U - u))

            print("Iteration:", i+1, "\tM:", self.M, "N:", self.N, "L1:", L1[i], "L2:", L2[i], "Linf:", Linf[i])

        print("Order estimation for Backward-Euler (L1):", np.polyfit(np.log(k_vector), np.log(L1), 1)[0])
        print("Order estimation for Backward-Euler (L2):", np.polyfit(np.log(k_vector), np.log(L2), 1)[0])
        print("Order estimation for Backward-Euler (Linf):", np.polyfit(np.log(k_vector), np.log(Linf), 1)[0])
        print("Backward-Euler Convergence in time calculation used t =", time.time() - start_t, "seconds")

        # Change back
        self.change_M(old_M)
        self.change_N(old_N)

        return k_vector, L1, L2, Linf
