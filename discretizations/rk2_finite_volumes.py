"""
    Finite volume for diffusion advection reaction equation.
    The functions to use are integrate_one_step and integrate_one_step_star.
    Theses function make a single step in time. It is not efficient
    (need to compute the matrix each time) but it is simple.
"""

import numpy as np
from utils_linalg import solve_linear
from discretizations.discretization import Discretization
import cv_rate


class Rk2FiniteVolumes(Discretization):
    """
        give default values of all variables.
    """

    def __init__(self,
                 A_DEFAULT=None,
                 C_DEFAULT=None,
                 D1_DEFAULT=None,
                 D2_DEFAULT=None,
                 M1_DEFAULT=None,
                 M2_DEFAULT=None,
                 SIZE_DOMAIN_1=None,
                 SIZE_DOMAIN_2=None,
                 LAMBDA_1_DEFAULT=None,
                 LAMBDA_2_DEFAULT=None,
                 DT_DEFAULT=None):
        self.A_DEFAULT, self.C_DEFAULT, self.D1_DEFAULT, self.D2_DEFAULT, \
            self.M1_DEFAULT, self.M2_DEFAULT, self.SIZE_DOMAIN_1, \
            self.SIZE_DOMAIN_2, self.LAMBDA_1_DEFAULT, \
            self.LAMBDA_2_DEFAULT, self.DT_DEFAULT = A_DEFAULT, \
            C_DEFAULT, D1_DEFAULT, D2_DEFAULT, \
            M1_DEFAULT, M2_DEFAULT, SIZE_DOMAIN_1, SIZE_DOMAIN_2, \
            LAMBDA_1_DEFAULT, LAMBDA_2_DEFAULT, DT_DEFAULT

    """
        Returns default values of a, c, dt or parameters if given.
    """

    def get_a_c_dt(self, a=None, c=None, dt=None):
        if a is None:
            a = self.A_DEFAULT
        if c is None:
            c = self.C_DEFAULT
        if dt is None:
            dt = self.DT_DEFAULT
        return a, c, dt

    """
        Entry point in the module.
        Provided equation parameters M, h, D, a, c, dt, f;
        Provided boundary condition bd_cond, phi_interface, u_interface, Lambda;
        Provided former state of the equation u_nm1;
        Returns (u_n, u_interface, phi_interface)
        u_n is the next state vector, {u, phi}_interface are the
        values of the state vector at interface,
        necessary to compute Robin conditions.

        If upper_domain is True, the considered domain is Omega_2 (atmosphere)
        bd_cond is then the Neumann condition of the top of the atmosphere.
        If upper_domain is False, the considered domain is Omega_1 (ocean)
        bd_cond is then the Dirichlet condition of the bottom of the ocean.
        h, D, f and u_nm1 have their first values ([0,1,..]) at the interface
        and their last values ([..,M-2, M-1]) at
        the top of the atmosphere (Omega_2) or bottom of the ocean (Omega_1)

        M is int
        a, c, dt, bd_cond, Lambda, u_interface, phi_interface are float
        h, D, f can be float or np.ndarray of dimension 1.
        if h is a ndarray: its size must be M
        if f is a ndarray: its size must be M
        if D is a ndarray: its size must be M+1
        u_nm1 must be a np.ndarray of dimension 1 and size M

    """

    def integrate_one_step(self,
                           M,
                           h,
                           D,
                           a,
                           c,
                           dt,
                           f,
                           f_nm1_2,
                           f_nm1,
                           bd_cond,
                           bd_cond_nm1_2,
                           bd_cond_nm1,
                           Lambda,
                           u_nm1,
                           u_interface,
                           phi_interface,
                           u_nm1_2_interface,
                           phi_nm1_2_interface,
                           u_nm1_interface,
                           phi_nm1_interface,
                           phi_for_FV,
                           upper_domain=True,
                           Y=None):
        assert phi_for_FV != []
        phi_nm1 = phi_for_FV[0] # c'est sale mais j'ai pas mieux
        a, c, dt = self.get_a_c_dt(a, c, dt)
        a, c, dt, bd_cond, Lambda, u_interface, phi_interface = float(a), \
            float(c), float(dt), float(bd_cond), float(Lambda), \
            float(u_interface), float(phi_interface)

        # Broadcasting / verification of type:
        D = np.zeros(M + 1) + D
        h = np.zeros(M) + h
        f = np.zeros(M) + f

        assert isinstance(
            u_nm1, np.ndarray) and u_nm1.ndim == 1 and u_nm1.shape[0] == M
        assert upper_domain is True or upper_domain is False

        if not upper_domain:  # from now h, D, f, u_nm1 are 0 at bottom of the ocean.
            h, D, f, u_nm1 = np.flipud(h), np.flipud(D), np.flipud(f), \
                np.flipud(u_nm1)
            f_nm1_2, f_nm1= np.flipud(f_nm1_2), np.flipud(f_nm1)

        if Y is None:
            Y = self.get_Y(M=M,
                           h=h,
                           D=D,
                           a=a,
                           c=c,
                           dt=dt,
                           Lambda=Lambda,
                           upper_domain=upper_domain)
        #actually Y is only here to get the size of the matrix xD

        # h and D consts !  (1/12phi + 10/12phi + 1/12phi = D[0]*np.diff(u)/h[0])
        assert np.linalg.norm(h-h[0]) < 1e-15
        assert np.linalg.norm(D-D[0]) < 1e-15

        Y_0, Y_1, Y_2 = Y
        Y_0[:-1] = 1/12
        Y_1[0:-1] = 10/12
        Y_2[0:] = 1/12

        if upper_domain:
            # Neumann :
            Y_0[-1] = 0
            Y_1[-1] = 1

            # Robin :
            Y_1[0] = -h[0]*Lambda * 5/(12*D[0]) + 1
            Y_2[0] = -h[0]*Lambda * 1/(12*D[0])

            def get_phi(u, u_interface, phi_interface, bd_cond):
                robin_cond = Lambda * (u_interface-u[0]) + phi_interface
                return solve_linear((Y_0, Y_1, Y_2),
                                  np.concatenate(([robin_cond],
                                                  D[0]*np.diff(u)/h[0],
                                                  [bd_cond*D[-1]])))

            def compute_k(f, u, bd_cond, u_interface, phi_interface):
                phi = get_phi(u, u_interface, phi_interface, bd_cond)
                return f + np.diff(phi)/h[0] - a*(phi[1:]/D[1:] + phi[:-1]/D[:-1])/2 - c*u
        else: # on est dans le domaine du bas
            # Dirichlet :
            Y_1[0] = -h[0] * 5/(12*D[0])
            Y_2[0] = -h[0] * 1/(12*D[0])

            # Robin :
            Y_1[-1] = h[0]*Lambda * 5/(12*D[0]) + 1
            Y_0[-1] = h[0]*Lambda * 1/(12*D[0])


            def get_phi(u, u_interface, phi_interface, bd_cond):
                robin_cond = Lambda * (u_interface-u[-1]) + phi_interface
                return solve_linear((Y_0, Y_1, Y_2),
                                  np.concatenate(([bd_cond - u[0]],
                                                  D[0]*np.diff(u)/h[0],
                                                  [robin_cond])))

            def compute_k(f, u, bd_cond, u_interface, phi_interface):
                phi = get_phi(u, u_interface, phi_interface, bd_cond)
                return f + np.diff(phi)/h - a*(phi[1:]/D[1:] + phi[:-1]/D[:-1])/2 - c*u

        k1 = compute_k(f_nm1, u_nm1, bd_cond_nm1,
                u_nm1_interface, phi_nm1_interface)

        k2 = compute_k(f_nm1_2, u_nm1 + dt/2 * k1, bd_cond_nm1_2,
                u_nm1_2_interface, phi_nm1_2_interface)

        u_n = u_nm1 + dt * k2

        phi_ret = get_phi(u_n, u_interface, phi_interface, bd_cond)

        d = phi_ret / D  # We take the derivative of u

        assert u_n.shape[0] == M
        if upper_domain:
            u_interface = u_n[0] - h[0] * d[1] / 12 - h[0] * d[0] * 5 / 12
            phi_interface = phi_ret[0]
        else:
            u_interface = u_n[-1] + h[-1] * d[-2] / 12 + h[-1] * d[-1] * 5 / 12
            phi_interface = phi_ret[-1]
            u_n = np.flipud(u_n)

        return u_n, u_interface, phi_interface

    """
        Same as integrate_one_step, but with full domain. The parameters are
        more or less the same

        Provided equation parameters (M, h, D, f){1,2}, a, c, dt;
        Provided boundary condition neumann, dirichlet
        Provided former state of the equation u_nm1;
        Returns (u_n, u_interface, phi_interface)
        u_n is the next state vector, {u, phi}_interface are the
        values of the state vector at interface,
        necessary to compare with Robin conditions.

        neumann is the Neumann condition (/!\ not a flux, just the derivative) of the top of the atmosphere.
        dirichlet is the Dirichlet condition of the bottom of the ocean.
        h{1,2}, D{1,2}, f{1,2} have their first values ([0,1,..]) at the interface
        and their last values ([..,M-2, M-1]) at
        the top of the atmosphere (h2, D2, f2) or bottom of the ocean (f1, D1, f1)
        u_nm1[0] is the bottom of the ocean
        and u_nm1[M1 + M2 - 1] is the top of the atmosphere

        M{1, 2} is int
        a, c, dt, neumann, dirichlet are float
        h{1,2}, D{1,2}, f{1,2} can be float or np.ndarray of dimension 1.
        if h is a ndarray: its size must be M{1,2}
        if f is a ndarray: its size must be M{1,2}
        if D is a ndarray: its size must be M+1
        u_nm1 must be a np.ndarray of dimension 1 and size M

    """

    def integrate_one_step_star(self, M1, M2, h1, h2, D1, D2, a, c, dt, f1, f2,
                                neumann, dirichlet, u_nm1, get_phi=False):
        a, c, dt = self.get_a_c_dt(a, c, dt)
        a, c, dt, neumann, dirichlet = float(a), float(c), float(dt), \
            float(neumann), float(dirichlet)
        assert dt > 0, "dt should be strictly positive"
        assert isinstance(
            u_nm1, np.ndarray) and u_nm1.ndim == 1 and u_nm1.shape[0] == M1 + M2

        assert isinstance(M2, int) and isinstance(M1, int)
        assert M2 > 0 and M1 > 0
        if isinstance(D1, int):
            print("Warning: type of diffusivity is int. casting to float...")
            D1 = float(D1)
        if isinstance(D2, int):
            print("Warning: type of diffusivity is int. casting to float...")
            D2 = float(D2)
        for arg, name in zip((h1, h2, D1, D2), ("h1", "h2", "D1", "D2")):
            assert isinstance(arg, float) or \
                isinstance(arg, np.float64) or \
                isinstance(arg, np.ndarray) and \
                arg.ndim == 1, name
            assert (np.array(arg) > 0).all(), name + " is negative or 0 !"

        # Broadcasting / verification of types:
        D1 = np.zeros(M1 + 1) + D1
        D2 = np.zeros(M2 + 1) + D2
        h1 = np.zeros(M1) + h1
        h2 = np.zeros(M2) + h2
        f1 = np.zeros(M1) + f1
        f2 = np.zeros(M2) + f2

        # Flipping arrays to have [0] at low altitudes rather than interface
        D1 = np.flipud(D1)
        h1 = np.flipud(h1)
        f1 = np.flipud(f1)
        Y = self.get_Y_star(M1=M1,
                            M2=M2,
                            h1=h1,
                            h2=h2,
                            D1=D1,
                            D2=D2,
                            a=a,
                            c=c,
                            dt=dt)

        f = np.concatenate((f1, f2))

        rhs = dt / (1 + dt * c) * (f[1:] - f[:-1] +
                                   (u_nm1[1:] - u_nm1[:-1]) / dt)
        dirichlet = dirichlet - dt / (1 + dt * c) * (f[0] + u_nm1[0] / dt) #val = bar(u) + phi * ..
        # and bar(u)^n+1 = bar(u)^n + dt/h (phi_3_2 - phi_1_2) (for heat equation)
        neumann = neumann * D2[-1]
        rhs = np.concatenate(([dirichlet], rhs, [neumann]))

        phi_ret = solve_linear(Y, rhs)
        d1 = phi_ret[:M1 + 1] / D1  # we go until interface
        d2 = phi_ret[M1:] / D2  # we start from interface

        d1_kp1 = d1[1:]
        d2_kp1 = d2[1:]
        d1_km1 = d1[:-1]
        d2_km1 = d2[:-1]
        D1_kp1_2 = D1[1:]
        D1_km1_2 = D1[:-1]
        D2_kp1_2 = D2[1:]
        D2_km1_2 = D2[:-1]

        u1_n = dt / (1 + dt * c) * (f[:M1] + u_nm1[:M1] / dt
                                    + (D1_kp1_2 * d1_kp1 - D1_km1_2 * d1_km1) / h1
                                    - a * (d1_kp1 + d1_km1) / 2)

        u2_n = dt / (1 + dt * c) * (f[M1:] + u_nm1[M1:] / dt
                                    + (D2_kp1_2 * d2_kp1 - D2_km1_2 * d2_km1) / h2
                                    - a * (d2_kp1 + d2_km1) / 2)

        assert u1_n.shape[0] == M1
        assert u2_n.shape[0] == M2

        u2_interface = u2_n[0] - h2[0] * d2[1] / 12 - h2[0] * d2[0] * 5 / 12
        u1_interface = u1_n[-1] + h1[-1] * d1[-2] / 12 + h1[-1] * d1[-1] * 5 / 12
        u1_bottom = u1_n[0] - h1[0] * d1[1] / 12 - h1[0] * d1[0] * 5 / 12

        #print(u1_interface, u2_interface)
        #print(h2[0], h1[-1])
        # pour avoir une quasi égalité entre les interfaces, il faut ajouter la contrainte sur la derivee seconde
        # ca donne r_(m+1,3) = - r_(m,3) + 1/3 * (d[?]-2d[?]+d)h
        # Bref, ça réécrit le système ailleurs^^
        #assert abs(u1_interface - u2_interface) < 1e-5
        # avec une telle assertion (^), on oublie le terme en h^3 (qui peut être >1e-5 si h>0.05)
        phi_interface = phi_ret[M1]

        """
        import matplotlib.pyplot as plt
        plt.plot(phi_ret, label="flux num")
        plt.plot(np.concatenate((np.diff(u1_n)/h1[1:], [0] , np.diff(u2_n)/h2[1:])), label="diff finies")
        """
        u_n = np.concatenate((u1_n, u2_n))
        
        if get_phi:
            return np.concatenate((u1_n, u2_n)), u1_interface, phi_interface, phi_ret

        return np.concatenate((u1_n, u2_n)), u1_interface, phi_interface
    """
        Returns the tridiagonal matrix Y in the shape asked by solve_linear.
        Y is the matrix we need to inverse to solve one of the half-domains.
        This function is useful to compute u_{1,2}, solution of Yu=f

        (Does not actually return a np matrix, it returns (Y_0, Y_1, Y_2).
        For details, see the documentation of utils_numeric.solve_linear)

        The returned matrix is of dimension MxM
        To compare with the full system:
            u*[0:M] = u1[0:M] (i.e. for all m, u*[m] = u1[m]
            u*[M-1:2M-1] = u2[0:M] (i.e. for all m, u*[M + m] = u2[m]

        h and D have their first values ([0]) for low altitudes
        (bottom of the ocean for Omega_1, interface for Omega_2)
        and their last values ([..M-2, M-1]) for high altitudes
        (interface for Omega_1, top of the atmosphere for Omega_2)
        /!\ it is different than in the notations or in integrate_* functions.

        h: step size (always positive) (float or ndarray, size: M)
        D: diffusivity (always positive) (float or ndarray, size: M+1)
            Note: if D is a np.ndarray, it should be given on the half-steps,
                    i.e. D[m] is D_{m+1/2}
        a: advection coefficient (should be positive) (float)
        c: reaction coefficient (should be positive) (float)
    """

    def get_Y(self, M, h, D, a, c, dt, Lambda, upper_domain=True):
        a, c, dt = self.get_a_c_dt(a, c, dt)
        a, c, dt, Lambda = float(a), float(c), float(dt), float(Lambda)
        D = np.zeros(M + 1) + D
        h = np.zeros(M) + h
        assert isinstance(M, int)
        assert upper_domain is True or upper_domain is False

        # We first use our great function get_Y_star:
        if upper_domain:
            Y_0, Y_1, Y_2 = self.get_Y_star(M1=1,
                                            M2=M,
                                            h1=1.0,
                                            h2=h,
                                            D1=D[0],
                                            D2=D,
                                            a=a,
                                            c=c,
                                            dt=dt)
            Y_0 = Y_0[1:]
            Y_1 = Y_1[1:]
            Y_2 = Y_2[1:]
            Y_0[:-1] = 1/12
            Y_1[:-1] = 10/12
            Y_2[:] = 1/12

            # Now we have the tridiagonal matrices, except for the Robin bd
            # condition
            dirichlet_cond_extreme_point = -dt / (1 + dt * c) * (
                1 / h[0] + a / (2 * D[0])) - h[0] * 5 / (12 * D[0])
            dirichlet_cond_interior_point = dt / (1 + dt * c) * (
                1 / h[0] - a / (2 * D[1])) - h[0] / (12 * D[1])
            # Robin bd condition are Lambda * Dirichlet + Neumann:
            # Except we work with fluxes:
            # Neumann condition is actually a Dirichlet bd condition
            # and Dirichlet is just a... pseudo-differential operator
            Y_1[0] = Lambda * dirichlet_cond_extreme_point + 1
            Y_2[0] = Lambda * dirichlet_cond_interior_point
        else:
            Y_0, Y_1, Y_2 = self.get_Y_star(M1=M,
                                            M2=1,
                                            h1=h,
                                            h2=1.0,
                                            D1=D,
                                            D2=D[0],
                                            a=a,
                                            c=c,
                                            dt=dt)
            # Here Y_0 and Y_2 are inverted because we need to take the
            # symmetric
            Y_0 = Y_0[:-1]
            Y_1 = Y_1[:-1]
            Y_2 = Y_2[:-1]
            Y_0[:] = 1/12
            Y_1[1:] = 10/12
            Y_2[1:] = 1/12
            # Now we have the tridiagonal matrices, except for the Robin bd
            # condition
            dirichlet_cond_extreme_point = dt / (1 + dt * c) * (
                1 / h[-1] - a / (2 * D[-1])) + h[-1] * 5 / (12 * D[-1])
            dirichlet_cond_interior_point = dt / (1 + dt * c) * (
                -1 / h[-1] - a / (2 * D[-2])) + h[-1] / (12 * D[-2])
            # Robin bd condition are Lambda * Dirichlet + Neumann:
            # Except we work with fluxes:
            # Neumann condition is actually a Dirichlet bd condition
            # and Dirichlet is just a... pseudo-differential operator
            Y_1[-1] = Lambda * dirichlet_cond_extreme_point + 1
            Y_0[-1] = Lambda * dirichlet_cond_interior_point
            # We take the flipped, symmetric of the matrix:
        return (Y_0, Y_1, Y_2)

    """
        Returns the tridiagonal matrix Y* in the shape asked by solve_linear.
        Y* is the matrix we need to inverse to solve the full domain.
        This function is useful to compute u*, solution of Y*u*=f*
        It is also used in get_Y, with one of the arguments M_{1 | 2} = 1

        (Does not actually return a np matrix, it returns (Y_0, Y_1, Y_2).
        For details, see the documentation of utils_numeric.solve_linear)

        The returned matrix is of dimension M_starxM_star
        To compare with the coupled system and get:
            u*[0:M] = u1[0:M] (i.e. for all m, u*[m] = u1[m]
            u*[M-1:2M-1] = u2[0:M] (i.e. for all m, u*[M + m] = u2[m]
        We should have:
            - M_star = M1+M2 - 1
            - D_star[0:M+1] = D1[0:M+1]
            - h_star[0:M] = h1[0:M]
            - D_star[M-1:2M-1] = D2[0:M]
            - h_star[M-1:2M-1] = h2[0:M]

        D1[-1] and D2[0] should both be the diffusivity at interface.
        (2 values because it can be discontinuous, so D1[-1] is D(0^{-}) )

        h{1,2}: step size (always positive) (float or ndarray, size: M{1,2})
        D{1,2}: diffusivity (always positive) (float or ndarray, size: M{1,2}+1)
            Note: if D{1,2} is a np.ndarray, it should be given on the half-steps,
                    i.e. D{1,2}[m] is D{1,2}_{m+1/2}
        a{1,2}: advection coefficient (should be positive) (float)
        c{1,2}: reaction coefficient (should be positive) (float)

    """

    def get_Y_star(self, M1, M2, h1, h2, D1, D2, a, c, dt):
        a, c, dt = self.get_a_c_dt(a, c, dt)
        a, c, dt = float(a), float(c), float(dt)
        if a < 0:
            print("Warning : a should probably not be negative")
        if c < 0:
            print("Warning : c should probably not be negative")

        assert isinstance(M2, int) and isinstance(M1, int)
        assert M2 > 0 and M1 > 0
        if isinstance(D1, int):
            print("Warning: type of diffusivity is int. casting to float...")
            D1 = float(D1)
        if isinstance(D2, int):
            print("Warning: type of diffusivity is int. casting to float...")
            D2 = float(D2)
        for arg, name in zip((h1, h2, D1, D2), ("h1", "h2", "D1", "D2")):
            assert isinstance(arg, float) or \
                isinstance(arg, np.float64) or \
                isinstance(arg, np.ndarray) and \
                arg.ndim == 1, name
            assert (np.array(arg) > 0).all(), name + " is negative or 0 !"

        # Broadcast or verification of size:
        D1 = np.zeros(M1 + 1) + D1
        h1 = np.zeros(M1) + h1
        D2 = np.zeros(M2 + 1) + D2
        h2 = np.zeros(M2) + h2
        # In the notations h is negative when considering \omega_1:

        # assert (h1 == h1[::-1]).all() # Warning with this,
        # it means they are constant... The code is not purely consistant
        # with itself though, so for now let's keep theses assertions
        #assert (D1 == D1[::-1]).all()
        #D1 = D1[::-1]

        # D_minus means we take the value of D1 for the interface
        D_minus = np.concatenate((D1[1:], D2[1:-1]))
        # D_plus means we take the value of D1 for the interface
        D_plus = np.concatenate((D1[1:-1], D2[:-1]))
        # D_mm1_2 is a D_plus means we take the value of D1 for the interface
        D_mm1_2 = np.concatenate((D1[:-1], D2[:-2]))
        # D_mm1_2 is a D_minus means we take the value of D1 for the interface
        D_mp3_2 = np.concatenate((D1[2:], D2[1:]))
        h = np.concatenate((h1, h2))
        h_m = h[:-1]
        h_mp1 = h[1:]

        # LEFT DIAGONAL: phi_{1/2} -> phi_{M+1/2}
        Y_0 = -dt / (1 + dt * c) * (1 / h_m + a / (2 * D_mm1_2)) \
                    + h_m / D_mm1_2 * (1/12) * (2 - (D_minus/D_plus)*h_mp1**2 / h_m**2)
        # Now we can put Neumann bd condition:
        Y_0 = np.concatenate((Y_0, [0]))  # Neumann bd condition
        # (actually Dirichlet bd because we work on the fluxes...)

        # RIGHT DIAGONAL:
        Y_2 = -dt / (1 + dt * c) * (1 / h_mp1 - a /
                                    (2 * D_mp3_2)) + h_mp1 / (12 * D_mp3_2)
        Y_2_bd = dt / (1 + dt * c) * (1 / h[0] - a /
                                      (2 * D1[1])) - h[0] / (12 * D1[1])
        Y_2 = np.concatenate(([Y_2_bd], Y_2))

        # MAIN DIAGONAL:
        Y_1 = dt / (1 + dt * c) * (1 / h_m + 1 / h_mp1 + a / 2 * (1 /
                                                                  D_plus - 1 / D_minus)) \
                + (h_m / D_minus + h_mp1 / D_plus) / 3 \
                + (h_m + h_mp1) / (12 * D_plus)
        Y_1_bd = -dt / (1 + dt * c) * (1 / h[0] + a /
                                       (2 * D1[0])) - h[0] * 5 / (12 * D1[0])
        Y_1 = np.concatenate(([Y_1_bd], Y_1, [1]))

        assert Y_1.shape[0] == M1 + M2 + 1
        assert Y_0.shape[0] == M1 + M2 == Y_2.shape[0]

        return (Y_0, Y_1, Y_2)

    """
        Precompute Y for integrate_one_step. useful when integrating over a long time
        The arguments are exactly the arguments of @integrate_one_step,
        except for u_nm1, and interface conditions.
        f is kept as an argument but is not used.
    """

    def precompute_Y(self,
                     M,
                     h,
                     D,
                     a,
                     c,
                     dt,
                     f,
                     bd_cond,
                     Lambda,
                     upper_domain=True):
        a, c, dt = self.get_a_c_dt(a, c, dt)
        a, c, dt, bd_cond, Lambda, = float(a), \
            float(c), float(dt), float(bd_cond), float(Lambda)

        # Broadcasting / verification of type:
        D = np.zeros(M + 1) + D
        h = np.zeros(M) + h

        assert upper_domain is True or upper_domain is False

        if not upper_domain:  # from now h, D, are 0 at bottom of the ocean.
            h, D = np.flipud(h), np.flipud(D)

        return self.get_Y(M=M,
                          h=h,
                          D=D,
                          a=a,
                          c=c,
                          dt=dt,
                          Lambda=Lambda,
                          upper_domain=upper_domain)


    def eta_dirneu(self, j, s=None, a=None, c=None, dt=None, M=None, D=None):
        """
            Gives the \\eta of the discretization:
            can be:
                -eta(1, ..);
                -eta(2, ..);
            returns tuple (etaj_dir, etaj_neu).
        """
        assert j == 1 or j == 2

        a, c, dt = self.get_a_c_dt(a, c, dt)
        if s is None:
            s = 1 / dt

        if j == 1:
            if M is None:
                M = self.M1_DEFAULT
            if D is None:
                D = self.D1_DEFAULT
            h = -self.SIZE_DOMAIN_1 / M
        elif j == 2: 
            if M is None:
                M = self.M2_DEFAULT
            if D is None:
                D = self.D2_DEFAULT
            h = self.SIZE_DOMAIN_2 / M

        Y_0 = -1 / (s + c) * (1 / h + a / (2 * D)) + h / (12 * D)
        Y_1 = 1 / (s + c) * 2 / h + 10 * h / (12 * D)
        Y_2 = -1 / (s + c) * (1 / h - a / (2 * D)) + h / (12 * D)

        lambda_moins = (Y_1 - np.sqrt(Y_1**2 - 4 * Y_0 * Y_2)) \
                                / (-2 * Y_2)
        lambda_plus = (Y_1 + np.sqrt(Y_1**2 - 4 * Y_0 * Y_2)) \
                                / (-2 * Y_2)

        # The computation is then different because the boundary condition is different (and we are in the finite domain case)
        if j == 1:
            lambda_moins, lambda_plus = lambda_plus, lambda_moins # we invert the l- and l+ to have |l-|<1
            xi = (-h/(12*D) * (s+c) + 1/h + a/(2*D)) / (5*h/(12*D) * (s+c) + 1/h - a/(2*D))
            eta1_dir = (-(1/h + a/(2*D))/(s+c) - 5*h/(12*D)) \
                    * (1 - (lambda_moins - xi) / (lambda_plus - xi) * (lambda_moins / lambda_plus)**(M-1)) \
                    + ((1/h - a/(2*D))/(s+c) - h/(12*D)) \
                    * (lambda_moins - lambda_plus* (lambda_moins - xi) / (lambda_plus - xi) *(lambda_moins / lambda_plus)**M)
            eta1_neu = 1 + (lambda_moins-xi) / (lambda_plus - xi) *(lambda_moins / lambda_plus) ** (M - 1)
            return eta1_dir, eta1_neu
        elif j == 2:
            eta2_dir = (-(1/h + a/(2*D))/(s+c) - 5*h/(12*D)) \
                    * (1 - (lambda_moins / lambda_plus)**M) \
                    + ((1/h - a/(2*D))/(s+c) - h/(12*D)) \
                    * (lambda_moins - lambda_plus*(lambda_moins / lambda_plus)**M)
            eta2_neu = 1 + (lambda_moins / lambda_plus) ** M
            return eta2_dir, eta2_neu


    """
        When D and h are constant, it is possible to find the convergence
        rate in frequency domain. analytic_robin_robin computes this convergence rate.
        s is 1/dt when considering the local-in-time case, otherwise it
        should be iw (with w the desired frequency)
        In the discrete time setting, the Z transform gives s = 1. / dt * (z - 1) / z
        for implicit euler discretisation.
    """

    def analytic_robin_robin_legacy(self,
                             s=None,
                             Lambda_1=None,
                             Lambda_2=None,
                             a=None,
                             c=None,
                             dt=None,
                             M1=None,
                             M2=None,
                             D1=None,
                             D2=None,
                             verbose=False):
        a, c, dt = self.get_a_c_dt(a, c, dt)

        if Lambda_1 is None:
            Lambda_1 = self.LAMBDA_1_DEFAULT
        if Lambda_2 is None:
            Lambda_2 = self.LAMBDA_2_DEFAULT
        if M1 is None:
            M1 = self.M1_DEFAULT
        if M2 is None:
            M2 = self.M2_DEFAULT
        if D1 is None:
            D1 = self.D1_DEFAULT
        if D2 is None:
            D2 = self.D2_DEFAULT
        if s is None:
            s = 1 / dt

        h1 = self.SIZE_DOMAIN_1 / M1
        h2 = self.SIZE_DOMAIN_2 / M2

        Y1_0 = -1 / (s + c) * (1 / h1 - a / (2 * D1)) + h1 / (12 * D1)
        Y1_1 = 1 / (s + c) * 2 / h1 + 5 * h1 / (12 * D1)
        Y1_2 = -1 / (s + c) * (1 / h1 + a / (2 * D1)) + h1 / (12 * D1)

        Y2_0 = -1 / (s + c) * (1 / h2 + a / (2 * D2)) + h2 / (12 * D2)
        Y2_1 = 1 / (s + c) * 2 / h2 + 5 * h2 / (12 * D2)
        Y2_2 = -1 / (s + c) * (1 / h2 - a / (2 * D2)) + h2 / (12 * D2)
        lambda2_moins = (Y2_1 - np.sqrt(Y2_1**2 - 4 * Y2_0 * Y2_2)) \
                                / (-2 * Y2_2)
        lambda1_moins = (Y1_1 - np.sqrt(Y1_1**2 - 4 * Y1_0 * Y1_2)) \
                                / (-2 * Y1_2)
        raise BaseException("I have no idea what this part refers to")
        eta2_0 = ((lambda2_moins - 1) / h2 - a * (lambda2_moins + 1) /
                  (2 * D2)) / (s + c) - h2 * (lambda2_moins + 5) / (12 * D2) # was + 2 and 6 * D2
        eta1_0 = ((1 - lambda1_moins) / h1 - a * (lambda1_moins + 1) /
                  (2 * D1)) / (s + c) + h1 * (lambda1_moins + 5) / (12 * D1) # ..

        rho_numerator = (Lambda_2 * eta1_0 + 1) * (Lambda_1 * eta2_0 + 1)
        rho_denominator = (Lambda_2 * eta2_0 + 1) * (Lambda_1 * eta1_0 + 1)

        return np.abs(rho_numerator / rho_denominator)

    def sigma_modified(self, w, order_time, order_equations):
        h1, h2 = self.get_h()
        h1, h2 = h1[0], h2[0]
        D1, D2 = self.D1_DEFAULT, self.D2_DEFAULT
        dt = self.DT_DEFAULT

        s = self.s_time_modif(w, dt, order_time) + self.C_DEFAULT
        s1 = s
        #if order_equations > 2: #warning, euler coefficients
        #    s1 -= dt**3 / 24 * w**4

        s2 = s
        #if order_equations > 2: #warning, euler coefficients
        #    s2 -= dt**3 / 24 * w**4

        sig1 = np.sqrt(s1/self.D1_DEFAULT)
        sig2 = -np.sqrt(s2/self.D2_DEFAULT)
        return sig1, sig2

    def eta_dirneu_modif(self, j, sigj, order_operators, w, *kwargs, **dicargs):
        # This code should not run and is here as an example
        h1, h2 = self.get_h()
        h1, h2 = h1[0], h2[0]
        D1, D2 = self.D1_DEFAULT, self.D2_DEFAULT
        dt = self.DT_DEFAULT
        if j==1:
            hj = h1
            Dj = D1
        else:
            hj = h2
            Dj = D2
        eta_neu_modif = Dj
        eta_dir_modif = 1/sigj
        if order_operators > 0:
            eta_dir_modif += hj**2*sigj/12 
        if order_operators > 1:
            eta_dir_modif += - hj**4*sigj**3/180- dt**2/2*w**2/(sigj) # warning, time term comes from euler
        return eta_dir_modif, eta_neu_modif

    def s_time_modif(self, w, dt, order):
        s = w * 1j - w**3 * dt**2/6 * 1j
        return s

    """
        Simple function to return h in each subdomains,
        in the framework of finite differences.
        returns uniform spaces between points (h1, h2).
        To recover xi, use:
        xi_1_2 = np.cumsum(np.concatenate(([0], hi)))
    """

    def get_h(self, size_domain_1=None, size_domain_2=None, M1=None, M2=None):
        if size_domain_1 is None:
            size_domain_1 = self.SIZE_DOMAIN_1
        if size_domain_2 is None:
            size_domain_2 = self.SIZE_DOMAIN_2
        if M1 is None:
            M1 = self.M1_DEFAULT
        if M2 is None:
            M2 = self.M2_DEFAULT
        h1 = size_domain_1 / M1 + np.zeros(M1)
        h2 = size_domain_2 / M2 + np.zeros(M2)
        # use the following line to use a functional:
        #h1 = np.diff(np.cumsum(np.concatenate(([0],h1)))**1)
        #h2 = np.diff(np.cumsum(np.concatenate(([0],h2)))**1)
        return h1, h2

    """
        Simple function to return D in each subdomains,
        in the framework of finite differences.
        provide continuous functions accepting ndarray
        for D1 and D2, and returns the right coefficients.
    """

    def get_D(self, h1, h2, function_D1=None, function_D2=None):
        if function_D1 is None:
            def function_D1(x): return self.D1_DEFAULT + np.zeros_like(x)
        if function_D2 is None:
            def function_D2(x): return self.D2_DEFAULT + np.zeros_like(x)
        # coordinates at half-points:
        x1_1_2 = np.cumsum(np.concatenate(([0], h1)))
        x2_1_2 = np.cumsum(np.concatenate(([0], h2)))
        D1 = function_D1(x1_1_2)
        D2 = function_D2(x2_1_2)
        return D1, D2

    def name(self):
        return "Volumes finis, RK2"

    def repr(self):
        return "finite volumes rk2"
    
    def modified_equations_fun(self):
        return cv_rate.continuous_analytic_rate_robin_robin_modified_vol_rk4


if __name__ == "__main__":
    from tests import test_finite_volumes
    test_finite_volumes.launch_all_tests()
