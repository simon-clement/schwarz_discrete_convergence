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


class FiniteVolumes(Discretization):
    """
        give default values of all variables.
    """


    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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
                           f,
                           bd_cond,
                           u_nm1,
                           u_interface,
                           u_nm1_interface,
                           phi_interface,
                           phi_nm1_interface,
                           phi_for_FV,
                           upper_domain=True,
                           Y=None,
                           **kwargs):
        assert phi_for_FV != []
        M, h, D, Lambda = self.M_h_D_Lambda(upper_domain)
        phi_nm1 = phi_for_FV[0]
        a, c, dt = self.get_a_c_dt()
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
            phi_nm1 = np.flipud(phi_nm1)

        if Y is None:
            Y = self.get_Y(upper_domain=upper_domain)

        # diffu_nm1 was before = u_nm1[1:] - u_nm1[:-1]
        diffu_nm1 = (phi_nm1[2:]/D[2:] * h[1:] +
                5*phi_nm1[1:-1]/D[1:-1] * (h[1:] + h[:-1]) +
                    phi_nm1[:-2]/D[:-2] * h[:-1])/12
        rhs = dt / (1 + dt * c) * (f[1:] - f[:-1] +
                                   diffu_nm1 / dt)
        if upper_domain:  # Neumann condition: user give derivative but I need flux
            cond_robin = Lambda * u_interface + phi_interface \
                - Lambda * dt / (1 + dt * c) * (f[0] + u_nm1[0] / dt)

            cond_M = bd_cond * D[-1]
            rhs = np.concatenate(([cond_robin], rhs, [cond_M]))
        else:  # Dirichlet condition: user gives value, rhs is more complicated
            cond_robin = Lambda * u_interface + phi_interface \
                - Lambda * dt / (1 + dt * c) * (f[-1] + u_nm1[-1] / dt)

            cond_M = bd_cond - dt / (1 + dt * c) * (f[0] + u_nm1[0] / dt)
            rhs = np.concatenate(([cond_M], rhs, [cond_robin]))

        phi_ret = solve_linear(Y, rhs)
        d = phi_ret / D  # We take the derivative of u

        d_kp1 = d[1:]
        d_km1 = d[:-1]
        D_mp1_2 = D[1:]
        D_mm1_2 = D[:-1]

        u_n = dt / (1 + dt * c) * (f + u_nm1 / dt
                                   + (D_mp1_2 * d_kp1 - D_mm1_2 * d_km1) / h
                                   - a * (d_kp1 + d_km1) / 2)

        assert u_n.shape[0] == M
        if upper_domain:
            u_interface = u_n[0] - h[0] * d[1] / 12 - h[0] * d[0] * 5 / 12
            phi_interface = phi_ret[0]
        else:
            u_interface = u_n[-1] + h[-1] * d[-2] / 12 + h[-1] * d[-1] * 5 / 12
            phi_interface = phi_ret[-1]
            u_n = np.flipud(u_n)
            phi_ret = np.flipud(phi_ret)

        return u_n, u_interface, phi_interface, phi_ret

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

    def integrate_one_step_star(self, f1, f2, neumann, dirichlet, u_nm1, get_phi=False):
        M1, h1, D1, _ = self.M_h_D_Lambda(upper_domain=False)
        M2, h2, D2, _ = self.M_h_D_Lambda(upper_domain=True)

        a, c, dt = self.get_a_c_dt()
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
        Y = self.get_Y_star()

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

    def get_Y(self, upper_domain=True):
        M, h, D, Lambda = self.M_h_D_Lambda(upper_domain)
        a, c, dt = self.get_a_c_dt()
        a, c, dt, Lambda = float(a), float(c), float(dt), float(Lambda)
        D = np.zeros(M + 1) + D
        h = np.zeros(M) + h
        assert isinstance(M, int)
        assert upper_domain is True or upper_domain is False

        # We first use our great function get_Y_star:
        if upper_domain:
            Y_0, Y_1, Y_2 = self.get_Y_star(M1=1)
            Y_0 = Y_0[1:]
            Y_1 = Y_1[1:]
            Y_2 = Y_2[1:]

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
            Y_0, Y_1, Y_2 = self.get_Y_star(M2=1)
            # Here Y_0 and Y_2 are inverted because we need to take the
            # symmetric
            Y_0 = Y_0[:-1]
            Y_1 = Y_1[:-1]
            Y_2 = Y_2[:-1]
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


    def get_Y_star(self, M1=-1, M2=-1):
        """
            put M1 = 1 or M2=1 to get (almost) the matrix Y1 or Y_2 (except boundaries)
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
        """
        if M1 == -1:
            M1, h1, D1, _ = self.M_h_D_Lambda(upper_domain=False)
        else:
            M1, h1, D1 = 1, 1., self.D2
        if M2 == -1:
            M2, h2, D2, _ = self.M_h_D_Lambda(upper_domain=True)
        else:
            M2, h2, D2 = 1, 1., self.D1

        a, c, dt = self.get_a_c_dt()
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

    def precompute_Y(self, f, bd_cond, upper_domain=True):
        assert upper_domain is True or upper_domain is False
        M, h, D, Lambda = self.M_h_D_Lambda(upper_domain)
        a, c, dt = self.get_a_c_dt()

        # Broadcasting / verification of type:
        D = np.zeros(M + 1) + D
        h = np.zeros(M) + h
        if not upper_domain:  # from now h, D, are 0 at bottom of the ocean.
            h, D = np.flipud(h), np.flipud(D)

        return self.get_Y(upper_domain=upper_domain)


    def eta_dirneu(self, j, s=None):
        """
            Gives the \\eta of the discretization:
            can be:
                -eta(1, ..);
                -eta(2, ..);
            returns tuple (etaj_dir, etaj_neu).
        """
        assert j == 1 or j == 2
        M, h, D, Lambda = self.M_h_D_Lambda(upper_domain=(j==2))

        a, c, dt = self.get_a_c_dt()
        if s is None:
            s = 1 / dt

        if j == 1:
            if M is None:
                M = self.M1
            if D is None:
                D = self.D1
            h = -self.SIZE_DOMAIN_1 / M
        elif j == 2: 
            if M is None:
                M = self.M2
            if D is None:
                D = self.D2
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


    def sigma_modified(self, w, order_time, order_equations):
        dt = self.DT
        s = self.s_time_modif(w, order_time) + self.C
        h1, h2 = self.get_h()
        D1 = self.D1
        D2 = self.D2
        s1, s2 = s, s
        if order_equations > 3:
            s1 = s + s**3*h1[0]**4/200/(D1*D1)
            s2 = s + s**3*h2[0]**4/200/(D2*D2)

        sig1 = np.sqrt(s1/D1)
        sig2 = -np.sqrt(s2/D2)
        return sig1, sig2

    def eta_dirneu_modif(self, j, sigj, order_operators, w, *kwargs, **dicargs):
        h1, h2 = self.get_h()
        h1, h2 = h1[0], h2[0]
        D1, D2 = self.D1, self.D2
        dt = self.DT
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
            eta_dir_modif += - hj**4*sigj**3/180 - dt**2/2*w**2/(sigj)
        return eta_dir_modif, eta_neu_modif

    def get_h(self, size_domain_1=None, size_domain_2=None, M1=None, M2=None):
        """
            Simple function to return h in each subdomains,
            in the framework of finite differences.
            returns uniform spaces between points (h1, h2).
            To recover xi, use:
            xi_1_2 = np.cumsum(np.concatenate(([0], hi)))
        """

        size_domain_1 = self.SIZE_DOMAIN_1
        size_domain_2 = self.SIZE_DOMAIN_2
        M1 = self.M1
        M2 = self.M2
        h1 = size_domain_1 / M1 + np.zeros(M1)
        h2 = size_domain_2 / M2 + np.zeros(M2)
        return h1, h2

    def get_D(self, h1=None, h2=None, function_D1=None, function_D2=None):
        """
            Simple function to return D in each subdomains,
            in the framework of finite differences.
            provide continuous functions accepting ndarray
            for D1 and D2, and returns the right coefficients.
        """
        if h1 is None or h2 is None:
            h1, h2 = self.get_h()
        if function_D1 is None:
            def function_D1(x): return self.D1 + np.zeros_like(x)
        if function_D2 is None:
            def function_D2(x): return self.D2 + np.zeros_like(x)
        # coordinates at half-points:
        x1_1_2 = np.cumsum(np.concatenate(([0], h1)))
        x2_1_2 = np.cumsum(np.concatenate(([0], h2)))
        D1 = function_D1(x1_1_2)
        D2 = function_D2(x2_1_2)
        return D1, D2

    def name(self):
        return "Volumes finis"

    def repr(self):
        return "finite volumes"
    
    def modified_equations_fun(self):
        return cv_rate.continuous_analytic_rate_robin_robin_modified_vol


if __name__ == "__main__":
    from tests import test_finite_volumes
    test_finite_volumes.launch_all_tests()
