"""
    Finite differences for diffusion advection reaction equation.
    The functions to use are integrate_one_step and integrate_one_step_star.
    Theses function make a single step in time. It is not efficient
    (need to compute the matrix each time) but it is simple.
"""

import numpy as np
from discretizations.discretization import Discretization
from utils_linalg import solve_linear_with_ultra_right, solve_linear


class FiniteDifferencesNoCorrectiveTerm(Discretization):
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
        Entry point in the class.
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
                           bd_cond,
                           Lambda,
                           u_nm1,
                           u_interface,
                           phi_interface,
                           upper_domain=True,
                           Y=None):
        a, c, dt = self.get_a_c_dt(a, c, dt)
        a, c, dt, bd_cond, Lambda, u_interface, phi_interface = float(a), \
            float(c), float(dt), float(bd_cond), float(Lambda), \
            float(u_interface), float(phi_interface)

        # Broadcasting / verification of type:
        D = np.zeros(M - 1) + D
        h = np.zeros(M - 1) + h
        f = np.zeros(M) + f

        assert isinstance(
            u_nm1, np.ndarray) and u_nm1.ndim == 1 and u_nm1.shape[0] == M
        assert upper_domain is True or upper_domain is False

        if Y is None:
            Y = self.precompute_Y(M=M,
                                  h=h,
                                  D=D,
                                  a=a,
                                  c=c,
                                  dt=dt,
                                  f=f,
                                  bd_cond=bd_cond,
                                  Lambda=Lambda,
                                  upper_domain=upper_domain)

        rhs = (f[1:-1] + u_nm1[1:-1] / dt) * (h[1:] + h[:-1])

        # extrapolation ? no! it is inside phi_interface
        cond_robin = Lambda * u_interface + phi_interface

        rhs = np.concatenate(([cond_robin], rhs, [bd_cond]))

        u_n = solve_linear_with_ultra_right(Y, rhs)

        new_u_interface = u_n[0]
        # extrapolation of flux: f(0) ~ f(h/2) - h/2*f'(h)
        phi_1_2 = D[0] / h[0] * (u_n[1] - u_n[0])
        phi_3_2 = D[1] / h[1] * (u_n[2] - u_n[1])
        new_phi_interface = ((2*h[0]+h[1])*phi_1_2 - h[0]*phi_3_2) / (h[0] + h[1])

        assert u_n.shape[0] == M
        return u_n, new_u_interface, new_phi_interface

    """
        See integrate_one_step. This function integrates in time
        the full system: should work better if D1[0] == D2[0].
        h1 should be negative, and h1[0] is the interface.
    """

    def integrate_one_step_star(self, M1, M2, h1, h2, D1, D2, a, c, dt, f1, f2,
                                neumann, dirichlet, u_nm1):
        a, c, dt = self.get_a_c_dt(a, c, dt)
        a, c, dt, neumann, dirichlet = float(a), float(c), float(dt), \
            float(neumann), float(dirichlet)
        # Theses assertions cannot be used because of the unit tests:
        # for arg, name in zip((a, c, neumann, dirichlet),
        #         ("a", "c", "neumann", "dirichlet")):
        #     assert arg >= 0, name + " should be positive !"
        a, c, dt = self.get_a_c_dt(a, c, dt)
        assert dt > 0, "dt should be strictly positive"

        assert isinstance(M2, int) and isinstance(M1, int)
        assert M2 > 0 and M1 > 0
        if isinstance(D1, int):
            print("Warning: type of diffusivity is int. casting to float...")
            D1 = float(D1)
        if isinstance(D2, int):
            print("Warning: type of diffusivity is int. casting to float...")
            D2 = float(D2)
        for arg, name in zip((h2, D1, D2), ("h1", "h2", "D1", "D2")):
            assert isinstance(arg, float) or \
                isinstance(arg, np.float64) or \
                isinstance(arg, np.ndarray) and \
                arg.ndim == 1, name
            assert (np.array(arg) > 0).all(), name + " is negative or 0 !"

        assert (np.array(h1) < 0).all(), "h1 is positive or 0 ! should be <0."

        # Broadcasting / verification of types:
        D1 = np.zeros(M1 - 1) + D1
        D2 = np.zeros(M2 - 1) + D2
        h1 = np.zeros(M1 - 1) + h1
        h2 = np.zeros(M2 - 1) + h2
        f1 = np.zeros(M1) + f1
        f2 = np.zeros(M2) + f2
        # Flipping arrays to have [0] at low altitudes rather than interface
        h1f, f1f, D1f = np.flipud(h1), np.flipud(f1), np.flipud(D1)
        h1f = -h1f  # return to positive h1

        D = np.concatenate((D1f, D2))
        h = np.concatenate((h1f, h2))
        f = np.concatenate((f1f[:-1], [
            f1f[-1] * h1f[-1] / (h1f[-1] + h2[0]) + f2[0] * h2[0] /
            (h1f[-1] + h2[0])
        ], f2[1:]))
        M = M1 + M2 - 1

        Y = self.get_Y_star(M_star=M, h_star=h, D_star=D, a=a, c=c)

        rhs = np.concatenate(
            ([dirichlet], (h[1:] + h[:-1]) * (f[1:-1] + u_nm1[1:-1] / dt),
             [neumann]))

        Y[1][1:-1] += (h[1:] + h[:-1]) * np.ones(M - 2) / dt

        u_n = solve_linear(Y, rhs)

        u1_n = np.flipud(u_n[:M1])
        u2_n = u_n[M1 - 1:]

        assert u2_n.shape[0] == M2

        phi1 = D1[0] * (u1_n[1] - u1_n[0]) / h1[0]
        phi2 = D2[0] * (u2_n[1] - u2_n[0]) / h2[0]
        phi_interface = (phi2 + phi1) / 2

        return u_n, u1_n[0], phi_interface

    """
        Returns the tridiagonal matrix Y in the shape asked by solve_linear.
        (Does not actually returns a np matrix, it returns (Y_0, Y_1, Y_2).
        For details, see the documentation of @solve_linear)

        The returned matrix is of dimension MxM
        Lambda is the free parameter of Robin boundary conditions

        The following parameters can either be given as floats or np.ndarray

        h: step size (>0 in Omega_2, <0 in Omega_1) (size: M-1)
        D: diffusivity (always positive) (size: M-1)
            Note: if D is a np.ndarray, it should be given on the half-steps,
                    i.e. D[m] is D_{m+1/2}
        a: advection coefficient (should be positive) (size: M-2)
        c: reaction coefficient (should be positive) (size: M-2)

        If upper_domain is True, returns Y2; otherwise returns Y1
        (The upper domain is \Omega_2)

        It is assumed that we use an implicit Euler discretisation in time.
        However, contrary to finite_volumes.get_Y the time discretisation
        is not contained in the returned matrix.
        (The assumption is for the corrective term)
        /!\ WARNING : DUPLICATE CODE between get_Y_star and get_Y (for lisibility)

    """

    def get_Y(self,
              M,
              Lambda,
              h,
              D,
              a=None,
              c=None,
              dt=None,
              upper_domain=True):
        a, c, dt = self.get_a_c_dt(a, c, dt)
        assert isinstance(a, float) or isinstance(a, int)
        assert isinstance(c, float) or isinstance(c, int)
        if isinstance(h, int):
            print("Warning: type of step size is int. casting to float...")
            h = float(h)
        if isinstance(D, int):
            print("Warning: type of diffusivity is int. casting to float...")
            D = float(D)
        assert isinstance(M, int)
        assert M > 0
        assert isinstance(Lambda, float)
        for arg in (h, D):
            assert isinstance(arg, float) or \
                isinstance(arg, np.ndarray) and \
                arg.ndim == 1 and arg.shape[0] == M - 1
        assert isinstance(upper_domain, bool)

        # Broadcast or verification of size:
        D = np.zeros(M - 1) + D
        h = np.zeros(M - 1) + h

        if (np.array(h) <= 0).any() and upper_domain:
            print("Warning : h should not be negative in Omega_2")
        if (np.array(h) >= 0).any() and not upper_domain:
            print("Warning : h should not be positive in Omega_1")
        if (np.array(D) < 0).any():
            print("Warning : D should never be negative")
        if a < 0:
            print("Warning : a should probably not be negative")
        if c < 0:
            print("Warning : c should probably not be negative")

        h_m = h[1:]  # h_{m}
        h_mm1 = h[:-1]  # h_{m-1}
        sum_both_h = h_m + h_mm1  # (h_{m-1} + h_{m})
        D_mp1_2 = D[1:]  # D_{m+1/2}
        D_mm1_2 = D[:-1]  # D_{m-1/2}
        # MAIN DIAGONAL
        Y_1 = np.empty(M)

        Y_1[1:M - 1] = c * sum_both_h + 2 * \
            (h_mm1 * D_mp1_2 + h_m * D_mm1_2) / (h_m * h_mm1)

        Y_1[0] = Lambda - (D[0] / h[0]) * (2*h[0]+h[1]) / (h[0] + h[1])
        # Robin bd conditions at interface
        Y_1[M - 1] = 1  # Neumann bd conditions for \Omega_2, Dirichlet for \Omega_1
        if upper_domain:
            Y_1[M - 1] /= h[-1]

        # RIGHT DIAGONAL
        Y_2 = np.empty(M - 1)
        Y_2[1:] = -2 * D_mp1_2 / h_m

        Y_2[0] = (D[0] / h[0]) * (2*h[0]+h[1]) / (h[0] + h[1]) + \
                (D[1] / h[1]) * h[0] / (h[0] + h[1])
        Y_2[1:] += a

        #ULTRA-RIGHT DIAGONAL
        Y_3 = np.zeros(M - 2)

        Y_3[0] = - (D[1] / h[1]) * h[0] / (h[0] + h[1])

        # LEFT DIAGONAL
        Y_0 = np.empty(M - 1)
        Y_0[:-1] = -2 * D_mm1_2 / h_mm1
        Y_0[:-1] -= a
        if not upper_domain:
            Y_0[-1] = 0  # In \Omega_1 we have Dirichlet bd conditions
        else:
            Y_0[-1] = -1 / h[-1]  # In \Omega_2 we have Neumann bd conditions

        return (Y_0, Y_1, Y_2, Y_3)

    """
        Returns the tridiagonal matrix Y* in the shape asked by solve_linear.
        Y* is the matrix we need to inverse to solve the full domain.
        This function is useful to compute u*, solution of Y*u*=f*

        (Does not actually return a np matrix, it returns (Y_0, Y_1, Y_2).
        For details, see the documentation of @solve_linear)

        The returned matrix is of dimension M_starxM_star
        To compare with the coupled system and get:
            u*[0:M] = u1[0:M] (i.e. for all m, u*[m] = u1[m]
            u*[M-1:2M-1] = u2[0:M] (i.e. for all m, u*[M + m] = u2[m]
        We should have:
            - M_star = 2M - 1
            - D_star[0:M] = D1[0:M]
            - h_star[0:M] = h1[0:M]
            - D_star[M-1:2M-1] = D2[0:M]
            - h_star[M-1:2M-1] = h2[0:M]

        The following parameters can either be given as floats or np.ndarray

        h: step size (always positive) (size: M-1)
        D: diffusivity (always positive) (size: M-1)
            Note: if D is a np.ndarray, it should be given on the half-steps,
                    i.e. D[m] is D_{m+1/2}
        a: advection coefficient (should be positive) (size: M-2)
        c: reaction coefficient (should be positive) (size: M-2)

        /!\ WARNING : DUPLICATE CODE between get_Y_star and get_Y (for lisibility)

    """

    def get_Y_star(self, M_star, h_star, D_star, a=None, c=None):
        a, c, _ = self.get_a_c_dt(a, c)
        M = M_star
        h = h_star
        D = D_star
        assert isinstance(a, float) or isinstance(a, int)
        assert isinstance(c, float) or isinstance(c, int)
        if isinstance(h, int):
            print("Warning: type of step size is int. casting to float...")
            h = float(h)
        if isinstance(D, int):
            print("Warning: type of diffusivity is int. casting to float...")
            D = float(D)
        assert isinstance(M, int)
        assert M > 0
        for arg in (h, D):
            assert isinstance(arg, float) or \
                isinstance(arg, np.ndarray) and \
                arg.ndim == 1
        if (np.array(h) < 0).any():
            print("Warning : h should never be negative")
        if (np.array(D) < 0).any():
            print("Warning : D should never be negative")
        if a < 0:
            print("Warning : a should probably not be negative")
        if c < 0:
            print("Warning : c should probably not be negative")

        # Broadcast or verification of size:
        D = np.zeros(M - 1) + D
        h = np.zeros(M - 1) + h
        # In the notations h is negative when considering \omega_1:

        h_m = h[1:]  # h_{m}
        h_mm1 = h[:-1]  # h_{m-1}
        sum_both_h = h_m + h_mm1  # (h_{m-1} + h_{m})
        D_mp1_2 = D[1:]  # D_{m+1/2}
        D_mm1_2 = D[:-1]  # D_{m-1/2}
        # MAIN DIAGONAL
        Y_1 = np.empty(M)

        Y_1[1:M - 1] = c * sum_both_h + 2 * \
            (h_mm1 * D_mp1_2 + h_m * D_mm1_2) / (h_m * h_mm1)
        Y_1[0] = 1  # Dirichlet
        Y_1[M - 1] = 1 / h[-1]  # Neumann

        # RIGHT DIAGONAL
        Y_2 = np.empty(M - 1)
        Y_2[1:] = -2 * D_mp1_2 / h_m + a
        Y_2[0] = 0

        # LEFT DIAGONAL
        Y_0 = np.empty(M - 1)
        Y_0[:-1] = -2 * D_mm1_2 / h_mm1 - a
        Y_0[-1] = -1 / h[-1]  # Neumann bd conditions on top

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
        a, c, dt, bd_cond, Lambda = float(a), \
            float(c), float(dt), float(bd_cond), float(Lambda)

        # Broadcasting / verification of type:
        D = np.zeros(M - 1) + D
        h = np.zeros(M - 1) + h

        assert upper_domain is True or upper_domain is False

        Y = self.get_Y(M=M,
                       h=h,
                       D=D,
                       a=a,
                       c=c,
                       dt=dt,
                       Lambda=Lambda,
                       upper_domain=upper_domain)
        Y[1][1:-1] += (np.ones(M - 2) / dt) * (h[1:] + h[:-1])
        return Y


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
            h = -self.SIZE_DOMAIN_1 / (M - 1)
        elif j == 2: 
            if M is None:
                M = self.M2_DEFAULT
            if D is None:
                D = self.D2_DEFAULT
            h = self.SIZE_DOMAIN_2 / (M - 1)

        Y_0 = -D / (h * h) - .5 * a / h
        Y_1 = 2 * D / (h * h) + c
        Y_2 = -D / (h * h) + .5 * a / h

        lambda_moins = (Y_1 + s - np.sqrt((Y_1 + s)**2 - 4 * Y_0 * Y_2)) \
                                / (-2 * Y_2)
        lambda_plus = (Y_1 + s + np.sqrt((Y_1 + s)**2 - 4 * Y_0 * Y_2)) \
                                / (-2 * Y_2)

        # The computation is then different because the boundary condition is different
        if j == 1:
            eta1_dir = 1 + (lambda_moins / lambda_plus) ** M
            eta1_neu = D/h * ((lambda_moins - 1) * (3/2 - lambda_moins/2) \
                    + (lambda_plus - 1) * (3/2 - lambda_plus/2) \
                    * (lambda_moins / lambda_plus)**M)
            return eta1_dir, eta1_neu
        elif j == 2:
            eta2_dir = 1 + (lambda_moins-1) / (lambda_plus - 1) *(lambda_moins / lambda_plus) ** (M - 1)
            eta2_neu = D/h * ((lambda_moins - 1) * (3/2 - lambda_moins/2) \
                    + (lambda_plus - 1)  * (3/2 - lambda_plus/2) \
                    * (lambda_moins-1) / (lambda_plus - 1) *(lambda_moins / lambda_plus) ** (M - 1))
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

        h1 = -self.SIZE_DOMAIN_1 / (M1 - 1)
        h2 = self.SIZE_DOMAIN_2 / (M2 - 1)

        Y1_0 = -D1 / (h1 * h1) - .5 * a / h1
        Y1_1 = 2 * D1 / (h1 * h1) + c
        Y1_2 = -D1 / (h1 * h1) + .5 * a / h1

        Y2_0 = -D2 / (h2 * h2) - .5 * a / h2
        Y2_1 = 2 * D2 / (h2 * h2) + c
        Y2_2 = -D2 / (h2 * h2) + .5 * a / h2
        lambda2_moins = (Y2_1 + s - np.sqrt((Y2_1 + s)**2 - 4 * Y2_0 * Y2_2)) \
                                / (-2 * Y2_2)
        lambda1_moins = (Y1_1 + s - np.sqrt((Y1_1 + s)**2 - 4 * Y1_0 * Y1_2)) \
                                / (-2 * Y1_2)

        lambda1 = lambda1_moins
        lambda2 = lambda2_moins
        teta1_0 = -D1/(2*h1) * lambda1**2 + \
                    lambda1 * 2*D1/h1 - 3*D1/(2*h1)
        teta2_0 = -D2/(2*h2) * lambda2**2 + \
                    lambda2 * 2*D2/h2 - 3*D2/(2*h2)
        teta1_0 *= -1
        teta2_0 *= -1
        rho_numerator = (Lambda_2 - teta1_0) * (Lambda_1 - teta2_0)
        rho_denominator = (Lambda_2 - teta2_0) * (Lambda_1 - teta1_0)
        if verbose:
            print("only with Lambda2:",
                  (Lambda_2 - teta1_0) / (Lambda_2 - teta2_0))
            print("only with Lambda1:",
                  (Lambda_1 - teta2_0) / (Lambda_1 - teta1_0))

        return np.abs(rho_numerator / rho_denominator)

    """
        Simple function to return h in each subdomains,
        in the framework of finite differences.
        returns uniform spaces between points (h1, h2).
        To recover xi, use:
        xi = np.cumsum(np.concatenate(([0], hi)))
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
        x1 = -np.linspace(0, size_domain_1, M1)**1
        x2 = np.linspace(0, size_domain_2, M2)**1
        return np.diff(x1), np.diff(x2)

    """
        Simple function to return D in each subdomains,
        in the framework of finite differences.
        provide continuous functions accepting ndarray
        for D1 and D2, and returns the right coefficients.
        By default, D1 and D2 are constant.
    """

    def get_D(self, h1, h2, function_D1=None, function_D2=None):
        if function_D1 is None:
            def function_D1(x): return self.D1_DEFAULT + np.zeros_like(x)
        if function_D2 is None:
            def function_D2(x): return self.D2_DEFAULT + np.zeros_like(x)
        x1 = np.cumsum(np.concatenate(([0], h1)))
        x2 = np.cumsum(np.concatenate(([0], h2)))
        # coordinates at half-points:
        x1_1_2 = x1[:-1] + h1 / 2
        x2_1_2 = x2[:-1] + h2 / 2
        D1 = function_D1(x1_1_2)
        D2 = function_D2(x2_1_2)
        return D1, D2

    def name(self):
        return "Différences finies : flux extrapolé"

    def repr(self):
        return "finite differences, no corrective term"

if __name__ == "__main__":
    from tests import test_finite_differences_no_corrective_term
    test_finite_differences_no_corrective_term.launch_all_tests()
