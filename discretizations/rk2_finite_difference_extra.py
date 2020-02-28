"""
    Finite differences for diffusion advection reaction equation.
    The functions to use are integrate_one_step and integrate_one_step_star.
    Theses function make a single step in time. It is not efficient
    (need to compute the matrix each time) but it is simple.
"""

import numpy as np
from discretizations.discretization import Discretization
from utils_linalg import solve_linear_with_ultra_right, solve_linear


class Rk2FiniteDifferencesExtra(Discretization):
    """
        give default values of all variables.
    """

    def __init__(self,
                 A=None,
                 C=None,
                 D1=None,
                 D2=None,
                 M1=None,
                 M2=None,
                 SIZE_DOMAIN_1=None,
                 SIZE_DOMAIN_2=None,
                 LAMBDA_1=None,
                 LAMBDA_2=None,
                 DT=None):
        self.A, self.C, self.D1, self.D2, \
            self.M1, self.M2, self.SIZE_DOMAIN_1, \
            self.SIZE_DOMAIN_2, self.LAMBDA_1, \
            self.LAMBDA_2, self.DT = A, \
            C, D1, D2, \
            M1, M2, SIZE_DOMAIN_1, SIZE_DOMAIN_2, \
            LAMBDA_1, LAMBDA_2, DT

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
                           f,
                           f_nm1_2,
                           f_nm1,
                           bd_cond,
                           bd_cond_nm1_2,
                           bd_cond_nm1,
                           u_nm1,
                           u_interface,
                           phi_interface,
                           u_nm1_2_interface,
                           phi_nm1_2_interface,
                           u_nm1_interface,
                           phi_nm1_interface,
                           upper_domain=True,
                           Y=None,
                           **kwargs):
        M, h, D, Lambda = self.M_h_D_Lambda(upper_domain)
        a, c, dt = self.get_a_c_dt()
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

        #h and D need *not* to be constant :
        #assert np.linalg.norm(h-h[0]) < 1e-11
        #assert np.linalg.norm(D-D[0]) < 1e-11

        robin_nm1_2 = Lambda * u_nm1_2_interface + phi_nm1_2_interface
        robin_np1 = Lambda * u_interface + phi_interface

        def compute_k(f, u):
            return f[1:-1] + np.diff(D*np.diff(u)/h)/((h[1:] + h[:-1])/2) - a*(u[2:] - u[:-2])/(h[1:] + h[:-1]) - c*u[1:-1]

        def compute_u_ni(step, bd_cond, robin_cond):
            u_ni = np.copy(u_nm1)
            u_ni[1:-1] += step
            # Robin : robin_cond = Lambda u + (2h0+h1)/(h0+h1)phi_1_2 -  h0/(h0+h1)phi_3_2
            phi_3_2 = D[1] / h[1] * (u_ni[2] - u_ni[1])
            u_ni[0] = (robin_cond + h[0]/(h[0]+h[1])*phi_3_2
                    - (2*h[0]+h[1])/(h[0]+h[1])*D[0]/h[0]*u_ni[1] ) / \
                (Lambda - (2*h[0]+h[1])/(h[0]+h[1])*D[0]/h[0])
            if upper_domain: # Neumann :
                u_ni[-1] = u_ni[-2] + h[-1]*bd_cond
            else: # Dirichlet :
                u_ni[-1] = bd_cond
            return u_ni

        k1 = compute_k(f_nm1, u_nm1)

        u_n1 = compute_u_ni(dt/2 * k1, bd_cond_nm1_2, robin_nm1_2)
        #u_n1 is our approximation of u at n+1/2
        k2 = compute_k(f_nm1_2, u_n1)
        u_n = compute_u_ni(dt * k2, bd_cond, robin_np1)

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

    def integrate_one_step_star(self, neumann, dirichlet, u_nm1):
        M1, h1, D1, _ = self.M_h_D_Lambda(upper_domain=False)
        M2, h2, D2, _ = self.M_h_D_Lambda(upper_domain=True)

        a, c, dt = self.get_a_c_dt()
        a, c, dt, neumann, dirichlet = float(a), float(c), float(dt), \
            float(neumann), float(dirichlet)
        # Theses assertions cannot be used because of the unit tests:
        # for arg, name in zip((a, c, neumann, dirichlet),
        #         ("a", "c", "neumann", "dirichlet")):
        #     assert arg >= 0, name + " should be positive !"
        a, c, dt = self.get_a_c_dt()
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

        Y = self.get_Y_star(M_star=M, h_star=h, D_star=D)

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
              upper_domain=True):
        M, h, D, Lambda = self.M_h_D_Lambda(upper_domain)
        a, c, dt = self.get_a_c_dt()
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

    def get_Y_star(self, M_star, h_star, D_star):
        a, c, _ = self.get_a_c_dt()
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

    def precompute_Y(self, f, bd_cond, upper_domain=True):
        M, h, D, Lambda = self.M_h_D_Lambda(upper_domain)
        a, c, dt = self.get_a_c_dt()
        a, c, dt, bd_cond, Lambda = float(a), \
            float(c), float(dt), float(bd_cond), float(Lambda)

        # Broadcasting / verification of type:
        D = np.zeros(M - 1) + D
        h = np.zeros(M - 1) + h

        assert upper_domain is True or upper_domain is False

        Y = self.get_Y(upper_domain=upper_domain)
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

        a, c, dt = self.get_a_c_dt()
        if s is None:
            s = 1 / dt

        if j == 1:
            if M is None:
                M = self.M1
            if D is None:
                D = self.D1
            h = -self.SIZE_DOMAIN_1 / (M - 1)
        elif j == 2: 
            if M is None:
                M = self.M2
            if D is None:
                D = self.D2
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

            sol_calc = (s*h/2 + np.sqrt((s*h/2)**2 + s*D))*(1 - s*h*h/(4*D) - h/(2*D)*np.sqrt((s*h/2)**2 + s*D))
            sol_calc2 = (-s*h*h/(2*D) + 1)*np.sqrt((s*h/2)**2 + s*D) - s**2*h**3/(4*D)
            if abs(sol_calc-D/h * ((lambda_moins - 1) * (3/2 - lambda_moins/2))) > 1e-6:
                print("Error (rk2) extra 1: computation is not right")
                print(abs(sol_calc-eta1_neu), eta1_neu)
                raise
            return eta1_dir, eta1_neu
        elif j == 2:
            eta2_dir = 1 + (lambda_moins-1) / (lambda_plus - 1) *(lambda_moins / lambda_plus) ** (M - 1)
            eta2_neu = D/h * ((lambda_moins - 1) * (3/2 - lambda_moins/2) \
                    + (lambda_plus - 1)  * (3/2 - lambda_plus/2) \
                    * (lambda_moins-1) / (lambda_plus - 1) *(lambda_moins / lambda_plus) ** (M - 1))
            sol_calc = (s*h/2 - np.sqrt((s*h/2)**2 + s*D))*(1 - s*h*h/(4*D) + h/(2*D)*np.sqrt((s*h/2)**2 + s*D))
            sol_calc2 = (s*h*h/(2*D) - 1)*np.sqrt((s*h/2)**2 + s*D) - s**2*h**3/(4*D)
            if abs(sol_calc-D/h * ((lambda_moins - 1) * (3/2 - lambda_moins/2))) > 1e-6:
                print("Error (rk2) extra 2: computation is not right")
                print(abs(sol_calc-eta2_neu), eta2_neu)
                raise

            return eta2_dir, eta2_neu


    def sigma_modified(self, w, order_time, order_equations):
        h1, h2 = self.get_h()
        h1, h2 = h1[0], h2[0]
        D1, D2 = self.D1, self.D2
        dt = self.DT

        s = self.s_time_modif(w, order_time) + self.C
        s1 = s
        if order_equations > 0:
            s1 += w**2 * (h1**2/(12*D1))
        if order_equations > 1:
            s1 -= 1j * w**3 * (- h1**4/(12*30*D1**2))

        s2 = s
        if order_equations > 0:
            s2 += w**2 * (h2**2/(12*D2))
        if order_equations > 1:
            s2 -= 1j * w**3 * (- h2**4/(12*30*D2**2))

        sig1 = np.sqrt(s1/self.D1)
        sig2 = -np.sqrt(s2/self.D2)
        return sig1, sig2

    def eta_dirneu_modif(self, j, sigj, order_operators, w, *kwargs, **dicargs):
        # This code should not run and is here as an example
        h1, h2 = self.get_h()
        h1, h2 = h1[0], h2[0]
        D1, D2 = self.D1, self.D2
        if j==1:
            sig1=sigj
            eta_dir_modif = 1
            eta_neu_modif = D1*sig1
            if order_operators > 0:
                eta_neu_modif -= D1*h1**2*sig1**3/3
            if order_operators > 1:
                eta_neu_modif -= D1*h1**3*sig1**4/4
            if order_operators > 2:
                eta_neu_modif -= D1*h1**4*sig1**5*7/60
            return eta_dir_modif, eta_neu_modif
        else:
            sig2=sigj
            eta_dir_modif = 1
            eta_neu_modif = D2*sig2
            if order_operators > 0:
                eta_neu_modif -= D2*h2**2*sig2**3/3
            if order_operators > 1:
                eta_neu_modif -= D2*h2**3*sig2**4/4
            if order_operators > 2:
                eta_neu_modif -= D2*h2**4*sig2**5*7/60
            return eta_dir_modif, eta_neu_modif


    def s_time_modif(self, w, order):
        s = w * 1j - w**3 * self.DT**2/6 * 1j
        return s

    """
        Simple function to return h in each subdomains,
        in the framework of finite differences.
        returns uniform spaces between points (h1, h2).
        To recover xi, use:
        xi = np.cumsum(np.concatenate(([0], hi)))
    """

    def get_h(self):
        size_domain_1 = self.SIZE_DOMAIN_1
        size_domain_2 = self.SIZE_DOMAIN_2
        M1 = self.M1
        M2 = self.M2
        x1 = -np.linspace(0, size_domain_1, M1)
        x2 = np.linspace(0, size_domain_2, M2)
        return np.diff(x1), np.diff(x2)

    """
        Simple function to return D in each subdomains,
        in the framework of finite differences.
        provide continuous functions accepting ndarray
        for D1 and D2, and returns the right coefficients.
        By default, D1 and D2 are constant.
    """

    def get_D(self, h1=None, h2=None, function_D1=None, function_D2=None):
        if h1 is None or h2 is None:
            h1, h2 = self.get_h()
        if function_D1 is None:
            def function_D1(x): return self.D1 + np.zeros_like(x)
        if function_D2 is None:
            def function_D2(x): return self.D2 + np.zeros_like(x)
        x1 = np.cumsum(np.concatenate(([0], h1)))
        x2 = np.cumsum(np.concatenate(([0], h2)))
        # coordinates at half-points:
        x1_1_2 = x1[:-1] + h1 / 2
        x2_1_2 = x2[:-1] + h2 / 2
        D1 = function_D1(x1_1_2)
        D2 = function_D2(x2_1_2)
        return D1, D2

    def name(self):
        return "Différences finies : flux extrapolé (rk2)"

    def repr(self):
        return "finite differences, extrapolated flux (rk2)"

if __name__ == "__main__":
    from tests import test_finite_differences_no_corrective_term
    test_finite_differences_no_corrective_term.launch_all_tests()
