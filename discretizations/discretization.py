import numpy as np
"""
    This module describe an abstract class: when creating a new
    discretization, one should inherit from this class.
    Provides prototypes and verification of arguments.
"""


class Discretization:

    def integrate_one_step(self, f, bd_cond, u_nm1,
                           u_interface, phi_interface, upper_domain, Y):
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

        raise NotImplementedError


    def s_time_modif(self, w, order):
        """ By default, we are in the euler implicit time scheme"""
        dt = self.DT
        s = w * 1j
        if order > 0:
            s += dt/2 * w**2
        if order > 1:
            s -= dt**2/6 * 1j * w**3
        if order > 2:
            s -= dt**3 / 24 * w**4
        if order > 3:
            s += dt**4/(120) * 1j * w**5
        return s

    def get_h(self):
        """
            Simple function to return h in each subdomains,
            in the framework of finite differences.
            returns uniform spaces between points (h1, h2).
            Maybe we should add a transformation function?
            or add optional arguments x1, x2 to allow
            the user to use non-uniform discretization
        """

        raise NotImplementedError

    def get_D(self):
        """
            Simple function to return D in each subdomains,
            in the framework of finite differences.
            provide continuous functions accepting ndarray
            for D1 and D2, and returns the right coefficients.
        """

        raise NotImplementedError

    def precompute_Y(self, f, bd_cond, upper_domain):
        """
            Precompute Y for integrate_one_step. useful when integrating over a long time
            The arguments are exactly the arguments of @integrate_one_step,
            except for u_nm1, and interface conditions.
            f is kept as an argument but should not be used.
            It is not mandatory to implement this method.
        """

        return None

    def name(self):
        """
            Returns the name of the discretization, no caps.
        """

        return "unknown discretization"

    def analytic_robin_robin(self, s=None):
        """
            When D and h are constant, it is possible to find the convergence
            rate in frequency domain. analytic_robin_robin computes this convergence rate.
            s is 1/dt when considering the local-in-time case, otherwise it
            should be iw (with w the desired frequency)
            In the discrete time setting, the Z transform gives s = 1. / dt * (z - 1) / z
            for implicit euler discretisation.

            This method should *not* be overriden, the
            particularity of the discretization is specified
            through the method eta_dirneu
        """
        eta1_dir, eta1_neu = self.eta_dirneu(1, s)
        eta2_dir, eta2_neu = self.eta_dirneu(2, s)
        rho_numerator = (self.LAMBDA_2*eta1_dir + eta1_neu) * (self.LAMBDA_1*eta2_dir + eta2_neu)
        rho_denominator = (self.LAMBDA_2*eta2_dir + eta2_neu) * (self.LAMBDA_1*eta1_dir + eta1_neu)
        return np.abs(rho_numerator / rho_denominator)

    def eta_dirneu(self, j, s=None):
        """
            Gives the \\eta used to compute the analytic rate (see analytic_robin_robin)
            can be:
                -eta(1, ..);      <- for domain \\Omega_1
                -eta(2, ..);      <- for domain \\Omega_2
            returns tuple (etaj_dir, etaj_neu).
        """
        raise NotImplementedError

    def sigma_modified(self, w, order_time, order_equations):
        # This code should not run and is here as an example
        raise NotImplementedError
        s = self.s_time_modif(w, self.DT, order_time) + self.C
        sig1 = np.sqrt(s/self.D1)
        sig2 = -np.sqrt(s/self.D2)
        return sig1, sig2

    def eta_dirneu_modif(self, j, sigj, order_operators, *kwargs, **dicargs):
        # This code should not run and is here as an example
        raise NotImplementedError
        if j==1:
            eta_dir_modif = 1
            eta_neu_modif = sigj * self.D1
            return eta_dir_modif, eta_neu_modif
        else:
            eta_dir_modif = 1
            eta_neu_modif = sigj * self.D2
            return eta_dir_modif, eta_neu_modif

    def analytic_robin_robin_modified(self, w, order_time=float('inf'), order_operators=float('inf'), order_equations=float('inf')):
        """
            When D and h are constant, it is possible to find the convergence
            rate in frequency domain. analytic_robin_robin computes this convergence rate.
            s is 1/dt when considering the local-in-time case, otherwise it
            should be iw (with w the desired frequency)
            In the discrete time setting, the Z transform gives s = 1. / dt * (z - 1) / z
            for implicit euler discretisation.

            This method should *not* be overriden, the
            particularity of the discretization is specified
            through the method eta_dirneu
        """
        sig1, sig2 = self.sigma_modified(w, order_time, order_equations)
        eta1_dir, eta1_neu = self.eta_dirneu_modif(j=1, sigj=sig1, order_operators=order_operators, w=w)
        eta2_dir, eta2_neu = self.eta_dirneu_modif(j=2, sigj=sig2, order_operators=order_operators, w=w)
        rho_numerator = (self.LAMBDA_2*eta1_dir + eta1_neu) * (self.LAMBDA_1*eta2_dir + eta2_neu)
        rho_denominator = (self.LAMBDA_2*eta2_dir + eta2_neu) * (self.LAMBDA_1*eta1_dir + eta1_neu)
        return np.abs(rho_numerator / rho_denominator)

    def M_h_D_Lambda(self, upper_domain):
        """
            returns M_j, h_j, D_j, Lambda_j with j = (2 if upper_domain else 1)
        """
        h1, h2 = self.get_h()
        if upper_domain:
            return self.M2, h2, self.D2, self.LAMBDA_2
        else:
            return self.M1, h1, self.D1, self.LAMBDA_1

    def get_a_c_dt(self):
        """
            Returns default values of a, c, dt or parameters if given.
        """
        return self.A, self.C, self.DT

    def clone(self):
        ret = self.__class__()
        ret.__dict__ = self.__dict__.copy()
        return ret


    """
        __eq__ and __hash__ are implemented, so that a discretization
        can be stored as key in a dict
        (it is useful for memoisation)
    """

    def __eq__(self, other):
        return self.__dict__ == other.__dict__ and self.repr() == other.repr()

    def __hash__(self):
        return hash(repr(sorted(self.__dict__.items())) + self.repr())

    def __repr__(self):
        return repr(sorted(self.__dict__.items())) + self.repr()
