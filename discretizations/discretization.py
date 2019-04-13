"""
    This module describe an abstract class: when creating a new
    discretization, one should inherit from this class.
    Provides prototypes and verification of arguments.
"""


class Discretization:
    """
        When D and h are constant, it is possible to find the convergence
        rate in frequency domain. analytic_robin_robin computes this convergence rate.
        s is 1/dt when considering the local-in-time case, otherwise it
        should be iw (with w the desired frequency)
        In the discrete time setting, the Z transform gives s = 1. / dt * (z - 1) / z
        for implicit euler discretisation.
    """

    def analytic_robin_robin(self, s, Lambda_1, Lambda_2, a, c, dt, M1, M2, D1,
                             D2, verbose):
        raise NotImplementedError

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

    def integrate_one_step(self, M, h, D, a, c, dt, f, bd_cond, Lambda, u_nm1,
                           u_interface, phi_interface, upper_domain, Y):
        raise NotImplementedError

    """
        Simple function to return h in each subdomains,
        in the framework of finite differences.
        returns uniform spaces between points (h1, h2).
        Maybe we should add a transformation function?
        or add optional arguments x1, x2 to allow
        the user to use non-uniform discretization
    """

    def get_h(self, size_domain_1, size_domain_2, M1, M2):
        raise NotImplementedError

    """
        Simple function to return D in each subdomains,
        in the framework of finite differences.
        provide continuous functions accepting ndarray
        for D1 and D2, and returns the right coefficients.
    """

    def get_D(self, h1, h2, function_D1, function_D2):
        raise NotImplementedError

    """
        Precompute Y for integrate_one_step. useful when integrating over a long time
        The arguments are exactly the arguments of @integrate_one_step,
        except for u_nm1, and interface conditions.
        f is kept as an argument but should not be used.
        It is not mandatory to implement this method.
    """

    def precompute_Y(self, M, h, D, a, c, dt, f, bd_cond, Lambda,
                     upper_domain):
        return None

    """
        Returns the name of the discretization, no caps.
    """

    def name(self):
        return "unknown discretization"

    """
        __eq__ and __hash__ are implemented, so that a discretization
        can be stored as key in a dict
        (it is useful for memoisation)
    """

    def __eq__(self, other):
        return self.__dict__ == other.__dict__ and self.name() == other.name()

    def __hash__(self):
        return hash(repr(sorted(self.__dict__.items())) + self.name())
