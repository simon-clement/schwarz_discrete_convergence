import numpy as np
"""
    This module describe an abstract class: when creating a new
    discretization, one should inherit from this class.
    Provides prototypes and verification of arguments.
"""


class Discretization:

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

    def analytic_robin_robin_modified(self, w, order_time=float('inf'), order_operators=float('inf'), order_equations=float('inf')):
        """
            When D and h are constant, it is possible to find the convergence
            rate in frequency domain. analytic_robin_robin computes this convergence rate.

            This method should *not* be overriden, the
            particularity of the discretization is specified
            through the method eta_dirneu

            An infinite order means the discrete analysis is done
            An order=0 means the continuous analysis is done.
            Between them, the modified analysis is used.
        """
        assert "LAMBDA_2" in self.__dict__
        assert "LAMBDA_1" in self.__dict__
        ########################################################
        # Computing s_d, s_c or s_m depending on order_time:
        ########################################################
        if order_time == float('inf'):
            s = self.s_time_discrete(w) + self.C
        else:
            s = self.s_time_modif(w, order_time) + self.C

        ###################################################################
        # Computing \\sigma_j or \\lambda_j depending on order_equations: #
        ###################################################################
        # The convention here is: \\lambda_- is the main root,
        # and \\lambda_+ is the secondary root.
        if order_equations == float('inf'):
            lam1, lam2, lam1_p, lam2_p = self.lambda_1_2_pm(s)
            sig1, sig2, sig1_p, sig2_p = np.log(lam1), np.log(lam2), np.log(lam1_p), np.log(lam2_p)
        else:
            sig1, sig2 = self.sigma_modified(s, w, order_equations)
            sig1_p = -sig1
            sig2_p = -sig2

        #########################################################
        # Computing \\eta_{j, op} depending on order_operators: #
        #########################################################

        if order_operators == float('inf'):
            eta1_dir, eta1_neu = self.eta_dirneu(j=1, lam_m=np.exp(sig1), lam_p=np.exp(sig1_p), s=s)
            eta2_dir, eta2_neu = self.eta_dirneu(j=2, lam_m=np.exp(sig2), lam_p=np.exp(sig2_p), s=s)
        else:
            eta1_dir, eta1_neu = self.eta_dirneu_modif(j=1, sigj=sig1, order_operators=order_operators, w=w)
            eta2_dir, eta2_neu = self.eta_dirneu_modif(j=2, sigj=sig2, order_operators=order_operators, w=w)

        #########################################################
        # Computing \\rho with the results:
        #########################################################
        rho_numerator = (self.LAMBDA_2*eta1_dir + eta1_neu) * (self.LAMBDA_1*eta2_dir + eta2_neu)
        rho_denominator = (self.LAMBDA_2*eta2_dir + eta2_neu) * (self.LAMBDA_1*eta1_dir + eta1_neu)
        return np.abs(rho_numerator / rho_denominator)

    def eta_dirneu(self, j, lam_m, lam_p, s=None):
        """
            lam_m, lam_p should be computed by @self.lambda1_2_pm or np.exp(@self.sigma_modified).
            lam_m is the *main* root, where lam_p is the secondary.
            Gives the \\eta used to compute the analytic rate (see analytic_robin_robin)
            can be:
                -eta(1, ..);      <- for domain \\Omega_1
                -eta(2, ..);      <- for domain \\Omega_2
            returns tuple (etaj_dir, etaj_neu).
        """
        raise NotImplementedError

    def s_time_discrete(self, w):
        """ By default, we are in the Backward Euler time scheme"""
        assert w is not None # verifying the setting is not local in time
        assert "DT" in self.__dict__ # verifying we are in a valid time discretization
        z = np.exp(w * 1j * self.DT)
        return 1. / self.DT * (z - 1) / z

    def eta_dirneu_modif(self, j, sigj, order_operators, *kwargs, **dicargs):
        """ Returns the modified eta variable of the time scheme, with specified order"""
        raise NotImplementedError
        # if j==1:
        #     eta_dir_modif = 1
        #     eta_neu_modif = sigj * self.D1
        #     return eta_dir_modif, eta_neu_modif
        # else:
        #     eta_dir_modif = 1
        #     eta_neu_modif = sigj * self.D2
        #     return eta_dir_modif, eta_neu_modif

    def s_time_modif(self, w, order):
        """ Returns the modified s variable of the time scheme, with specified order"""
        assert order < float('inf')
        """ By default, we are in the Backward Euler time scheme"""
        assert "DT" in self.__dict__ # verifying we are in a valid time discretization
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

    def get_D(self, **kwargs):
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

    def lambda_1_2_pm(self, s):
        # The convention here is: \\lambda_- is the main root,
        # and \\lambda_+ is the secondary root.
        """
            Gives the \\lambda_\\pm:
            returns \\lambda_{-, j=1}, \\lambda_{-, j=2}, \\lambda_{+, j=1}, \\lambda_{+, j=2}.
        """
        raise NotImplementedError

    def sigma_modified(self, s, w, order_equations):
        # The convention here is: \\sigma_- is the main root,
        # and \\sigma_+ is the secondary root.
        if order_equations == 0: # no need for space discretization in this case
            sig1 = np.sqrt(s/self.D1)
            sig2 = -np.sqrt(s/self.D2)
            return sig1, sig2
        else:
            raise NotImplementedError

    def M_h_D_Lambda(self, upper_domain):
        """
            returns M_j, h_j, D_j, Lambda_j with j = (2 if upper_domain else 1)
        """
        assert "M2" in self.__dict__
        assert "M1" in self.__dict__
        assert "D2" in self.__dict__
        assert "D1" in self.__dict__
        assert "LAMBDA_2" in self.__dict__
        assert "LAMBDA_1" in self.__dict__
        h1, h2 = self.get_h()
        if upper_domain:
            return self.M2, h2, self.D2, self.LAMBDA_2
        else:
            return self.M1, h1, self.D1, self.LAMBDA_1

    def get_a_c_dt(self):
        """
            Returns default values of a, c, dt or parameters if given.
        """
        assert "A" in self.__dict__
        assert "C" in self.__dict__
        assert "DT" in self.__dict__
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
