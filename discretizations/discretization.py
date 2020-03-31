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

    
    #####################
    # INTEGRATION FUNCTIONS:
    #####################
    def A_interior(self, upper_domain):
        """
            gives A, such as inside the domain, A \\partial_t u = Bu
            For finite differences, A is identity.
        """
        raise NotImplementedError

    def B_interior(self, upper_domain):
        """
            gives f, such as inside the domain, A \\partial_t u = Bu
            This function supposes there is no forcing. this forcing should be added
            in the time integration.
            For finite differences, A is identity.
        """
        raise NotImplementedError

    def add_boundaries(self, to_inverse, rhs, interface_cond, bd_cond, upper_domain,
                    coef_implicit, coef_explicit, dt, f, sol_for_explicit, sol_unm1, additional):
        """
            Take the matrix to_inverse and vector rhs:
            concatenate bd and interface conditions with rhs, and add the
            lines to "to_inverse" to be able to compute the boundaries of solution.
            The discretization must implement the functions "@self.discretization_bd_cond"
            and "@self.discretization_interface"
            Returns to_inverse and rhs: warning, the input must be tridiagonal but the output
            may not be.
        """
        assert len(to_inverse) == 3
        assert len(rhs.shape) == 1
        new_rhs = np.concatenate(([interface_cond], rhs, [bd_cond]))
        list_bd_cond = self.discretization_bd_cond(upper_domain=upper_domain)
        list_interface = self.discretization_interface(upper_domain=upper_domain)
        if list_bd_cond is None:
            list_bd_cond, new_rhs[-1] = self.hardcoded_bd_cond(upper_domain=upper_domain,
                    bd_cond=bd_cond, dt=dt, f=f, sol_for_explicit=sol_for_explicit,
                    sol_unm1=sol_unm1, additional=additional,
                    coef_explicit=coef_explicit, coef_implicit=coef_implicit)
        if list_interface is None:
            list_interface, new_rhs[0] = self.hardcoded_interface(upper_domain=upper_domain,
                    robin_cond=interface_cond, dt=dt, f=f, sol_for_explicit=sol_for_explicit,
                    sol_unm1=sol_unm1,
                    additional=additional, coef_explicit=coef_explicit, coef_implicit=coef_implicit)
        # let's begin with the boundary condition:
        new_Y = []
        assert len(list_bd_cond) == 1 or len(list_bd_cond) == 2
        assert len(list_interface) >= 1
        Y_0, Y_1, Y_2 = to_inverse
        Y_1 = np.concatenate(([list_interface[0]], Y_1, [list_bd_cond[0]]))
        Y_0 = np.concatenate((Y_0,
            [0 if len(list_bd_cond) == 1 else list_bd_cond[1]]))

        Y_2 = np.concatenate(([0 if len(list_interface) == 1 else list_interface[1]] , Y_2))

        ret = [Y_0, Y_1, Y_2]
        for additional_coeff_interface in list_interface[2:]:
            ret += [np.zeros(ret[-1].shape[0] - 1)] # new diagonal above the last one
            ret[-1][0] = additional_coeff_interface
        return tuple(ret), new_rhs

    def discretization_bd_cond(self, upper_domain):
        """
        Gives the coefficients in front of u to compute the bd condition,
        either at the top of the atmosphere (Dirichlet)
        or at the bottom of the Ocean (Neumann).
        Returns a list, for which the index 0 is the value
        to set at the diagonal, and the index 1 (if it exists) is
        the subdiagonal.
        """
        raise NotImplementedError

    def discretization_interface(self, upper_domain):
        """
        Gives the coefficients in front of u to compute the bd condition,
        at the interface between the ocean and atmosphere.
        Returns a list, representing the vector which will make a
        scalar product with u. The vector should be of size 1, 2 or 3.
        The other numbers in the vector are assumed to be zeros.
        """
        raise NotImplementedError

    def update_additional(self, result, additional, dt, upper_domain, f, coef_reaction_implicit, reaction_explicit):
        # reaction_explicit is either 0 or a multiple of additional ! the idea is that
        # coef_reaction_implicit is 0 or 1 and reaction_explicit = coef_reaction_explicit * additional
        pass

    def create_additional(self, upper_domain):
        return None

    def projection_result(self, result, upper_domain, partial_t_result0, f, additional):
        """
            given the result of the inversion, returns (u_np1, u_interface, phi_interface)
        """
        raise NotImplementedError

    def hardcoded_bd_cond(self, upper_domain, bd_cond, coef_implicit, coef_explicit, dt, f, sol_for_explicit, sol_unm1, additional):
        """
            For schemes that use corrective terms or any mechanism of time derivative inside bd cond,
            this method allows the time scheme to correct the boundary condition.
            Called when discretization_bd_cond returns None. if it is never the case, no need
            to implement this method.
        """
        raise NotImplementedError

    def hardcoded_interface(self, upper_domain, robin_cond, coef_implicit, coef_explicit, dt, f, sol_for_explicit, sol_unm1, additional):
        """
            For schemes that use corrective terms or any mechanism of time derivative
            inside interface condition,
            this method allows the time scheme to correct the boundary condition.
            Called when discretization_interface returns None. if it is never the case, no need
            to implement this method.
        """
        raise NotImplementedError

    def size_f(self, upper_domain):
        M, h, D, Lambda = self.M_h_D_Lambda(upper_domain=upper_domain)
        return M

    def size_prognostic(self, upper_domain):
        M, h, D, Lambda = self.M_h_D_Lambda(upper_domain=upper_domain)
        return M

    #####################
    # ANALYSIS FUNCTIONS:
    #####################

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
        raise NotImplementedError

    def eta_dirneu_modif(self, j, sigj, order_operators, *kwargs, **dicargs):
        """ Returns the modified eta variable of the time scheme, with specified order"""
        raise NotImplementedError

    def s_time_modif(self, w, order):
        """ Returns the modified s variable of the time scheme, with specified order"""
        raise NotImplementedError

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
        D1, D2 = self.get_D(h1=h1, h2=h2)
        if upper_domain:
            return self.M2, h2, D2, self.LAMBDA_2
        else:
            return self.M1, h1, D1, self.LAMBDA_1

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
