import numpy as np
"""
    This module describe an abstract class: when creating a new
    discretization, one should inherit from this class.
    Provides prototypes and verification of arguments.
    Discretizations in space are in the space/ folder,
    Discretizations in time are in the time/ folder,
    Space-time discretizations can also be created from this class,
    by implementing integrate_one_step.
    The idea is that there are two kinds of functions:
    functions that can be implemented by a space discretization,
    and functions that can be implemented by a time discretization.
    The constructor is called by the space discretization (arbitrary choice)

    The idea of the separation if the following:
    The time discretization just implements integrate_one_step
    and uses all the functions provided by the space discretization.

    The space discretization is represented by matrices A, B such as:
    A @ \\partial_t u[1:-1] = B @ u[1:-1] + f[1:-1]
    For example, A is Identity and B is the tridiagonal (1, -2, 1)/h^2 in FD.
    A and B are respectively provided in A_interior and B_interior.

    The space discretization also provides a representation of the boundary conditions.
    See @self.add_boundaries
"""


class Discretization:

    def __init__(self,
                 A=None, # Advection speed
                 C=None, # reaction coefficient
                 D1=None, # Diffusivity in \\Omega_1
                 D2=None, # Diffusivity in \\Omega_2
                 M1=None, # Number of points in \\Omega_1
                 M2=None, # Number of points in \\Omega_2
                 SIZE_DOMAIN_1=None, # Size of \\Omega_1
                 SIZE_DOMAIN_2=None, # Size of \\Omega_1
                 LAMBDA_1=None, # First Robin parameter
                 LAMBDA_2=None, # Second Robin parameter
                 DT=None): # Time step
        """
            The data needed is passed through this constructor.
            There is no implementaton of the courant number.
            If DT is changed, M1 and M2 won't change, hence the
            Courant number will change.
            The space step depends on the space discretization.
            It will be either SIZE_DOMAIN_j / M_j or SIZE_DOMAIN_j / (M_j-1)
        """
        self.A, self.R, self.D1, self.D2, \
            self.M1, self.M2, self.SIZE_DOMAIN_1, \
            self.SIZE_DOMAIN_2, self.LAMBDA_1, \
            self.LAMBDA_2, self.DT = A, C, D1, D2, \
            M1, M2, SIZE_DOMAIN_1, SIZE_DOMAIN_2, \
            LAMBDA_1, LAMBDA_2, DT

    ##############################################################################"
    #
    #   TIME DISCRETIZATIONS FUNCTIONS
    #
    #   The time discretization classes can override following functions:
    #   - integrate_one_step making a time integration from t_{n-1} to t_n
    #   - precompute_Y (optional: to speed up the time integration)
    #   - s_time_discrete: gives equivalent in a discrete setting s_d
    #   - s_time_modif: gives equivalent frequency variable in modified setting
    ##############################################################################"


    def integrate_one_step(self, f, # Right hand side of the PDE
            bd_cond, # Boundary condition at the top of atmosphere or bottom of ocean
            u_nm1, # Previous prognostic variable (can be u or phi depending on space scheme)
            u_interface, # Value of u at the interface
            phi_interface, # Value of u at the interface
            upper_domain, # True if integrating in j=2, False if j=1
            additional, # additional diasnostic variable (\\Bar{u} in finite volumes, None in FD)
            Y): # if Y was precomputed with precompute_Y, it is given here
        """
            This function will be called every time step for a single domain.
        ##############################
        # USE DOCUMENTATION:         #
        ##############################
            Entry point in the scheme.
            Provided boundary condition bd_cond, phi_interface, u_interface
            Provided former state of the equation u_nm1;
            Returns (u_n, u_interface, phi_interface)
            u_n is the next state vector, {u, phi}_interface are the
            values of u, D_j\\partial_x u at interface,
            necessary to compute Robin conditions.

            If upper_domain is True, the considered domain is Omega_2 (atmosphere)
                bd_cond is then the Neumann (/!\ not the flux, just du/dx)
                condition of the top of the atmosphere.
            If upper_domain is False, the considered domain is Omega_1 (ocean)
                bd_cond is then the Dirichlet condition of the bottom of the ocean.
            f and u_nm1 have their first values ([0,1,..]) at the interface
            and their last values ([..,M-2, M-1]) at
            the top of the atmosphere (Omega_2) or bottom of the ocean (Omega_1)

            u_interface, phi_interface, bd_cond are functions:
                u_interface(0) must give u_interface at t_{n-1}
                u_interface(1) must give u_interface at t_n
                u_interface(1/2) must give u_interface at (t_{n-1}+t_n/2)

            f can be float or np.ndarray of dimension 1.
            u_nm1 must be a np.ndarray of dimension 1

            The desired size of f can be obtained through the function size_f
            The desired size of u_nm1 can be obtained through the function size_prognostic

        ##############################
        # INHERITANCE DOCUMENTATION: #
        ##############################
            It should be implemented in a time (or space-time) discretization class.
            The space discretization is represented through the functions (they should all be used):
                -A_interior
                -B_interior
                -add_boundaries (or, alternatively, add_boundaries_to_Y and get_rhs)
                -projection_result

                -create_additional: use it when additional is []
                -update_additional: after time stepping the prognostic variable
        """

        raise NotImplementedError

    def precompute_Y(self, upper_domain):
        """
            Precompute Y for integrate_one_step. useful when integrating over a long time
            upper_domain = (j is 2)
            It is not mandatory to implement this method.
        """
        return None

    def s_time_discrete(self, w):
        """
            Equivalent frequency variable of the time scheme in a discrete setting.
            This function is sometimes undefinable
        """
        raise NotImplementedError

    def s_time_modif(self, w, order):
        """ Returns the modified s variable of the time scheme, with specified order.
            This function is always undefinable. The order 0 must be the continuous
            frequency variable, i.e. w*1j
        """
        if order == 0:
            return w*1j
        else:
            raise NotImplementedError

    ##############################################################################"
    #
    #   SPACE DISCRETIZATIONS FUNCTIONS
    #
    #   The time discretization classes can override following functions:
    #
    #   SIMULATION:
    #
    #   - A_interior
    #   - B_interior
    #   - projection_result (taking the results to return the tuple (prognostic, u(0), D\\partial_x u(0))
    #   - discretization_bd_cond (if it depends on the time scheme, implement rather hardcoded_bd_cond) 
    #   - discretization_interface  (if it depends on the time scheme, implement hardcoded_interface) 
    #   - size_f (return M ? return M-1 ? return M+1 ? depends on the space scheme.
    #   Note that most time scheme consider that f[1:-1] is the rhs of the interior.)
    #   - size_prognostic  (return M ? return M-1 ? return M+1 ? depends on the space scheme)
    #   - get_h (return SIZE_DOMAIN_j / M_j ? SIZE_DOMAIN_j/(M_j-1) ? depends on the space scheme)
    #   - get_D (if you need a variable diffusivity. also, useful to provide the size of D)
    #
    #   For particular cases:
    #       If an additional (diagnostic) state variable is needed, e.g. \\Bar{u} (default is None):
    #   - create_additional: should return a valid initialization
    #   - update_additional: is called after time steps.
    # 
    #       If the boundary conditions depends on more than the state vector:
    #   - hardcoded_bd_cond (returns the coefficients in front of the state vector and the rhs)
    #   - hardcoded_interface (identitcal to hardcoded_bd_cond but with Robin condition at interface)
    #   - crop_f_as_prognostic (when f[1:-1] is not to be added with A.prognostic
    #   
    #   ANALYSIS:
    #
    #   - eta_dirneu: analysis of interface operators in a discrete setting.
    #   - eta_dirneu_modif:  analysis of interface operators in a modified setting.
    #   - lambda_1_2_pm: lambda, giving the analytical solution in the frequency domain.
    #   - sigma_modified: equivalent of lambda_1_2_pm but in continuous or modified setting.
    ##############################################################################"
    
    #####################
    # INTEGRATION FUNCTIONS:
    #####################
    def A_interior(self, upper_domain):
        """
            gives A, such as inside the domain, A \\partial_t u = Bu + f
            For finite differences, A is identity.
            A must be returned as a tuple of 3 diagonals of same size:
            (lower diagonal, diagonal, upper diagonal)
            The boundaries conditions are given in add_boundaries_to_Y
        """
        raise NotImplementedError

    def B_interior(self, upper_domain):
        """
            gives B, such as inside the domain, A \\partial_t u = Bu + f
            B must be returned as a tuple of 3 diagonals of same size:
            (lower diagonal, diagonal, upper diagonal)
            For finite differences, B is:
            (np.ones(M-1), -2*np.ones(M-1), np.ones(M-1))/h^2
        """
        raise NotImplementedError

    def projection_result(self, result, upper_domain, partial_t_result0, f, additional, result_explicit):
        """
            given the result of the inversion, returns (u_np1, u_interface, phi_interface)
            partial_t_result0 is (\\partial_t u)(x=0), used in the corrective term in FD.
            result_explicit is used in the corrective term in FD, instead of
            result. It allows to make a corrective term explicit or semi-explicit.
        """
        raise NotImplementedError

    def discretization_bd_cond(self, upper_domain):
        """
        Gives the coefficients in front of u to compute the bd condition,
        either at the top of the atmosphere (Dirichlet)
        or at the bottom of the Ocean (Neumann).
        Returns a list, for which the index 0 is the value
        to set at the diagonal, and the index 1 (if it exists) is
        the subdiagonal.
        This ensures:
        np.dot(self.discretization_bd_cond(upper_domain=True), u_n[::-1]) = Neumann
        np.dot(self.discretization_bd_cond(upper_domain=False), u_n[::-1]) = Dirichlet
        """
        return None

    def discretization_interface(self, upper_domain):
        """
        Gives the coefficients in front of u to compute the bd condition,
        at the interface between the ocean and atmosphere.
        Returns a list, representing the vector which will make a
        scalar product with u. The vector should be of size 1, 2 or 3.
        The other numbers in the vector are assumed to be zeros.
        This ensures:
        np.dot(self.discretization_interface(upper_domain=True), u_n) = Robin
        np.dot(self.discretization_interface(upper_domain=False), u_n) = Robin
        """
        return None

    def new_additional(self, **kwargs):
        return None

    def update_additional(self, result, additional, dt, upper_domain, f, coef_reaction_implicit, reaction_explicit):
        """
            additional[m] is, in the case of finite volumes, the average of u in a cell m.
            The other space schemes give an additional=None.
            update_additional must be called after each time steps, to have an updated \\Bar{u}_n.
            coef_reaction_implicit = 1 in Euler Backward, theta in Theta Scheme, 0 in an explicit scheme.
            reaction_explicit is 0 in an implicit scheme or coef_reaction_implicit*additional_{n-1}.
        """
        # reaction_explicit is either 0 or a multiple of additional ! the idea is that
        # coef_reaction_implicit is 0 or 1 and reaction_explicit = coef_reaction_explicit * additional
        return None

    def create_additional(self, upper_domain):
        """
            Returns None or a valid "additional" variable. In finite volumes cases, it is a vector
            of M zeros.
        """
        return None

    def hardcoded_bd_cond(self, upper_domain, bd_cond, coef_implicit, coef_explicit, dt, f, sol_for_explicit, sol_unm1, additional):
        """
            Implement this method if the boundary condition is *not* np.dot(constant_vector, u)=bd_cond
            For schemes that use corrective terms or any mechanism of time derivative inside bd cond,
            this method allows the time scheme to correct the boundary condition.
            Called when discretization_bd_cond returns None. if it is never the case, no need
            to implement this method.
            additional, f, sol_unm1, sol_for_explicit, bd_cond must not be involved in the resulting list_interface.
            returns list_bd, rhs, such as the scheme ensures: np.dot(list_bd, u_n[::-1]) = rhs
        """
        raise NotImplementedError

    def hardcoded_interface(self, upper_domain, robin_cond, coef_implicit, coef_explicit, dt, f, sol_for_explicit, sol_unm1, additional):
        """
            Implement this method if the boundary condition is *not* np.dot(constant_vector, u)=Robin
            For schemes that use corrective terms or any mechanism of time derivative
            inside interface condition,
            this method allows the time scheme to correct the boundary condition.
            Called when discretization_interface returns None. if it is never the case, no need
            to implement this method.
            returns list_interface, rhs, such as the scheme ensures np.dot(list_interface, u_n) = rhs
            additional, f, sol_unm1 and sol_for_explicit, robin_cond must not be involved in the resulting list_interface.
            returns list_interface, rhs, such as the scheme ensures: np.dot(list_interface, u_n) = rhs
        """
        raise NotImplementedError

    def size_f(self, upper_domain):
        """
            Returns the size of the right hand side. Note that the time schemes will consider that:
            A_interior @ \\partial_t u[1:-1] = B_interior @ u[1:-1] + f[1:-1]
            f may need to contains 2 additional elements at extremity to be consistent with u.
        """
        M, h, D, Lambda = self.M_h_D_Lambda(upper_domain=upper_domain)
        return M

    def crop_f_as_prognostic(self, f, **args):
        return f[1:-1]

    def size_prognostic(self, upper_domain):
        """
            Returns the size of the prognostic state variable. This variable is often named "u" but
            it could represent the fluxes. (FV approaches)
        """
        M, h, D, Lambda = self.M_h_D_Lambda(upper_domain=upper_domain)
        return M


    # The following functions are not supposed to be overriden:


    def add_boundaries(self, to_inverse, rhs, interface_cond, bd_cond, upper_domain,
                    coef_implicit, coef_explicit, dt, f, sol_for_explicit, sol_unm1, additional, **kwargs):
        """
            Take the matrix to_inverse and vector rhs:
            interface_cond is of the form Lambda * u_interface + phi_interface.
            bd_cond is Neumann if j=2, Dirichlet if j=1.

            this function concatenate bd and interface conditions with rhs, and add the
            lines to "to_inverse" to be able to compute the boundaries of solution.
            The discretization must implement the functions "@self.discretization_bd_cond"
            and "@self.discretization_interface"
            Returns to_inverse and rhs: warning, the input must be tridiagonal but the output
            may not be.
            calling this function is equivalent to call add_boundaries_to_Y and get_rhs.

            coef_implicit and coef_explicit are used when the boundary conditions use
            additional elements like time derivatives (corrective terms, FV approaches).
            sol_for_explicit is the prognostic variable that is used to compute the
            time derivative in this case.
        """
        assert len(to_inverse) == 3
        assert len(rhs.shape) == 1
        new_rhs = np.concatenate(([interface_cond], rhs, [bd_cond]))
        list_bd_cond = self.discretization_bd_cond(upper_domain=upper_domain)
        list_interface = self.discretization_interface(upper_domain=upper_domain)
        if list_bd_cond is None:
            print("list_bd_cond is none")
            list_bd_cond, new_rhs[-1] = self.hardcoded_bd_cond(upper_domain=upper_domain,
                    bd_cond=bd_cond, dt=dt, f=f, sol_for_explicit=sol_for_explicit,
                    sol_unm1=sol_unm1, additional=additional,
                    coef_explicit=coef_explicit, coef_implicit=coef_implicit, **kwargs)
        if list_interface is None:
            print("list_interface is none")
            list_interface, new_rhs[0] = self.hardcoded_interface(upper_domain=upper_domain,
                    robin_cond=interface_cond, dt=dt, f=f, sol_for_explicit=sol_for_explicit,
                    sol_unm1=sol_unm1,
                    additional=additional, coef_explicit=coef_explicit, coef_implicit=coef_implicit, **kwargs)
        # let's begin with the boundary condition:
        new_Y = []
        assert len(list_bd_cond) >= 1
        #TODO : handle duplicate code (?)
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

        for additional_coeff_bd_cond in list_bd_cond[2:]:
            ret = [np.zeros(ret[0].shape[0] - 1)] + ret # new diagonal under the last one
            ret[0][-1] = additional_coeff_bd_cond

        assert max(len(list_bd_cond), 2) + max(len(list_interface), 2) - 1 == len(ret)
        return tuple(ret), new_rhs

    def add_boundaries_to_Y(self, to_inverse, upper_domain, coef_implicit, coef_explicit, dt):
        """
            Take the matrix to_inverse:
            concatenate bd and interface conditions with rhs, and add the
            lines to "to_inverse" to be able to compute the boundaries of solution.
            The discretization must implement the functions "@self.discretization_bd_cond"
            and "@self.discretization_interface"
            Returns to_inverse: warning, the input must be tridiagonal but the output
            may not be.
        """

        f = np.zeros(self.size_f(upper_domain))
        u = np.zeros(self.size_prognostic(upper_domain))
        add = self.create_additional(upper_domain=upper_domain)
        assert len(to_inverse) == 3
        list_bd_cond = self.discretization_bd_cond(upper_domain=upper_domain)
        list_interface = self.discretization_interface(upper_domain=upper_domain)
        if list_bd_cond is None:
            list_bd_cond, _ = self.hardcoded_bd_cond(upper_domain=upper_domain,dt=dt, 
                    coef_explicit=coef_explicit, coef_implicit=coef_implicit,
                    bd_cond=0., f=f, sol_for_explicit=u, sol_unm1=u, additional=add)
        if list_interface is None:
            list_interface, _ = self.hardcoded_interface(upper_domain=upper_domain, dt=dt,
                    coef_explicit=coef_explicit, coef_implicit=coef_implicit,
                    robin_cond=0., f=f, sol_for_explicit=u, sol_unm1=u, additional=add)
        # let's begin with the boundary condition:
        new_Y = []
        assert len(list_bd_cond) >= 1
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

        for additional_coeff_bd_cond in list_bd_cond[2:]:
            ret = [np.zeros(ret[0].shape[0] - 1)] + ret # new diagonal under the last one
            ret[0][-1] = additional_coeff_bd_cond

        assert max(len(list_bd_cond), 2) + max(len(list_interface), 2) - 1 == len(ret)

        return tuple(ret)

    def get_rhs(self, rhs, interface_cond, bd_cond, upper_domain,
                    coef_implicit, coef_explicit, dt, f, sol_for_explicit, sol_unm1, additional):
        """
            concatenate bd and interface conditions with rhs, optionnaly adding the terms needed,
            then returns rhs.
            The discretization must implement the functions @self.discretization_bd_cond
            and @self.discretization_interface
            also, to add terms needed, discretization may return None in the previous methods
            and implement @self.hardcoded_bd_cond and @self.hardcoded_interface
        """
        assert len(rhs.shape) == 1
        new_rhs = np.concatenate(([interface_cond], rhs, [bd_cond]))
        if self.discretization_bd_cond(upper_domain=upper_domain) is None:
            _, new_rhs[-1] = self.hardcoded_bd_cond(upper_domain=upper_domain,
                    bd_cond=bd_cond, dt=dt, f=f, sol_for_explicit=sol_for_explicit,
                    sol_unm1=sol_unm1, additional=additional,
                    coef_explicit=coef_explicit, coef_implicit=coef_implicit)
        if self.discretization_interface(upper_domain=upper_domain) is None:
            _, new_rhs[0] = self.hardcoded_interface(upper_domain=upper_domain,
                    robin_cond=interface_cond, dt=dt, f=f, sol_for_explicit=sol_for_explicit,
                    sol_unm1=sol_unm1,
                    additional=additional, coef_explicit=coef_explicit, coef_implicit=coef_implicit)
        return new_rhs


    #####################
    # ANALYSIS FUNCTIONS:
    #####################

    def analytic_robin_robin_modified(self, w, order_time=float('inf'), order_operators=float('inf'), order_equations=float('inf')):
        """
            When D and h are constant, it is possible to find the convergence
            factor in frequency domain. analytic_robin_robin_modified computes this convergence rate.

            This method should *not* be overriden, the
            particularity of the discretization is specified
            through the methods:
            eta_dirneu{,_modif}, sigma_modified, lambda_1_2_pm, and s_time_{modif, discrete}

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
            s = self.s_time_discrete(w)
        else:
            s = self.s_time_modif(w, order_time)

        ###################################################################
        # Computing \\sigma_j or \\lambda_j depending on order_equations: #
        ###################################################################
        # The convention here is: \\lambda_- is the main root,
        # and \\lambda_+ is the secondary root.
        if order_equations == float('inf'):
            lam1, lam2, lam1_p, lam2_p = self.lambda_1_2_pm(s)
            h1, h2 = self.get_h()
            h1, h2 = h1[0], h2[0]
            sig1, sig2, sig1_p, sig2_p = np.log(lam1)/h1, np.log(lam2)/h2, np.log(lam1_p)/h1, np.log(lam2_p)/h2
        else:
            sig1, sig2 = self.sigma_modified(w, s, order_equations)
            sig1_p = -sig1
            sig2_p = -sig2

        #########################################################
        # Computing \\eta_{j, op} depending on order_operators: #
        #########################################################

        if order_operators == float('inf'):
            h1, h2 = self.get_h()
            h1, h2 = h1[0], h2[0]
            eta1_dir, eta1_neu = self.eta_dirneu(j=1, lam_m=np.exp(h1*sig1), lam_p=np.exp(h1*sig1_p), s=s)
            eta2_dir, eta2_neu = self.eta_dirneu(j=2, lam_m=np.exp(h2*sig2), lam_p=np.exp(h2*sig2_p), s=s)
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

    def eta_dirneu_modif(self, j, sigj, order_operators, *kwargs, **dicargs):
        """ Returns the modified eta variable of the time scheme, with specified order"""
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

    def sigma_modified(self, w, s, order_equations):
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

    def get_a_r_dt(self):
        """
            Returns default values of a, c, dt or parameters if given.
        """
        assert "A" in self.__dict__
        assert "R" in self.__dict__
        assert "DT" in self.__dict__
        return self.A, self.R, self.DT

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
