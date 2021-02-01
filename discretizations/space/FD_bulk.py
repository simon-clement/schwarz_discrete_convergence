"""
    Finite differences for diffusion advection reaction equation.
    The functions to use are integrate_one_step and integrate_one_step_star.
    Theses function make a single step in time. It is not efficient
    (need to compute the matrix each time) but it is simple.
    WWWE ARE  SOLVING ON THE DERIVATIVES AND NOT ON THE FLUXES !
"""

import numpy as np
from discretizations.discretization import Discretization
from utils_linalg import solve_linear


class FiniteDifferencesBulk(Discretization):

    def __init__(self, *args, **kwargs):
        """
            give default values of all variables.
        """
        super().__init__(*args, **kwargs)

    
    #####################
    # INTEGRATION FUNCTIONS:
    #####################
    def A_interior(self, upper_domain):
        """
            gives A, such as inside the domain, A \\partial_t phi = Bu
            For finite differences, A is identity.
        """
        M, h, D, _ = self.M_h_D_Lambda(upper_domain=upper_domain)
        return (np.zeros(M-2), np.ones(M-2), np.zeros(M-2))

    def B_interior(self, upper_domain):
        """
            gives B, such as inside the domain, A \\partial_t phi = Bu
        """
        a, c, dt = self.get_a_r_dt()
        M, h, D, _ = self.M_h_D_Lambda(upper_domain=upper_domain)
        #left diagonal: applies to u[:-2]
        Y_0 = D[2:]/(h[1:]*h[1:]) + a / (h[1:] + h[:-1])
        #right diagonal: applies to u[2:]
        Y_2 = D[:-2]/(h[:-1]*h[1:]) - a / (h[1:] + h[:-1])
        # diagonal: applies to u[1:-1]
        Y_1 = - 2*D[1:-1]/(h[:-1]*h[1:]) - c
        return (Y_0, Y_1, Y_2)

    def discretization_bd_cond(self, upper_domain):
        """
        see hardcoded_bd_cond. using temporal disc.
        """
        return None

    def hardcoded_bd_cond(self, upper_domain, bd_cond, coef_implicit, coef_explicit, dt, f, sol_for_explicit, additional, override_r=None, **kwargs):
        """
            For schemes that use corrective terms or any mechanism of time derivative inside bd cond,
            this method allows the time scheme to correct the boundary condition.
            in this case, sol_for_explicit will be \\phi (in n-1/2 or n-1)
            and additional will be u (at time n-1)
            f is the right hand side of the PDE.
            the part of time scheme is represented by doing the integration
            (u^{n+1} - u^n)/dt = coef_implicit * dx^2 u^{n+1} + coef_explicit * dx^2 u^n
            don't forget to interpolate f in time before calling function

            TODO: include reaction in the computation
            must return a tuple (coefficients, bd_cond)
        """
        if not upper_domain: # Neumann
            M, h, D, _ = self.M_h_D_Lambda(upper_domain=upper_domain)
            return ([1], bd_cond)
        else: # Dirichlet :
            M, h, D, _ = self.M_h_D_Lambda(upper_domain=upper_domain)
            a, c, _ = self.get_a_r_dt()

            if override_r is not None:
                c = override_r

            assert h[-1] > 0
            assert sol_for_explicit is not None
            assert additional is not None
            return D[-1] * dt / h[-1] * np.array([3/2, -2, 1/2]), bd_cond - 3/2 * additional[-1] + additional[-2] / 2 + dt / 2 * (f[-2] - 3*f[-1])

    def discretization_interface(self, upper_domain):
        return None

    def hardcoded_interface(self, upper_domain, robin_cond, coef_implicit, coef_explicit, dt, f, sol_for_explicit, additional, override_r=None, **kwargs):
        """
            For schemes that use corrective terms or any mechanism of time derivative inside interface condition,
            this method allows the time scheme to correct the boundary condition.
            in this case, sol_for_explicit will be \\phi and additional will be u (both in n-1)
            f is the right hand side of the PDE.
            the part of time scheme is represented by doing the integration
            (u^{n+1} - u^n)/dt = coef_implicit * dx^2 u^{n+1} + coef_explicit * dx^2 u^n
            don't forget to interpolate f in time before calling function
        """
        M, h, D, Lambda = self.M_h_D_Lambda(upper_domain=upper_domain)
        a, c, _ = self.get_a_r_dt()
        if override_r is not None:
            c = override_r

        return (D[0] * np.array([1, 0, 0]) + Lambda * np.array([-3/2, 2, -1/2]) * D[0] * dt / h[0],
               robin_cond + Lambda * (additional[1]/2 - 3*additional[0]/2 + dt/2*(f[1] - 3*f[0])))

    def create_additional(self, upper_domain): # u at a point
        M, h, D, Lambda = self.M_h_D_Lambda(upper_domain=upper_domain)
        return np.zeros(M-1)

    def update_additional(self, result, additional, dt, f, upper_domain,
            reaction_explicit, coef_reaction_implicit):
        # starting from additional=u^n, making additional=u^{n+1}
        # average of u on a cell.
        a, c, _ = self.get_a_r_dt()
        M, h, D, _ = self.M_h_D_Lambda(upper_domain=upper_domain)
        return additional + dt* (np.diff(D*result) / h + f)

    def new_additional(self, result, upper_domain, cond):
        print("Why are you creating a new variable u ? Are you sure it is the right thing to do ?")
        raise
        if upper_domain: # cond is a Dirichlet condition
            _, h2 = self.get_h()
            _, D = self.get_D()
            # first: determine the solution at u{-1}:
            u_boundary = cond - result[-1] / h2[-1] / 2
            # then the cumulative sum: 
            return np.flip(u_boundary + np.cumsum(np.flip(result[:-1] / h2)))
        else: # cond is the Robin condition
            h1, _ = self.get_h()
            D, _ = self.get_D()
            Lambda = self.LAMBDA_1
            # first: determine the solution at u{-1}:
            u_boundary = (cond - result[0]) / Lambda + result[0] / h1[0] / 2
            # then the cumulative sum: 
            return u_boundary + np.cumsum(result[1:] / h1)

    def crop_f_as_prognostic(self, f, upper_domain):
        _, h, _, _ = self.M_h_D_Lambda(upper_domain=upper_domain)
        return np.diff(f) / ((h[1:] + h[:-1])/2)

    def projection_result(self, result, upper_domain, additional, f, result_explicit, **kwargs):
        """
            given the result of the inversion, returns (phi_np1, u_interface, phi_interface)
        """
        M, h, D, _ = self.M_h_D_Lambda(upper_domain=upper_domain)
        u = additional[0] - result[0] * h[0]/2
        return result, u, D[0]*result[0], additional

    def eta_dirneu(self, j, lam_m, lam_p, s=None):
        """
            note that lam_m is by convention the main lambda
            whereas lam_p is the secondary lambda
            
            Gives the \\eta of the discretization:
            can be:
                -eta(1, ..);
                -eta(2, ..);
            returns tuple (etaj_dir, etaj_neu).
        """
        assert j == 1 or j == 2
        M, h, D, _ = self.M_h_D_Lambda(upper_domain=(j==2))
        a, c, dt = self.get_a_r_dt()
        raise NotImplementedError()

    def size_f(self, upper_domain):
        if upper_domain:
            return self.M2 - 1
        else:
            return self.M1 - 1


    def eta_dirneu_modif(self, j, sigj, order_operators, w, *kwargs, **dicargs):
        # This code should not run and is here as an example
        h1, h2 = self.get_h()
        raise NotImplementedError()

    #####################
    # ANALYSIS FUNCTIONS:
    #####################

    def lambda_1_2_pm(self, s):
        """
            Gives the \\lambda_\\pm:
            returns \\lambda_{-, j=1}, \\lambda_{-, j=2}, \\lambda_{+, j=1}, \\lambda_{+, j=2}.
        """
        assert s is not None
        a, c, dt = self.get_a_r_dt()
        M1, h1, D1, _ = self.M_h_D_Lambda(upper_domain=False)
        M2, h2, D2, _ = self.M_h_D_Lambda(upper_domain=True)
        h1, h2 = h1[0], h2[0]
        D1, D2 = D1[0], D2[0]

        Y_0 = -D1 / (h1 * h1) - .5 * a / h1
        Y_1 = 2 * D1 / (h1 * h1) + c
        Y_2 = -D1 / (h1 * h1) + .5 * a / h1

        lambda1_moins = (Y_1 + s - np.sqrt((Y_1 + s)**2 - 4 * Y_0 * Y_2)) \
                                / (-2 * Y_2)
        lambda1_plus = (Y_1 + s + np.sqrt((Y_1 + s)**2 - 4 * Y_0 * Y_2)) \
                                / (-2 * Y_2)

        Y_0 = -D2 / (h2 * h2) - .5 * a / h2
        Y_1 = 2 * D2 / (h2 * h2) + c
        Y_2 = -D2 / (h2 * h2) + .5 * a / h2

        lambda2_moins = (Y_1 + s - np.sqrt((Y_1 + s)**2 - 4 * Y_0 * Y_2)) \
                                / (-2 * Y_2)
        lambda2_plus = (Y_1 + s + np.sqrt((Y_1 + s)**2 - 4 * Y_0 * Y_2)) \
                                / (-2 * Y_2)
        return lambda1_moins, lambda2_moins, lambda1_plus, lambda2_plus, 

    def sigma_modified(self, w, s, order_equations):
        h1, h2 = self.get_h()
        h1, h2 = h1[0], h2[0]
        D1, D2 = self.D1, self.D2

        s1 = np.copy(s)
        if order_equations > 0:
            s1 -= (s + self.R)**2 * (h1**2/(12*D1))
        if order_equations > 1:
            s1 += (s + self.R)**3 * h1**4/(90*D1**2)

        s2 = np.copy(s)
        if order_equations > 0:
            s2 -= (s + self.R)**2 * (h2**2/(12*D2))
        if order_equations > 1:
            s2 += (s + self.R)**3 * h2**4/(90*D2**2)

        sig1 = np.sqrt((s1+self.R)/self.D1)
        sig2 = -np.sqrt((s2+self.R)/self.D2)
        raise NotImplementedError()
        return sig1, sig2


    def get_h(self):
        """
            Simple function to return h in each subdomains,
            in the framework of finite differences.
            returns uniform spaces between points (h1, h2).
            To recover xi, use:
            xi = np.cumsum(np.concatenate(([0], hi)))
        """
        size_domain_1 = self.SIZE_DOMAIN_1
        size_domain_2 = self.SIZE_DOMAIN_2
        M1 = self.M1
        M2 = self.M2
        x1 = -np.linspace(0, size_domain_1, M1)
        x2 = np.linspace(0, size_domain_2, M2)
        return np.diff(x1), np.diff(x2)


    def get_D(self, h1=None, h2=None, function_D1=None, function_D2=None):
        """
            Simple function to return D in each subdomains,
            in the framework of finite differences.
            provide continuous functions accepting ndarray
            for D1 and D2, and returns the right coefficients.
            By default, D1 and D2 are constant.
        """
        if h1 is None or h2 is None:
            h1, h2 = self.get_h()
        if function_D1 is None:
            def function_D1(x): return self.D1 + np.zeros_like(x)
        if function_D2 is None:
            def function_D2(x): return self.D2 + np.zeros_like(x)
        x1 = np.cumsum(np.concatenate(([0], h1)))
        x2 = np.cumsum(np.concatenate(([0], h2)))
        # coordinates at half-points:
        D1 = function_D1(x1)
        D2 = function_D2(x2)
        return D1, D2

    def name(self):
        return "Différences finies bulk"

    def repr(self):
        return "finite differences bulk"


if __name__ == "__main__":
    from tests import test_finite_differences
    test_finite_differences.launch_all_tests()
