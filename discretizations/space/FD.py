"""
    Finite differences for diffusion advection reaction equation.
    The functions to use are integrate_one_step and integrate_one_step_star.
    Theses function make a single step in time. It is not efficient
    (need to compute the matrix each time) but it is simple.
"""

import numpy as np
from discretizations.discretization import Discretization
from utils_linalg import solve_linear


class FiniteDifferences(Discretization):

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
            gives A, such as inside the domain, A \\partial_t u = Bu
            For finite differences, A is identity.
        """
        M, h, D, _ = self.M_h_D_Lambda(upper_domain=upper_domain)
        return (np.zeros(M-2), np.ones(M-2), np.zeros(M-2))

    def B_interior(self, upper_domain):
        """
            gives B, such as inside the domain, A \\partial_t u = Bu
        """
        a, c, dt = self.get_a_c_dt()
        M, h, D, _ = self.M_h_D_Lambda(upper_domain=upper_domain)
        #left diagonal: applies to u[:-2]
        Y_0 = D[1:]/h[1:]/((h[1:] + h[:-1])/2) + a / (h[1:] + h[:-1])
        #right diagonal: applies to u[2:]
        Y_2 = D[:-1]/h[:-1]/((h[1:] + h[:-1])/2) - a / (h[1:] + h[:-1])
        # diagonal: applies to u[1:-1]
        Y_1 = - (D[1:]/h[1:] + D[:-1]/h[:-1])/((h[1:] + h[:-1])/2) - c
        return (Y_0, Y_1, Y_2)

    def discretization_bd_cond(self, upper_domain):
        """
        Gives the coefficients in front of u to compute the bd condition,
        either at the top of the atmosphere (Dirichlet)
        or at the bottom of the Ocean (Neumann)
        """
        # starting from index -1
        if upper_domain: # Dirichlet:
            return [1]
        else: # Neumann:
            M, h, D, _ = self.M_h_D_Lambda(upper_domain=upper_domain)
            return [1/h[-1], -1/h[-1]]

    def discretization_interface(self, upper_domain):
        _, _, _, Lambda = self.M_h_D_Lambda(upper_domain=upper_domain)
        ret = self.discretization_flux_interface(upper_domain)
        if ret is None:
            return None
        ret[0] += Lambda
        return ret

    def discretization_flux_interface(self, upper_domain):
        raise NotImplementedError("Use FD_naive, FD_corr or FD_extra")

    def projection_result(self, result, upper_domain, **kwargs):
        """
            given the result of the inversion, returns (u_np1, u_interface, phi_interface)
        """
        M, h, D, Lambda = self.M_h_D_Lambda(upper_domain=upper_domain)
        phi = 0
        for coeff, um in zip(self.discretization_flux_interface(upper_domain), result):
            phi += um * coeff
        return result, result[0], phi

    #####################
    # ANALYSIS FUNCTIONS:
    #####################

    def lambda_1_2_pm(self, s):
        """
            Gives the \\lambda_\\pm:
            returns \\lambda_{-, j=1}, \\lambda_{-, j=2}, \\lambda_{+, j=1}, \\lambda_{+, j=2}.
        """
        assert s is not None
        a, c, dt = self.get_a_c_dt()
        M1, h1, D1, Lambda_1 = self.M_h_D_Lambda(upper_domain=False)
        M2, h2, D2, Lambda_2 = self.M_h_D_Lambda(upper_domain=True)
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
            s1 -= (s + self.C)**2 * (h1**2/(12*D1))
        if order_equations > 1:
            s1 += (s + self.C)**3 * h1**4/(90*D1**2)

        s2 = np.copy(s)
        if order_equations > 0:
            s2 -= (s + self.C)**2 * (h2**2/(12*D2))
        if order_equations > 1:
            s2 += (s + self.C)**3 * h2**4/(90*D2**2)

        sig1 = np.sqrt((s1+self.C)/self.D1)
        sig2 = -np.sqrt((s2+self.C)/self.D2)
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
        x1_1_2 = x1[:-1] + h1 / 2
        x2_1_2 = x2[:-1] + h2 / 2
        D1 = function_D1(x1_1_2)
        D2 = function_D2(x2_1_2)
        return D1, D2

    def name(self):
        return "Différences finies"

    def repr(self):
        return "finite differences"


if __name__ == "__main__":
    from tests import test_finite_differences
    test_finite_differences.launch_all_tests()
