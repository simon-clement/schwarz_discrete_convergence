"""
    Warning: the scheme analysis is only correct with Backward Euler time scheme.
    the other schemes fail to recover the full solution.
    For instance, ThetaMethod should consider the boundary conditions given
    to be at time (n+theta), which is not compatible to the system solved,
    where we need the boundary conditions at n+1.
"""

import numpy as np
from discretizations.space.FD import FiniteDifferences
from utils_linalg import solve_linear


class FiniteDifferencesCorr(FiniteDifferences):
    """
        The corrective term only works with a Backward Euler Scheme right now !
    """

    def __init__(self, *args, **kwargs):
        """
            give default values of all variables.
        """
        super().__init__(*args, **kwargs)

    def discretization_flux_interface(self, upper_domain):
        """
            it is not possible to implement this bd condition,
            since there is a derivative in time. We need to implement hardcoded_interface instead
        """
        return None

    def projection_result(self, result, upper_domain, partial_t_result0, f, result_explicit, **kwargs):
        """
            given the result of the inversion, returns (u_np1, u_interface, phi_interface)
        """
        M, h, D, Lambda = self.M_h_D_Lambda(upper_domain=upper_domain)
        a, c, dt = self.get_a_c_dt()
        h, D = h[0], D[0]
        phi = D*(result_explicit[1] - result_explicit[0])/h
        phi -= h/2 * (partial_t_result0 \
                + a * (result_explicit[1] - result_explicit[0]) / h
                + c * result_explicit[0] - f[0])

        return result, result_explicit[0], phi

    def hardcoded_interface(self, upper_domain, robin_cond, coef_implicit, coef_explicit, dt, f, sol_for_explicit, sol_unm1, additional, override_r=None, **kwargs):
        """
            For schemes that use corrective terms or any mechanism of time derivative inside interface condition,
            this method allows the time scheme to correct the boundary condition.
            in this case, sol_for_explicit will be \\phi and additional will be \\Bar(u) (both in n-1)
            f is the right hand side of the PDE.
            the part of time scheme is represented by doing the integration
            (u^{n+1} - u^n)/dt = coef_implicit * dx^2 u^{n+1} + coef_explicit * dx^2 u^n
            don't forget to interpolate f in time before calling function
        """
        M, h, D, Lambda = self.M_h_D_Lambda(upper_domain=upper_domain)
        a, c, _ = self.get_a_c_dt()
        if override_r is not None:
            c = override_r

        h, D = h[0], D[0]

        if coef_implicit == 0: # special case : problem
            print("Warning: using FD corr with an explicit scheme, it was not coded for this")
            return np.array([Lambda - D/h, D/h]) - \
                   h/2 * np.array([(1/dt + coef_implicit* (c - a/h) ), coef_implicit*a/h]), \
                   robin_cond + h/2 * (-sol_unm1[0] / dt - f[0] +\
                           coef_explicit*(a*(sol_for_explicit[1] - sol_for_explicit[0])/h + c*sol_for_explicit[0]))
        else:
            return coef_implicit * np.array([Lambda - D/h, D/h]) - \
                   h/2 * np.array([(1/dt + coef_implicit* (c - a/h) ), coef_implicit*a/h]), \
                   robin_cond - coef_explicit * np.dot(np.array([Lambda - D/h, D/h]), sol_for_explicit[:2]) \
                   + h/2 * (-sol_unm1[0] / dt - f[0] +\
                           coef_explicit*(a*(sol_for_explicit[1] - sol_for_explicit[0])/h + c*sol_for_explicit[0]))


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
        M, h, D, Lambda = self.M_h_D_Lambda(upper_domain=(j==2))
        a, c, dt = self.get_a_c_dt()
        D = D[0]
        if s is None:
            s = 1 / dt

        if j == 1:
            h = -self.SIZE_DOMAIN_1 / (M - 1)
        elif j == 2: 
            h = self.SIZE_DOMAIN_2 / (M - 1)

        lambda_moins = lam_m
        lambda_plus = lam_p

        # The computation is then different because the boundary condition is different
        if j == 1:
            eta1_dir = 1 + (lambda_moins / lambda_plus) ** M
            eta1_neu = (D/h - a/2) * (lambda_moins - 1 + (lambda_plus - 1) * (lambda_moins / lambda_plus)**M) - h/2 * (s + c) * (1 + (lambda_moins / lambda_plus)**M)
            return eta1_dir, eta1_neu
        elif j == 2:
            eta2_dir = 1 + (lambda_moins-1) / (lambda_plus - 1) *(lambda_moins / lambda_plus) ** (M - 1)
            eta2_neu = (D/h - a/2) * (lambda_moins - 1 + (lambda_plus - 1) * (lambda_moins-1) / (lambda_plus - 1) *(lambda_moins / lambda_plus) ** (M - 1)) - h/2 * (s+c) * (1 + (lambda_moins-1) / (lambda_plus - 1) *(lambda_moins / lambda_plus) ** (M - 1))
            return eta2_dir, eta2_neu


    def eta_dirneu_modif(self, j, sigj, order_operators, w, *kwargs, **dicargs):
        # This code should not run and is here as an example
        h1, h2 = self.get_h()
        h1, h2 = h1[0], h2[0]
        D1, D2 = self.D1, self.D2
        dt = self.DT
        if j==1:
            eta_dir_modif = 1
            if  order_operators == 0:
                eta_neu_modif = D1*sigj
            if  order_operators > 0:
                eta_neu_modif = D1*sigj*np.exp(h1*sigj/2) - h1/2*(1j*w)
            if  order_operators > 1:
                eta_neu_modif += D1*h1**2*sigj**3/24*np.exp(h1*sigj/2) - h1/4*(dt*w*w)
            return eta_dir_modif, eta_neu_modif
        else:
            eta_dir_modif = 1
            if  order_operators == 0:
                eta_neu_modif = D2*sigj
            if order_operators > 0:
                eta_neu_modif = D2*sigj*np.exp(h2*sigj/2) - h2/2*(1j*w)
            if order_operators > 1:
                eta_neu_modif += D2*h2**2*sigj**3/24*np.exp(h2*sigj/2) - h2/4*(dt*w*w)
            return eta_dir_modif, eta_neu_modif

