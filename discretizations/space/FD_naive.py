"""
    Finite differences for diffusion advection reaction equation.
    The functions to use are integrate_one_step and integrate_one_step_star.
    Theses function make a single step in time. It is not efficient
    (need to compute the matrix each time) but it is simple.
"""

import numpy as np
from discretizations.space.FD import FiniteDifferences
from utils_linalg import solve_linear


class FiniteDifferencesNaive(FiniteDifferences):

    def __init__(self, *args, **kwargs):
        """
            give default values of all variables.
        """
        super().__init__(*args, **kwargs)

    def discretization_flux_interface(self, upper_domain):
        M, h, D, Lambda = self.M_h_D_Lambda(upper_domain=upper_domain)
        return [-D[0]/h[0], D[0]/h[0]]


    def eta_dirneu(self, j, lam_m, lam_p, s=None):
        """
            Gives the \\eta of the discretization:
            can be:
                -eta(1, ..);
                -eta(2, ..);
            returns tuple (etaj_dir, etaj_neu).
        """
        assert j == 1 or j == 2
        M, h, D, Lambda = self.M_h_D_Lambda(upper_domain=(j==2))
        a, c, dt = self.get_a_c_dt()
        h, D = h[0], D[0]
        if s is None:
            s = 1 / dt

        lambda_moins = lam_m
        lambda_plus = lam_p

        # The computation is then different because the boundary condition is different
        if j == 1:
            eta1_dir = 1 + (lambda_moins / lambda_plus) ** M
            eta1_neu = D/h * (lambda_moins - 1 + (lambda_plus - 1) * (lambda_moins / lambda_plus)**M)
            return eta1_dir, eta1_neu
        elif j == 2:
            eta2_dir = 1 + (lambda_moins-1) / (lambda_plus - 1) *(lambda_moins / lambda_plus) ** (M - 1)
            eta2_neu = D/h * (lambda_moins - 1 + (lambda_plus - 1) * (lambda_moins-1) / (lambda_plus - 1) *(lambda_moins / lambda_plus) ** (M - 1))
            return eta2_dir, eta2_neu

    def eta_dirneu_modif(self, j, sigj, order_operators, w, *kwargs, **dicargs):
        # This code should not run and is here as an example
        h1, h2 = self.get_h()
        h1, h2 = h1[0], h2[0]
        D1, D2 = self.D1, self.D2
        if j==1:
            hj = h1
            Dj = D1
        else:
            hj = h2
            Dj = D2

        eta_dir_modif = 1
        if order_operators == 0:
            eta_neu_modif = Dj*sigj
        if order_operators > 0:
            eta_neu_modif = Dj*sigj*np.exp(hj*sigj/2)
        if order_operators > 1:
            eta_neu_modif += Dj*hj**2*sigj**3/24*np.exp(hj*sigj/2)
        return eta_dir_modif, eta_neu_modif

