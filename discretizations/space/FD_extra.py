"""
    Finite differences for diffusion advection reaction equation.
    The functions to use are integrate_one_step and integrate_one_step_star.
    Theses function make a single step in time. It is not efficient
    (need to compute the matrix each time) but it is simple.
"""

import numpy as np
from discretizations.space.FD import FiniteDifferences
from utils_linalg import solve_linear_with_ultra_right, solve_linear


class FiniteDifferencesExtra(FiniteDifferences):

    def __init__(self, *args, **kwargs):
        """
            give default values of all variables.
        """
        super().__init__(*args, **kwargs)

    def discretization_flux_interface(self, upper_domain):
        M, h, D, Lambda = self.M_h_D_Lambda(upper_domain=upper_domain)
        return [-(2*h[0]+h[1])/(h[0]+h[1])*D[0]/h[0],
                (2*h[0]+h[1])/(h[0]+h[1])*D[0]/h[0] + h[0]/(h[0]+h[1]) * D[1] / h[1],
                -h[0]/(h[0]+h[1])*D[1] / h[1]]

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
        h, D = h[0], D[0]
        a, c, dt = self.get_a_r_dt()
        if s is None:
            s = 1 / dt

        lambda_moins = lam_m
        lambda_plus = lam_p

        # The computation is then different because the boundary condition is different
        if j == 1:
            eta1_dir = 1 + (lambda_moins / lambda_plus) ** M
            eta1_neu = D/h * ((lambda_moins - 1) * (3/2 - lambda_moins/2) \
                    + (lambda_plus - 1) * (3/2 - lambda_plus/2) \
                    * (lambda_moins / lambda_plus)**M)

            sol_calc = (s*h/2 + np.sqrt((s*h/2)**2 + s*D))*(1 - s*h*h/(4*D) - h/(2*D)*np.sqrt((s*h/2)**2 + s*D))
            sol_calc2 = (-s*h*h/(2*D) + 1)*np.sqrt((s*h/2)**2 + s*D) - s**2*h**3/(4*D)
            #if np.any(np.abs(sol_calc-D/h * ((lambda_moins - 1) * (3/2 - lambda_moins/2))) > 1e-6):
            #    print("Error Euler extra 1: computation is not right")
            #    print(abs(sol_calc-eta1_neu)/abs(eta1_neu), eta1_neu)
            #    #raise
            return eta1_dir, eta1_neu
        elif j == 2:
            eta2_dir = 1 + (lambda_moins-1) / (lambda_plus - 1) *(lambda_moins / lambda_plus) ** (M - 1)
            eta2_neu = D/h * ((lambda_moins - 1) * (3/2 - lambda_moins/2) \
                    + (lambda_plus - 1)  * (3/2 - lambda_plus/2) \
                    * (lambda_moins-1) / (lambda_plus - 1) *(lambda_moins / lambda_plus) ** (M - 1))
            sol_calc = (s*h/2 - np.sqrt((s*h/2)**2 + s*D))*(1 - s*h*h/(4*D) + h/(2*D)*np.sqrt((s*h/2)**2 + s*D))
            sol_calc2 = (s*h*h/(2*D) - 1)*np.sqrt((s*h/2)**2 + s*D) - s**2*h**3/(4*D)
            #if np.any(np.abs(sol_calc-D/h * ((lambda_moins - 1) * (3/2 - lambda_moins/2))) > 1e-6):
            #    print("Error Euler extra 1: computation is not right")
            #    print(abs(sol_calc-eta2_neu)/abs(eta2_neu), eta2_neu)


            return eta2_dir, eta2_neu

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

