"""
    Finite volume for diffusion advection reaction equation.
    The functions to use are integrate_one_step and integrate_one_step_star.
    Theses function make a single step in time. It is not efficient
    (need to compute the matrix each time) but it is simple.
"""

import numpy as np
from utils_linalg import solve_linear
from discretizations.discretization import Discretization
from discretizations.space.FD import FiniteDifferences
from utils_linalg import solve_linear
from utils_linalg import scal_multiply
from utils_linalg import multiply_interior
from utils_linalg import multiply
from utils_linalg import add_banded


class PadeFDCorr(FiniteDifferences):

    def __init__(self, *args, **kwargs):
        """
            give default values of all variables.
        """
        super().__init__(*args, **kwargs)

    def integrate_one_step(self, f, bd_cond, u_nm1, u_interface,
            phi_interface, upper_domain=True, Y=None, **kwargs):

        a = 1. + np.sqrt(2)
        b = 1. + 1/np.sqrt(2)

        # WITH 3rd degree interpolation of scipy.interp1d :
        # def get_star(func): return func(b-a)

        # WITH GAMMA = b + z(b-a) = z - b*(z-1)
        def get_star(func): return b*func(0) + (b-a)*func(1)

        # WITH GAMMA = b/2 (1 - z^2) + z(2b-a) = z - b*(z-1) - b/2 * (z-1)**2
        # def get_star(func): return b/2*(func(0) - func(2)) + (2*b-a)*func(1)

        u_star_interface = get_star(u_interface)
        phi_star_interface = get_star(phi_interface)
        f_star = get_star(f)
        bd_cond_star = get_star(bd_cond)

        f_nm1 = f(0)
        f_nm1_2 = f(1/2)
        f = f(1)
        bd_cond_nm1 = bd_cond(0)
        bd_cond = bd_cond(1)
        u_nm1_interface = u_interface(0)
        u_interface = u_interface(1)
        phi_nm1_interface = phi_interface(0)
        phi_interface = phi_interface(1)

        assert upper_domain is True or upper_domain is False

        M, h, D, Lambda = self.M_h_D_Lambda(upper_domain=upper_domain)

        # The idea is that in semi-discrete in space, we have: A\partial_t u = B u
        A = self.A_interior(upper_domain=upper_domain)
        B = self.B_interior(upper_domain=upper_domain)

        # naive method for rhs: f_first_step = (b*f_star - a*f_nm1)
        f_first_step = b*f_star - a*f_nm1
        # f_first_step = np.zeros_like(f_nm1_2)
        f_bd = f_first_step

        ###################
        # FIRST STEP : find the time derivative at star time
        # The equation is A(u*-u_nm1)/dt - (b*Bu* - a*Bu_nm1) + reaction*Au_nm1 = f(1/2)
        ###################
        to_inverse = add_banded(scal_multiply(A, 1./self.DT), scal_multiply(B, -b))
        rhs = multiply_interior(A, u_nm1) * (1 / self.DT) \
                - a * multiply_interior(B, u_nm1) + f_first_step[1:-1]

        ###########################
        # the equation given by add_boundaries is (with e = e^{n+1} when without "*"):
        # Lambda (b* - a)e + (b* - a) phi - h/2 * ((e*-e)/dt + c (b* - a) e - f) = cond_robin_star
        ##############################
        cond_robin_star = Lambda * (b*u_star_interface - a*u_nm1_interface) + \
                b*phi_star_interface - a*phi_nm1_interface + \
                h[0]/2 * (u_star_interface - u_nm1_interface)/self.DT + \
                h[0]/2 * self.R*(b*u_star_interface - a*u_nm1_interface) - \
                h[0]/2 * f_first_step[0]
                

        Y, rhs = self.add_boundaries(to_inverse=to_inverse, rhs=rhs, interface_cond=cond_robin_star,
                                     coef_explicit=-a, coef_implicit=b,
                                     dt=self.DT, f=f_bd, sol_for_explicit=u_nm1, sol_unm1=u_nm1,
                                     bd_cond=bd_cond_star, upper_domain=upper_domain,
                                     additional=None)
        result_star = solve_linear(Y, rhs)

        ###################
        # SECOND STEP : We need to inverse the same tridiagonal matrix, except the boundaries
        # The equation is (u_np1-u*)/dt - b*Bu_np1 = 0, WITHOUT REACTION AND FORCING TERM
        ###################
        # keeping the same matrix to inverse
        # to_inverse = add_banded(scal_multiply(A, 1/self.DT), scal_multiply(B, -b))
        # This time the rhs does not take into account the forcing term

        # naive method for rhs: f_second_step = b*f
        f_second_step = b*f
        #f_second_step = np.zeros_like(f)
        f_bd = f_second_step/b
        rhs = multiply_interior(A, result_star) / self.DT + f_second_step[1:-1]

        ###########################
        # the equation given by add_boundaries is:
        # Lambda e + phi - h/2 * ((e-e*)/(b dt) + c e - f) = cond_robin_star (with e = e^{n+1})
        ##############################
        cond_robin = Lambda * u_interface + phi_interface + \
                h[0]/2 * (u_interface - u_star_interface) / (b * self.DT) + \
                h[0]/2 * self.R * u_interface - \
                h[0]/2 * f_bd[0]
    
        Y, rhs = self.add_boundaries(to_inverse=to_inverse, rhs=rhs, interface_cond=cond_robin,
                                     coef_explicit=0., coef_implicit=1.,
                                     dt=b*self.DT, f=f_bd, sol_for_explicit=result_star,
                                     sol_unm1=result_star,
                                     bd_cond=bd_cond, upper_domain=upper_domain,
                                     additional=None)
        result = solve_linear(Y, rhs)

        return result, result[0], D[0] * (result[1] - result[0])/h[0]

    def hardcoded_interface(self, upper_domain, robin_cond, coef_implicit, coef_explicit, dt, f, sol_for_explicit, sol_unm1, additional, override_r=None, **kwargs):
        """
            For schemes that use corrective terms or any mechanism of time derivative inside interface condition,
            this method allows the time scheme to correct the boundary condition.
            in this case, sol_for_explicit will be \\phi and additional will be \\Bar(u) (both in n-1)
            f is the right hand side of the PDE.
            the part of time scheme is represented by doing the integration
            (u^{n+1} - u^n)/dt = coef_implicit * dx^2 u^{n+1} + coef_explicit * dx^2 u^n
            don't forget to interpolate f in time before calling function
            It is actually the same function as in FD_corr
        """
        M, h, D, Lambda = self.M_h_D_Lambda(upper_domain=upper_domain)
        a, c, _ = self.get_a_r_dt()
        if override_r is not None:
            c = override_r

        h, D = h[0], D[0]

        return coef_implicit * np.array([Lambda - D/h, D/h]) - \
               h/2 * np.array([(1/dt + coef_implicit* (c - a/h) ), coef_implicit*a/h]), \
               robin_cond - coef_explicit * np.dot(np.array([Lambda - D/h, D/h]), sol_for_explicit[:2]) \
               + h/2 * (-sol_unm1[0] / dt - f[0] +\
                       coef_explicit*(a*(sol_for_explicit[1] - sol_for_explicit[0])/h + c*sol_for_explicit[0]))


    # s_time_discrete is not a variable but an operator for Pade scheme
    def s_time_modif(self, w, order):
        s = w * 1j
        if order > 1:
            s += (4+3*np.sqrt(2))* (w*1j)**3 * self.DT**2/6 # warning: bugs with reaction
        return s

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
        a, c, dt = self.get_a_r_dt()
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

    def discretization_flux_interface(self, upper_domain):
        return None

    ############################### LEGACYYY ##################################
    def projection_result(self, result, upper_domain, partial_t_result0, f, result_explicit, **kwargs):
        """
            given the result of the inversion, returns (u_np1, u_interface, phi_interface)
        """
        M, h, D, Lambda = self.M_h_D_Lambda(upper_domain=upper_domain)
        a, c, dt = self.get_a_r_dt()
        h, D = h[0], D[0]
        phi = D*(result_explicit[1] - result_explicit[0])/h
        phi -= h/2 * (partial_t_result0 \
                + a * (result_explicit[1] - result_explicit[0]) / h
                + c * result_explicit[0] - f[0])

        return result, result_explicit[0], phi
