"""
    Finite volume for diffusion advection reaction equation.
    The functions to use are integrate_one_step and integrate_one_step_star.
    Theses function make a single step in time. It is not efficient
    (need to compute the matrix each time) but it is simple.
"""

import numpy as np
from utils_linalg import solve_linear
from discretizations.discretization import Discretization
from utils_linalg import solve_linear
from utils_linalg import scal_multiply
from utils_linalg import multiply_interior
from utils_linalg import multiply
from utils_linalg import add_banded


class RK2(Discretization):

    def __init__(self, *args, **kwargs):
        """
            give default values of all variables.
        """
        pass

    def integrate_one_step(self, f, bd_cond, u_nm1, u_interface,
            phi_interface, upper_domain=True, Y=None, additional=[], **kwargs):

        f_nm1 = f(0)
        f_nm1_2 = f(.5)
        f = f(1)
        bd_cond_nm1 = bd_cond(0)
        bd_cond_nm1_2 = bd_cond(0.5)
        bd_cond = bd_cond(1)
        u_nm1_interface = u_interface(0)
        u_nm1_2_interface = u_interface(0.5)
        u_interface = u_interface(1)
        phi_nm1_interface = phi_interface(0)
        phi_nm1_2_interface = phi_interface(0.5)
        phi_interface = phi_interface(1)
        if len(additional) == 0:
            additional = self.create_additional(upper_domain=upper_domain)
        else:
            additional = additional[0]
        
        assert upper_domain is True or upper_domain is False

        M, h, D, Lambda = self.M_h_D_Lambda(upper_domain=upper_domain)

        # L'idée c'est qu'en semi-discret en espace, on a: A\partial_t u = B u
        A = self.A_interior(upper_domain=upper_domain)
        B = self.B_interior(upper_domain=upper_domain)
        assert len(A) == 3 # tridiagonal matrix
        assert len(B) == 3 # tridiagonal matrix

        ###################
        # FIRST STEP : find the time derivative at middle step
        ###################
        to_inverse = scal_multiply(A, 2/self.DT)
        rhs = multiply_interior(A, u_nm1) * 2 / self.DT +\
                multiply_interior(B, u_nm1) + self.crop_f_as_prognostic(f_nm1, upper_domain=upper_domain)
        # Here f is cropped, but its extremal
        # values can serve in some space schemes in the interface conditions

        # extrapolation ? no! it is inside phi_interface
        cond_robin_nm1_2 = Lambda * u_nm1_2_interface + phi_nm1_2_interface

        Y, rhs = self.add_boundaries(to_inverse=to_inverse, rhs=rhs, interface_cond=cond_robin_nm1_2,
                                     coef_explicit=1., coef_implicit=0.,
                                     dt=self.DT/2, f=f_nm1, sol_for_explicit=u_nm1, sol_unm1=u_nm1,
                                     bd_cond=bd_cond_nm1_2, upper_domain=upper_domain,
                                     additional=additional)
        result_nm1_2 = solve_linear(Y, rhs)

        ###################
        # SECOND STEP : time derivative at middle step is A^{-1} (B u_nm1_2 + f), let's use it
        ###################

        to_inverse = scal_multiply(A, 1/self.DT)
        rhs = multiply_interior(A, u_nm1) / self.DT +\
                multiply_interior(B, result_nm1_2) + self.crop_f_as_prognostic(f_nm1_2, upper_domain=upper_domain)

        cond_robin = Lambda * u_interface + phi_interface
    
        Y, rhs = self.add_boundaries(to_inverse=to_inverse, rhs=rhs, interface_cond=cond_robin,
                                     coef_explicit=1., coef_implicit=0.,
                                     dt=self.DT, f=f, sol_for_explicit=result_nm1_2,
                                     sol_unm1=u_nm1,
                                     bd_cond=bd_cond, upper_domain=upper_domain,
                                     additional=additional)
        result = solve_linear(Y, rhs)

        #Because of the reaction term, the RK scheme needs to be also developped on additional
        additional_nm1_2 = self.update_additional(result=u_nm1, additional=additional, dt=self.DT/2,
                upper_domain=upper_domain, f=f_nm1, reaction_explicit=additional, coef_reaction_implicit=0) # Now additional_nm1_2 is in time n-1/2

        additional = self.update_additional(result=result_nm1_2, additional=additional, dt=self.DT,
                upper_domain=upper_domain, f=f_nm1_2, reaction_explicit=additional_nm1_2, coef_reaction_implicit=0) # Now additional is in time n
        additional = self.new_additional(result=result, upper_domain=upper_domain,
                cond=bd_cond if upper_domain else cond_robin)

        partial_t_result0 = (result[0] - u_nm1[0])/self.DT
        return self.projection_result(result=result, upper_domain=upper_domain,
                additional=additional, partial_t_result0=partial_t_result0, f=f, result_explicit=result_nm1_2)

    # s_time_discrete is not a variable but an operator for RK2
    def s_time_modif(self, w, order):
        s = w * 1j
        if order > 1:
            s -= w**3 * self.DT**2/6 * 1j
        return s

