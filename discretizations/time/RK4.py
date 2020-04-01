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


class RK4(Discretization):

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
        if additional is None:
            additional_nm1_2 = None
        else:
            additional_nm1_2 = additional.copy()
        
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
                multiply_interior(B, u_nm1) + f_nm1[1:-1]
        # Here f is cropped, but its extremal
        # values can serve in some space schemes in the interface conditions

        # extrapolation ? no! it is inside phi_interface
        cond_robin_nm1_2 = Lambda * u_nm1_2_interface + phi_nm1_2_interface

        Y, rhs = self.add_boundaries(to_inverse=to_inverse, rhs=rhs, interface_cond=cond_robin_nm1_2,
                                     coef_explicit=1., coef_implicit=0.,
                                     dt=self.DT/2, f=f_nm1, sol_for_explicit=u_nm1, sol_unm1=u_nm1,
                                     bd_cond=bd_cond_nm1_2, upper_domain=upper_domain,
                                     additional=additional)
        result_nm1_2_k1 = solve_linear(Y, rhs)

        ###################
        # SECOND STEP : computing u+k2/2. almost the same but the rhs is with u+k1/2 (i.e. result_nm1_2_k1)
        ###################
        to_inverse = scal_multiply(A, 2/self.DT)
        rhs = multiply_interior(A, u_nm1) * 2 / self.DT +\
                multiply_interior(B, result_nm1_2_k1) + f_nm1_2[1:-1]
        # Here f is cropped, but its extremal
        # values can serve in some space schemes in the interface conditions

        # extrapolation ? no! it is inside phi_interface
        cond_robin_nm1_2 = Lambda * u_nm1_2_interface + phi_nm1_2_interface

        Y, rhs = self.add_boundaries(to_inverse=to_inverse, rhs=rhs, interface_cond=cond_robin_nm1_2,
                                     bd_cond=bd_cond_nm1_2, upper_domain=upper_domain,
                                     coef_explicit=1., coef_implicit=0.,
                                     dt=self.DT/2, f=f_nm1_2, sol_for_explicit=result_nm1_2_k1,
                                     sol_unm1=u_nm1,
                                     additional=additional)
        result_nm1_2_k2 = solve_linear(Y, rhs)

        ###################
        # THIRD STEP : computing u+k3 with u+k2/2 (i.e. result_nm1_2_k2)
        ###################
        to_inverse = scal_multiply(A, 1/self.DT)
        rhs = multiply_interior(A, u_nm1) / self.DT +\
                multiply_interior(B, result_nm1_2_k2) + f[1:-1]
        # Here f is cropped, but its extremal
        # values can serve in some space schemes in the interface conditions

        # extrapolation ? no! it is inside phi_interface
        cond_robin = Lambda * u_interface + phi_interface

        Y, rhs = self.add_boundaries(to_inverse=to_inverse, rhs=rhs, interface_cond=cond_robin,
                                     bd_cond=bd_cond, upper_domain=upper_domain,
                                     coef_explicit=1., coef_implicit=0.,
                                     dt=self.DT, f=f, sol_for_explicit=result_nm1_2_k2,
                                     sol_unm1=u_nm1, additional=additional)
        result_k3 = solve_linear(Y, rhs)

        #####################
        # LAST STEP : weighted average of k's
        #####################
        result_averaged = (u_nm1 + 2*result_nm1_2_k1 + 2*result_nm1_2_k2 + result_k3)/6
        f_averaged = (f_nm1 + 4*f_nm1_2 + f)/6
        to_inverse = scal_multiply(A, 1/self.DT)
        rhs = multiply_interior(A, u_nm1) / self.DT +\
                multiply_interior(B, result_averaged) + f_averaged[1:-1]

        cond_robin = Lambda * u_interface + phi_interface

        Y, rhs = self.add_boundaries(to_inverse=to_inverse, rhs=rhs, interface_cond=cond_robin,
                                     bd_cond=bd_cond, upper_domain=upper_domain,
                                     coef_explicit=1., coef_implicit=0.,
                                     dt=self.DT, f=f_averaged, sol_for_explicit=result_averaged,
                                     sol_unm1=u_nm1, additional=additional)
        result = solve_linear(Y, rhs)

        #Because of the reaction term, the RK scheme needs to be also developped on additional
        adpk1 = self.update_additional(result=u_nm1, additional=additional, dt=self.DT/2,
                upper_domain=upper_domain, f=f_nm1_2, reaction_explicit=additional, coef_reaction_implicit=0)
        adpk2 = self.update_additional(result=result_nm1_2_k1, additional=additional, dt=self.DT/2,
                upper_domain=upper_domain, f=f_nm1_2, reaction_explicit=adpk1, coef_reaction_implicit=0)
        adpk3 = self.update_additional(result=result_nm1_2_k2, additional=additional, dt=self.DT,
                upper_domain=upper_domain, f=f_nm1_2, reaction_explicit=adpk2, coef_reaction_implicit=0)
        # Since reaction is linear, we can sum right now additional variables
        additional_averaged = None if additional is None else (additional + 2*adpk1 + 2*adpk2 + adpk3)/6

        additional = self.update_additional(result=result_averaged, additional=additional, dt=self.DT,
                        upper_domain=upper_domain, f=f_averaged, reaction_explicit=additional_averaged,
                        coef_reaction_implicit=0) # Now additional is in time n


        partial_t_result0 = (result[0] - u_nm1[0])/self.DT
        return self.projection_result(result=result, upper_domain=upper_domain,
                additional=additional, partial_t_result0=partial_t_result0, f=f,
                result_explicit=result_averaged)

    # s_time_discrete is not a variable but an operator for RK4
    def s_time_modif(self, w, order):
        s = w * 1j
        return s

