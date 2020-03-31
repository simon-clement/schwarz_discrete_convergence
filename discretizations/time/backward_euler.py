"""
    Finite differences for diffusion advection reaction equation.
    The functions to use are integrate_one_step and integrate_one_step_star.
    Theses function make a single step in time. It is not efficient
    (need to compute the matrix each time) but it is simple.
"""

import numpy as np
from discretizations.discretization import Discretization
from utils_linalg import solve_linear
from utils_linalg import scal_multiply
from utils_linalg import multiply_interior
from utils_linalg import multiply
from utils_linalg import add_banded


class BackwardEuler(Discretization):

    def __init__(self, *args, **kwargs):
        """
            The variables are initialized by space discretization.
        """
        pass


    def integrate_one_step(self, f, bd_cond, u_nm1, u_interface, phi_interface,
                           upper_domain=True,
                           Y=None, additional=[], **kwargs):

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

        # Pour le schéma B-E, on veut faire (A/dt-B) u^{np1} = A u^n /dt + f
        # Pour le schéma C-N, on veut faire (A/dt+B/2) u^{np1} = (A/dt + B/2) u^n + f

        to_inverse = add_banded(scal_multiply(A, 1/self.DT), scal_multiply(B, -1.))
        rhs = multiply_interior(A, u_nm1) / self.DT + f[1:-1] #Here f is cropped, but its extremal
        # value can serve in some space schemes in the interface conditions

        # extrapolation ? no! it is inside phi_interface
        cond_robin = Lambda * u_interface + phi_interface

        Y, rhs = self.add_boundaries(to_inverse=to_inverse, rhs=rhs, interface_cond=cond_robin,
                                     coef_explicit=0., coef_implicit=1., dt=self.DT, f=f, sol_unm1=u_nm1,
                                     sol_for_explicit=u_nm1,
                                     bd_cond=bd_cond, upper_domain=upper_domain, additional=additional)
        result = solve_linear(Y, rhs)
        self.update_additional(result=result, additional=additional, dt=self.DT,
                upper_domain=upper_domain, f=f, reaction_explicit=0, coef_reaction_implicit=1.)

        partial_t_result0 = (result[0] - u_nm1[0])/self.DT
        return self.projection_result(result=result, upper_domain=upper_domain, additional=additional, partial_t_result0=partial_t_result0, f=f)

    def s_time_discrete(self, w):
        assert w is not None # verifying the setting is not local in time
        assert "DT" in self.__dict__ # verifying we are in a valid time discretization
        z = np.exp(w * 1j * self.DT)
        return 1. / self.DT * (z - 1) / z

    def s_time_modif(self, w, order):
        """ Returns the modified s variable of the time scheme, with specified order"""
        assert order < float('inf')
        assert "DT" in self.__dict__ # verifying we are in a valid time discretization
        dt = self.DT
        s = w * 1j
        if order > 0:
            s += dt/2 * w**2
        if order > 1:
            s -= dt**2/6 * 1j * w**3
        if order > 2:
            s -= dt**3 / 24 * w**4
        if order > 3:
            s += dt**4/(120) * 1j * w**5
        return s
