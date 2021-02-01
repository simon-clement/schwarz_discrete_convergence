"""
alpha is here dependent on the norm of (u_10-u0)
"""

import numpy as np
from discretizations.space.FD_bulk import FiniteDifferencesBulk
from discretizations.time.backward_euler import BackwardEuler
from utils_linalg import solve_linear
from utils_linalg import scal_multiply
from utils_linalg import multiply_interior
from utils_linalg import multiply
from utils_linalg import add_banded

class be_fd_bulk(BackwardEuler, FiniteDifferencesBulk):
    """
    Note: both boundary conditions are Dirichlet conditions
    The interface prescribes fluxes for both domains
    """

    def __init__(self, *args, alpha=1., theta=1., ratio_density=1., **kwargs):
        FiniteDifferencesBulk.__init__(self, *args, **kwargs)
        BackwardEuler.__init__(self, *args, **kwargs)
        self.ALPHA_NONLINEAR = alpha
        self.THETA = theta
        self.RHO_2_OVER_RHO_1 = ratio_density

    def projection_result(self, result, upper_domain, additional, f, result_explicit, **kwargs):
        """
            given the result of the inversion, returns (phi_np1, u_interface, phi_interface)
        """
        M, h, D, _ = self.M_h_D_Lambda(upper_domain=upper_domain)
        return result, additional[0], D[0]*result[0], additional

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
        M, h, D, _ = self.M_h_D_Lambda(upper_domain=upper_domain)
        a, c, _ = self.get_a_r_dt()

        if override_r is not None:
            c = override_r

        assert h[-1] > 0 and upper_domain or h[-1] < 0 and not upper_domain
        assert sol_for_explicit is not None
        assert additional is not None
        return D[-1] * dt / h[-1] * np.array([3/2, -2, 1/2]), bd_cond - 3/2 * additional[-1] + additional[-2] / 2 + dt / 2 * (f[-2] - 3*f[-1])

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

        if upper_domain:
            theta = self.THETA
            alpha = self.ALPHA_NONLINEAR
            return (D[0] * np.array([1+theta * self.DT*alpha/h[0], -theta * self.DT*alpha/h[0]]), robin_cond)
        else:
            return (D[0] * np.array([1]), robin_cond)

    def integrate_one_step(self, f, bd_cond, u_nm1, u_interface, phi_interface,
                           upper_domain, Y, additional,
                           selfu_interface, selfphi_interface, **kwargs):
        f = f(1)
        bd_cond = bd_cond(1)
        u_interface = u_interface[-1](1)
        phi_interface = phi_interface[-1](1)

        C_D = 1.2e-3
        self.ALPHA_NONLINEAR = C_D * np.abs(selfu_interface[-1](1) - u_interface)

        if len(additional) == 0:
            additional = self.create_additional(upper_domain=upper_domain)
        else:
            additional = additional[0]
        
        assert upper_domain is True or upper_domain is False

        M, h, D, Lambda = self.M_h_D_Lambda(upper_domain=upper_domain)
        self.LAMBDA_1 = 0.
        self.LAMBDA_2 = 0.

        A = self.A_interior(upper_domain=upper_domain)
        B = self.B_interior(upper_domain=upper_domain)
        rhs = multiply_interior(A, u_nm1) / self.DT + self.crop_f_as_prognostic(f=f, upper_domain=upper_domain)

        # old cond_robin = Lambda * u_interface + phi_interface
        if upper_domain:
            theta = self.THETA
            alpha = self.ALPHA_NONLINEAR
            cond_interface = alpha * (theta * additional[0] + (1-theta)*selfu_interface[-1](1) - u_interface)
        else:
            cond_interface = self.RHO_2_OVER_RHO_1 * phi_interface


        rhs = self.get_rhs(rhs=rhs, interface_cond=cond_interface, bd_cond=bd_cond,
                upper_domain=upper_domain, coef_explicit=0., coef_implicit=1.,
                dt=self.DT, f=f, sol_for_explicit=u_nm1, sol_unm1=u_nm1, additional=additional)

        if Y is None:
            Y = self.precompute_Y(upper_domain=upper_domain)

        result = solve_linear(Y,rhs)
        additional = self.update_additional(result=result, additional=additional, dt=self.DT,
                upper_domain=upper_domain, f=f, reaction_explicit=0, coef_reaction_implicit=1.)

        partial_t_result0 = (result[0] - u_nm1[0])/self.DT
        return self.projection_result(result=result, upper_domain=upper_domain, additional=additional, partial_t_result0=partial_t_result0, f=f, result_explicit=result)

    def convergence_rate(self, w):
        # _, h1, D1, _ = self.M_h_D_Lambda(upper_domain=False)
        _, h2, _, _ = self.M_h_D_Lambda(upper_domain=True)
        h = h2[0]

        z = np.exp(w * 1j * self.DT)
        s = self.s_time_discrete(w)
        lam1, lam2, _, _ = self.lambda_1_2_pm(s)
        # now we have D2*(1+implicit_part) * B_k = explicit_part * D2 * B_{k-1}
        # thus D2*(1+implicit_part) * A_k = explicit_part * A_{k-1}
        implicit_part = self.THETA*self.DT*self.ALPHA/h * (1-lam2) * z/ (z-1)
        explicit_part = self.ALPHA/h / s * \
                ((1 - self.THETA) * (lam2-1) + self.RHO_2_OVER_RHO_1*(lam1-1) )
        return explicit_part/((1+implicit_part))




