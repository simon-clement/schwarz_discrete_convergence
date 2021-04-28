import numpy as np
from scipy.linalg import solve_banded

class OceanBEFV():
    def __init__(self, r, # reaction coefficient
                 nu, # Diffusivity
                 M, # Number of collocation points
                 SIZE_DOMAIN, # Size of \\Omega_1
                 LAMBDA,
                 DT): # Time step
        """
            The data needed is passed through this constructor.
            The space step is SIZE_DOMAIN / (M-1)
            there is M-1 points for u, M points for phi
        """
        self.r, self.nu, self.M, self.size_domain, self.Lambda, self.dt = \
            r, nu, M, SIZE_DOMAIN, LAMBDA, DT
        self.h = SIZE_DOMAIN / (M - 1)
        from cv_factor_onestep import rho_BE_FV
        self.discrete_rate = rho_BE_FV

    def size_u(self):
        return self.M - 1

    def interface_values(self, prognosed, diagnosed, overlap):
        u_interface = diagnosed[overlap] + self.h / 3 * prognosed[overlap] + self.h / 6 * prognosed[overlap+1]
        phi_interface = prognosed[overlap]
        return u_interface, phi_interface

    def integrate_in_time(self, prognosed, diagnosed, interface_robin, forcing, boundary):
        """
            Given the information, returns the tuple (phi_(n+1), u_(n+1)).

            Parameters:
            phi_n (prognostic variable, space derivative of solution), (size M)
            u_n (diagnosed average of solution on each volume), (size M-1)
            interface_robin, forcing, boundary: tuples for time (tn, t*, t{n+1})
            time t* is not used.

            forcing: the forcing in the diffusion-reaction equation, averaged on each volume
            boundary conditions :
                -Dirichlet(t)=boundary(t) at the bottom of ocean
                -Robin(t)=interface_robin(t) at interface

            scheme is:
                -Noting Y = 1+delta_x^2/6, b=1+1/sqrt(2), a=1+sqrt(2)
                -Noting R = r*dt, Gamma = nu*dt/h^2

                    (Y + (R Y - Gamma delta_x^2))phi_np1 = Y phi_n
                See pdf for the implementation.
        """
        
        phi_n = prognosed
        u_n = diagnosed
        Gamma = self.nu * self.dt / self.h**2
        R = self.r * self.dt
        h, dt, nu = self.h, self.dt, self.nu
        tilde_p = self.Lambda
        f_n, _, f_np1 = forcing
        robin_n, _, robin_np1 = interface_robin
        bd_n, _, bd_np1 = boundary

        def prod_banded(tridiag_mat, x):
            ret = tridiag_mat[1]*x
            ret[:-1] += tridiag_mat[0,1:]*x[1:]
            ret[1:] += tridiag_mat[2,:-1]*x[:-1]
            return ret

        rhs_c = np.concatenate((
            [1/h*((1+R)*robin_np1 - robin_n - dt*self.Lambda*f_np1[0])],
                        dt*np.diff(-f_np1)/h,
                        [1/h*(bd_n - (1+R)*bd_np1 + dt*f_np1[-1])]))

        Y_FV = 1/6*np.vstack((np.concatenate(([1e100, tilde_p], np.ones(self.M-2))),#up_diag
            np.concatenate(([2*tilde_p + 6*nu/h], 4*np.ones(self.M-2), [2])), # diag
            np.concatenate((np.ones(self.M-1), [1e100]))))# low_diag
        D_FV = np.vstack((np.concatenate(([1e100,tilde_p], np.ones(self.M-2))),#up_diag
            np.concatenate(([-tilde_p], -2*np.ones(self.M-2), [-1])), # diag
            np.concatenate((np.ones(self.M-1), [1e100]))))# low_diag
        matrix_to_inverse = (1+R)*Y_FV - Gamma*D_FV

        rhs_step = prod_banded(Y_FV, phi_n) + rhs_c

        phi_np1 = solve_banded(l_and_u=(1, 1), ab=matrix_to_inverse, b=rhs_step)

        # we use diff(-phi) because we are reversed
        u_np1 = (u_n + (h*Gamma*np.diff(-phi_np1) + dt*f_np1)) / (1 + R)

        return phi_np1, u_np1

    """
        __eq__ and __hash__ are implemented, so that a discretization
        can be stored as key in a dict
        (it is useful for memoisation)
    """

    def repr(self):
        return "OceBEFV"

    def __eq__(self, other):
        return self.__dict__ == other.__dict__ and self.repr() == other.repr()

    def __hash__(self):
        return hash(repr(sorted(self.__dict__.items())) + self.repr())

    def __repr__(self):
        return repr(sorted(self.__dict__.items())) + self.repr()
