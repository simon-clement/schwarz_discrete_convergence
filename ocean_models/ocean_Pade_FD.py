import numpy as np
from scipy.linalg import solve_banded

class OceanPadeFD():
    def __init__(self, r: float, # reaction coefficient
            nu: float, # Diffusivity
            M: int, # Number of collocation points
            SIZE_DOMAIN: float, # Size of \\Omega_1
            LAMBDA: float,
            DT: float, k_c: int=0): # Time step
        """
            The data needed is passed through this constructor.
            The space step is SIZE_DOMAIN / (M-1)
            there is M-1 points for u, M points for phi
        """
        self.r, self.nu, self.M, self.size_domain, self.Lambda, self.dt = \
            r, nu, M, SIZE_DOMAIN, LAMBDA, DT
        assert abs(k_c)<1e-10 or abs(k_c - 1) < 1e-10
        self.k_c: int = 0 if abs(k_c) < 1e-10 else 1
        self.h: float = SIZE_DOMAIN / (M - 1)
        from cv_factor_pade import rho_Pade_FD_corr0
        self.discrete_rate = rho_Pade_FD_corr0
        if self.k_c == 1:
            print("warning: discrete rate available")
            from cv_factor_pade import rho_Pade_FD_corr1
            self.discrete_rate = rho_Pade_FD_corr1

    def size_u(self):
        return self.M

    def interface_values(self, prognosed, diagnosed):
        u_interface = prognosed[0]
        phi_interface = diagnosed[0]
        return u_interface, phi_interface

    def integrate_in_time(self, prognosed, diagnosed, interface_robin, forcing, boundary):
        """
            Given the information, returns the tuple (phi_(n+1), u_(n+1)).

            Parameters:
            prognosed=u_n (diagnosed variable, space derivative of solution), (size M)
            diagnosed=phi_n (diagnosed average of solution on each volume), (size M-1)
            interface_robin, forcing, boundary: tuples for time (tn, t*, t{n+1})

            forcing: the forcing in the diffusion-reaction equation, averaged on each volume
            boundary conditions :
                -Dirichlet(t)=boundary(t) at the bottom of ocean
                -Robin(t)=interface_robin(t) at interface

            scheme is:
                -Noting Y = 1+delta_x^2/6, b=1+1/sqrt(2), a=1+sqrt(2)
                -Noting R = r*dt, Gamma = nu*dt/h^2

                    (Y + b (R Y - Gamma delta_x^2))phi_star = (Y + a (R Y - Gamma delta_x^2)) phi_n
                    (Y + b (R Y - Gamma delta_x^2))phi_np1 = Y phi_star
                See pdf for the implementation.
        """
        
        u_n = prognosed
        Gamma = self.nu * self.dt / self.h**2
        R = self.r * self.dt
        b = 1+1/np.sqrt(2)
        h, dt, nu = self.h, self.dt, self.nu
        tilde_p = self.Lambda
        f_n, f_star, f_np1 = forcing
        robin_n, robin_star, robin_np1 = interface_robin
        bd_n, bd_star, bd_np1 = boundary

        def prod_banded(tridiag_mat, x):
            ret = tridiag_mat[1]*x
            ret[:-1] += tridiag_mat[0,1:]*x[1:]
            ret[1:] += tridiag_mat[2,:-1]*x[:-1]
            return ret

        c_n = np.concatenate(([[robin_n], f_n[1:-1], [bd_n]]))
        c_star = np.concatenate(([[robin_star], f_star[1:-1], [bd_star]]))
        c_np1 = np.concatenate(([[robin_np1], f_np1[1:-1], [bd_np1]]))
        rhs_c_step1 = b*dt*c_star + (1-2*b)*dt*c_n
        rhs_c_step2 = b*dt*c_np1

        Y_FD = np.vstack((np.concatenate(([1e100], np.zeros(self.M-1))),#up_diag
            np.concatenate(([h/2*self.k_c], np.ones(self.M-2), [0])), # diag
            np.concatenate((np.zeros(self.M-1), [1e100]))))# low_diag
        D_FD = np.vstack((np.concatenate(([1e100,h], np.ones(self.M-2))),#up_diag
            np.concatenate(([-h-h**2/nu*tilde_p], -2*np.ones(self.M-2), [-h**2/nu])), # diag
            np.concatenate((np.ones(self.M-2), [0, 1e100]))))# low_diag

        matrix_to_inverse = (1+b*R)*Y_FD - b*Gamma*D_FD

        rhs_step1 = prod_banded(Y_FD + (1-2*b)*(Gamma*D_FD - R*Y_FD), u_n) + rhs_c_step1

        u_star = solve_banded(l_and_u=(1, 1), ab=matrix_to_inverse, b=rhs_step1)

        rhs_step2 = prod_banded(Y_FD, u_star) + rhs_c_step2

        u_np1 = solve_banded(l_and_u=(1, 1), ab=matrix_to_inverse, b=rhs_step2)
        phi_np1 = np.diff(-u_np1)/h

        #slight modification of phi[0] if corrective term:
        derivative_u0 = ((1+R)*u_np1[0] - u_star[0])/(b*dt)
        phi_np1[0] += self.h/2 *self.k_c/nu * derivative_u0

        return u_np1, phi_np1

    """
        __eq__ and __hash__ are implemented, so that a discretization
        can be stored as key in a dict
        (it is useful for memoisation)
    """

    def repr(self):
        return "OcePadeV"

    def __eq__(self, other):
        return self.__dict__ == other.__dict__ and self.repr() == other.repr()

    def __hash__(self):
        return hash(repr(sorted(self.__dict__.items())) + self.repr())

    def __repr__(self):
        return repr(sorted(self.__dict__.items())) + self.repr()
