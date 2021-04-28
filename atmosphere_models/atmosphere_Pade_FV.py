import numpy as np
from scipy.linalg import solve_banded

class AtmospherePadeFV():
    def __init__(self, r, # reaction coefficient
                 nu, # Diffusivity
                 M, # Number of collocation points
                 SIZE_DOMAIN, # Size of \\Omega_1
                 LAMBDA, # Robin parameter
                 DT): # Time step
        """
            The data needed is passed through this constructor.
            The space step is SIZE_DOMAIN / (M-1)
            there is M-1 points for u, M points for phi
        """
        self.r, self.nu, self.M, self.size_domain, self.Lambda, self.dt = \
            r, nu, M, SIZE_DOMAIN, LAMBDA, DT
        self.h = SIZE_DOMAIN / (M - 1)

    def size_u(self):
        return self.M - 1

    def interface_values(self, prognosed, diagnosed, overlap):
        u_interface = diagnosed[overlap] - self.h / 3 * prognosed[overlap] - self.h / 6 * prognosed[overlap+1]
        phi_interface = prognosed[overlap]
        return u_interface, phi_interface

    def integrate_in_time(self, prognosed, diagnosed, interface_robin, forcing, boundary):
        """
            Given the information, returns the tuple (phi_(n+1), u_(n+1)).

            Parameters:
            phi_n (prognostic variable, space derivative of solution), (size M)
            u_n (average of solution on each volume), (size M-1)
            interface_robin, forcing, boundary: tuples for time (tn, t*, t{n+1})

            forcing: the forcing in the diffusion-reaction equation, averaged on each volume
            boundary conditions :
                -Dirichlet(t)=boundary(t) at the top of atmosphere
                -Robin(t)=interface_robin(t) at interface

            scheme is:
                -Noting Y = 1+delta_x^2/6, b=1+1/sqrt(2), a=1+sqrt(2)
                -Noting R = r*dt, C = nu*dt/h^2

                    (Y + b (R Y - C delta_x^2))phi_star = (Y + a (R Y - C delta_x^2)) phi_n
                    (Y + b (R Y - C delta_x^2))phi_np1 = Y phi_star
        """
        
        phi_n = prognosed
        u_n = diagnosed
        Gamma = self.nu * self.dt / self.h**2
        R = self.r * self.dt
        a = 1+np.sqrt(2)
        b = 1+1/np.sqrt(2)
        h, dt, nu = self.h, self.dt, self.nu
        tilde_p = -self.Lambda
        f_n, f_star, f_np1 = forcing
        robin_n, robin_star, robin_np1 = interface_robin
        bd_n, bd_star, bd_np1 = boundary

        def prod_banded(tridiag_mat, x):
            ret = tridiag_mat[1]*x
            ret[:-1] += tridiag_mat[0,1:]*x[1:]
            ret[1:] += tridiag_mat[2,:-1]*x[:-1]
            return ret

        c_n = np.concatenate(([1/h*((robin_star - (1-(1-b)*R)*robin_n)/((1-b)*dt)-self.Lambda*f_n[0])],
                np.diff(f_n)/h, [1/h*((bd_star- (1-(1-b)*R)*bd_n)/((1-b)*dt) - f_n[-1])]))
        c_star = np.concatenate(([1/h*(((1+(1-b)*R)*robin_star - robin_n)/((1-b)*dt)-self.Lambda*f_star[0])],
                np.diff(f_star)/h, [1/h*(((1+(1-b)*R)*bd_star- bd_n)/((1-b)*dt) - f_star[-1])]))
        c_np1 = np.concatenate(([1/h*(((1+b*R)*robin_np1 - robin_star)/(b*dt)-self.Lambda*f_np1[0])],
                np.diff(f_np1)/h, [1/h*(((1+b*R)*bd_np1- bd_star)/(b*dt) - f_np1[-1])]))
        rhs_c_step1 = b*dt*c_star + (1-2*b)*dt*c_n
        rhs_c_step2 = b*dt*c_np1

        Y_FV = 1/6*np.vstack((np.concatenate(([1e100, tilde_p], np.ones(self.M-2))),#up_diag
            np.concatenate(([2*tilde_p + 6*nu/h], 4*np.ones(self.M-2), [2])), # diag
            np.concatenate((np.ones(self.M-1), [1e100]))))# low_diag
        D_FV = np.vstack((np.concatenate(([1e100,tilde_p], np.ones(self.M-2))),#up_diag
            np.concatenate(([-tilde_p], -2*np.ones(self.M-2), [-1])), # diag
            np.concatenate((np.ones(self.M-1), [1e100]))))# low_diag
        matrix_to_inverse = (1+b*R)*Y_FV - b*Gamma*D_FV

        rhs_step1 = prod_banded(Y_FV + (1-2*b)*(Gamma*D_FV - R*Y_FV), phi_n) + rhs_c_step1

        phi_star = solve_banded(l_and_u=(1, 1), ab=matrix_to_inverse, b=rhs_step1)

        rhs_step2 = prod_banded(Y_FV, phi_star) + rhs_c_step2

        phi_np1 = solve_banded(l_and_u=(1, 1), ab=matrix_to_inverse, b=rhs_step2)

        u_star = (u_n + b*(h*Gamma*np.diff(phi_star) + dt*f_star) \
                + a*(R*u_n - h*Gamma*np.diff(phi_n) - dt*f_n)) / (1 + b*R)
        u_np1 = (u_star + b*(h*Gamma*np.diff(phi_np1) + dt*f_np1)) / (1 + b*R)

        return phi_np1, u_np1

    """
        __eq__ and __hash__ are implemented, so that a discretization
        can be stored as key in a dict
        (it is useful for memoisation)
    """

    def repr(self):
        return "AtmPadeFV"

    def __eq__(self, other):
        return self.__dict__ == other.__dict__ and self.repr() == other.repr()

    def __hash__(self):
        return hash(repr(sorted(self.__dict__.items())) + self.repr())

    def __repr__(self):
        return repr(sorted(self.__dict__.items())) + self.repr()
