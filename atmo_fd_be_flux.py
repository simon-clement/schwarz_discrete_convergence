import numpy as np
from scipy.linalg import solve_banded

class AtmoFdBeFlux():
    def __init__(self, r, # reaction coefficient
                 nu, # Diffusivity
                 M, # Number of collocation points
                 SIZE_DOMAIN, # Size of \\Omega_1
                 DT): # Time step
        """
            The data needed is passed through this constructor.
            The space step is SIZE_DOMAIN / (M-1)
            there is M-1 points for u, M points for phi
        """
        self.r, self.nu, self.M, self.size_domain, self.dt = \
            r, nu, M, SIZE_DOMAIN, DT
        self.h = SIZE_DOMAIN / (M - 1)

    def integrate_in_time(self, phi_n, u_n, interface_info_next_time, forcing_next_time, alpha, theta, boundary):
        """
            Given the information, returns the tuple (phi_(n+1), u_(n+1)).

            Parameters:
            -phi_n (prognostic variable, space derivative of u_n), (size M)
            -u_n (diagnostic variable), (size M-1)
            -interface_info_next_time = alpha (theta u^{n-1}/(1+r dt) + (1-theta) u^{k-1} - u_ocean)
                    (where k is schwarz index, n is time index and if omitted, variables
                        are taken at iteration k and time n, and at interface level)
            -forcing_next_time the forcing in the diffusion-reaction equation

            boundary conditions:
                -Dirichlet=0 at the top of atmosphere
                -Robin: depending on theta:
                    nu phi_0 - theta dt alpha (phi_1 - phi_0)/h = interface_info_next_time
            
            scheme is:
            d phi / dt + r phi = nu/h^2 (phi_{j+1} - 2phi_j + phi_{j-1}) + (f{j+1/2} - f_{j-1/2})/h
            where d/dt uses a Backward-Euler scheme.
        """
        
        reaction_effect = 1/(1+self.r*self.dt)
        
        C = self.nu * self.dt / (self.h*self.h) # Courant Parabolic number
        # system is (1 + r dt + 2 C) * phi_j - C * (phi_{j-1}+phi_{j+1}) = phi_j^n + dt diff (f)/h
        diag_interior = 1 + self.r * self.dt + 2 * C
        diag_lower = - C
        diag_upper = - C

        ################# At interface (first row of matrix) ################
        # nu phi_0 + reaction_effect theta nu dt alpha (phi0 - phi_1)/h = interface_info_next_time
        cond_up_diag_interface =  - reaction_effect * theta*self.nu*self.dt*alpha / self.h
        cond_diag_interface = self.nu + reaction_effect * theta*self.nu*self.dt*alpha / self.h

        up_diag = np.concatenate(([1e100, cond_up_diag_interface],
            diag_upper * np.ones(self.M-2)))

        ################ At upper boundary (last row of matrix) #############
        #  nu dt / h (phi_{-1} - phi_{-2})^{n+1}
        #  = (1+ dt r) u_0 # (where u_0 = Dirichlet=dirichlet)
        #  - u_n[-1] - dt (f_{-1/2})

        diag = np.concatenate(([cond_diag_interface],
            diag_interior * np.ones(self.M - 2), [self.nu * self.dt / self.h]))

        low_diag = np.concatenate((diag_lower * np.ones(self.M-2),
            [-self.nu * self.dt / self.h, 1e100]))

        # For right hand side:
        # (1+ dt r) u_0 - 3 (u_(1/2) + u_(3/2))/2 + dt (f_(3/2) - 3 f_(1/2)) / 2 # (at height 0)
        # then inside the domain: phi_j^n + dt diff (f) / h
        # then at interface: interface_info_next_time
        rhs = np.concatenate(([interface_info_next_time],
                phi_n[1:-1] + self.dt * np.diff(forcing_next_time) / self.h,
                [(1+self.dt*self.r)*boundary - u_n[-1] - self.dt * forcing_next_time[-1]]))

        # solve banded, with 1 lower, 2 upper.
        phi_next = solve_banded(l_and_u=(1, 1),
                ab=np.vstack((up_diag, diag, low_diag)),
                overwrite_ab=True,
                b=rhs, overwrite_b=True)

        u_next = reaction_effect * (u_n + self.dt * (self.nu * np.diff(phi_next)/self.h + forcing_next_time))

        return phi_next, u_next
