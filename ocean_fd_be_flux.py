import numpy as np
from scipy.linalg import solve_banded

class OceanFdBeFlux():
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

    def integrate_in_time(self, phi_n, u_n, interface_flux_next_time, forcing_next_time, boundary):
        """
            Given the information, returns the tuple (phi_(n+1), u_(n+1)).

            Parameters:
            phi_n (prognostic variable, space derivative of u_n), (size M)
            u_n (diagnostic variable), (size M-1)
            forcing_next_time the forcing in the diffusion-reaction equation
            boundary conditions :
                -Dirichlet=boundary at the bottom of ocean
                -Neumann: interface_flux_next_time at interface

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
        # At interface (last row of matrix) we have nu*phi_j = interface_flux_next_time.

        # first we give lower: C on all the interior, and at the extremity (interface), 0.
        # note the 1e100 which corresponds to the "*" in the doc of solve_banded
        low_diag = np.concatenate((diag_lower * np.ones(self.M - 2), [0., 1e100]))
        ################ At the bottom (first row of matrix) #############
        # nu dt / h (phi_1 - phi_0)
        # = (1 + dt r) u_0 # (where u_0 = Dirichlet = boundary)
        # - u_n[0] - dt f_(1/2)

        diag = np.concatenate(( [-self.nu * self.dt / self.h],
            diag_interior * np.ones(self.M - 2), [self.nu]))
        up_diag = np.concatenate(( [1e100, self.nu * self.dt / self.h],
            diag_upper * np.ones(self.M-2)))

        # For right hand side:
        # (1 + dt r)u_0 - dt f_(1/2) # (at height 0)
        # then inside the domain: phi_j^n + dt diff (f) / h
        # then at interface: interface_flux_next_time
        rhs = np.concatenate(([(1+self.r*self.dt)*boundary - u_n[0] - self.dt * forcing_next_time[0]],
                phi_n[1:-1] + self.dt * np.diff(forcing_next_time) / self.h,
                [interface_flux_next_time]))

        # solve banded, with 1 lower, 2 upper.
        phi_next = solve_banded(l_and_u=(1, 1),
                ab=np.vstack((up_diag, diag, low_diag)),
                overwrite_ab=False,
                b=rhs, overwrite_b=False)

        u_next = reaction_effect * (u_n + self.dt * (self.nu * np.diff(phi_next)/self.h + forcing_next_time))
        return phi_next, u_next
