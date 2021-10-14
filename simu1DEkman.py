import numpy as np

array = np.ndarray

class Simu1dEkman():
    def __init__(self, z_levels: array, h_cl: float, dt: float=1.,
            u_geostrophy: float=10., K_mol: float=1e-7,
            C_D: float=1e-3, f: float=1e-4) -> None:
        """
            z_levels starts at 0 and contains all the full levels $z_m$.
            h_cl is the height of the planetary boundary layer;
            dt is the time step
            u_geostrophy is used as a top boundary condition
            K_mol is the background diffusivity
            C_D is the drag coefficient used in the wall law
            f is the Coriolis parameter
        """
        assert z_levels.shape[0] > 2 and np.abs(z_levels[0]) < 1e-3

        # X_full[m] represents in LaTeX $X_m$ whereas
        # X_half[m] represents in LaTeX $X_{m+1/2}$.
        # both have M+1 points (0<=m<=M).
        self.z_full: array = np.copy(z_levels)
        self.h_half: array = np.diff(self.z_full,
                append=2*self.z_full[-1] - self.z_full[-2])
        self.z_half: array = self.z_full + self.h_half/2
        self.h_full: array = np.diff(self.z_half, prepend=-self.z_half[0])
        self.M: int = self.z_full.shape[0] - 1
        self.h_cl: float = h_cl
        self.dt: float = dt
        self.K_mol: float = K_mol
        self.kappa: float = 0.4
        self.u_g: float = u_geostrophy
        self.f: float = f
        self.C_D: float = C_D


    def FV_KPP(self, u_t0: array, phi_t0: array, u_firstlevel: array,
            forcing: array, time_method: str="BE", sf_scheme: str="FD",
            z_star: float=1e-4) -> array:
        """
            Integrates in time with Backward Euler the model with KPP
            and Finite differences.

            u_t0 should be given at half-levels (follow self.z_half)
            phi_t0 should be given at full-levels (follow self.z_full)
            u_firstlevel should start at t=0 and finish at N*dt
                                    where N is the number of time step
            forcing should be given as averaged on each volume for all times
            method support only BE for now
            sf_scheme is the surface flux scheme:
                - "FD" for a FD interpretation (classical but biaised)
                - "FV" for a FV interpretation (better)
                - "FV free" for a FV interpretation with a free \delta_{cfl}
                    (not implemented)
            returns a numpy array of shape (M)
                                    where M is the number of space points 
        """
        assert u_t0.shape[0] == self.M and phi_t0.shape[0] == self.M + 1
        N: int = u_firstlevel.shape[0] - 1 # number of time steps
        assert forcing.shape == (N + 1, self.M)

        # K_full = u_star * G_full + K_mol.
        G_full: array = self.kappa * self.z_full * \
                (1 - self.z_full/self.h_cl)**2 \
                * np.heaviside(1 - self.z_full/self.h_cl, 1)
        u_current: array = np.copy(u_t0)
        phi_current: array = np.copy(phi_t0)
        for n in range(1,N+1):
            u_flevel_current, forcing_current = u_firstlevel[n], forcing[n]

            u_star: float = np.sqrt(self.C_D) * np.abs(u_flevel_current)
            K_full: array = u_star * G_full + self.K_mol # viscosity

            Y_diag: array = np.concatenate(([0], 2/3*np.ones(self.M-1), [1/3]))
            Y_udiag: array = np.concatenate(([0],
                [self.h_half[m]/6./self.h_full[m] for m in range(1, self.M)]))
            Y_ldiag: array =  np.concatenate((
                [self.h_half[m-1]/6./self.h_full[m] for m in range(1, self.M)],
                [1/6.]))

            D_diag: array = np.concatenate(([K_full[0]],
                [-2 * K_full[m] / self.h_half[m] / self.h_half[m-1]
                                            for m in range(1, self.M)],
                [-K_full[self.M] / self.h_half[self.M - 1]**2]))
            D_udiag: array = np.concatenate(([0],
                [K_full[m+1]/self.h_full[m] / self.h_half[m]
                                            for m in range(1, self.M)]))
            D_ldiag: array = np.concatenate((
                [K_full[m-1]/self.h_full[m] / self.h_half[m-1]
                                            for m in range(1, self.M)],
                [K_full[self.M-1] / self.h_half[self.M - 1]**2]))

            if sf_scheme == "FD":
                Y_diag[0] = Y_udiag[0] = D_udiag[0] = 0.
                D_diag[0] = K_full[0]
            elif sf_scheme == "FV":
                Y_diag[0] = Y_udiag[0] = Y_ldiag[0] = Y_diag[1] = \
                        Y_udiag[1] = D_udiag[0] = D_ldiag[0] = D_udiag[1] = 0.
                D_diag[0], D_diag[1] = K_full[0], K_full[1]
            else:
                raise NotImplementedError(sf_scheme + " surface flux scheme unknown")

            if time_method == "BE":
                c_I1: float
                c_I2: float
                if sf_scheme == "FD":
                    c_I1 = self.C_D * np.abs(u_flevel_current)*u_flevel_current
                    c_I2 = 1/self.h_full[1] * (forcing_current[1] - forcing_current[0])
                elif sf_scheme == "FV":
                    u_star = self.kappa * u_flevel_current / \
                            ((1+z_star / self.z_full[1]) * \
                            np.log(1+self.z_full[1] / z_star) - 1)
                    u_cfl = u_star / self.kappa * np.log(1+ self.z_full[1]/ z_star)
                    c_I1 = c_I2 = u_star**2 * u_cfl / np.abs(u_cfl)
                else:
                    raise NotImplementedError(sf_scheme + " surface flux scheme unknown")

                c_T = 1j*self.f * self.u_g - forcing_current[self.M-1] / self.h_half[self.M-1]
                c = np.concatenate(([c_I1, c_I2], np.diff(forcing_current[1:]) / \
                                                    self.h_full[2:-1], [c_T]))
                phi_current = self.__backward_euler(
                        Y=(np.zeros(self.M), Y_diag, np.zeros(self.M)),
                        D=(D_ldiag, D_diag, D_udiag), c=c, u=phi_current)
                u_current = 1/(1+self.dt*1j*self.f) * (u_current + self.dt * \
                        (np.diff(phi_current * K_full / self.h_half) \
                        + forcing_current))

            else:
                raise NotImplementedError("Integration method not supported")

        return u_current



    def FD_KPP(self, u_t0: array, u_firstlevel: array,
            forcing: array, method: str="BE",
            z_star: float=1e-4) -> array:
        """
            Integrates in time with Backward Euler the model with KPP
            and Finite differences.

            u_t0 should be given at half-levels (follow self.z_half)
            u_firstlevel should start at t=0 and finish at N*dt
                                    where N is the number of time step
            forcing should be given at half-levels for all times
            method support only BE for now
            returns a numpy array of shape (M) 
                                    where M is the number of space points 
        """
        assert u_t0.shape[0] == self.M + 1
        N: int = u_firstlevel.shape[0] - 1 # number of time steps
        assert forcing.shape == (N + 1, self.M + 1)

        # K_full = u_star * G_full + K_mol.
        G_full: array = self.kappa * self.z_full * \
                (1 - self.z_full/self.h_cl)**2 \
                * np.heaviside(1 - self.z_full/self.h_cl, 1)
        Y_diag: array = np.concatenate((np.ones(self.M), [0]))
        u_current: array = np.copy(u_t0)
        for n in range(1,N+1):
            u_flevel_current, forcing_current = u_firstlevel[n], forcing[n]

            u_star: float = np.sqrt(self.C_D) * np.abs(u_flevel_current)
            K_full: array = u_star * G_full + self.K_mol # viscosity
            D_diag: array = np.concatenate((
                [-K_full[1] / (self.h_half[0] * self.h_full[1])],
                [(-K_full[m+1]/self.h_full[m+1] - K_full[m]/self.h_full[m]) \
                        / self.h_half[m] for m in range(1, self.M)],
                [-1]))
            D_udiag: array = np.concatenate((
                [K_full[1]/self.h_half[0]/self.h_full[1]],
                [K_full[m+1]/self.h_full[m+1] / self.h_half[m]
                    for m in range(1, self.M)]))
            D_ldiag: array = np.concatenate((
                [K_full[m]/self.h_full[m] / self.h_half[m]
                    for m in range(1, self.M)]
                ,[0]))

            c_I: float = forcing_current[0] - \
                    self.C_D * np.abs(u_flevel_current) \
                    * u_flevel_current / self.h_half[0]
            c: array = np.concatenate(([c_I],
                forcing_current[1:-1], [self.u_g]))

            if method == "BE":
                u_current = self.__backward_euler(
                        Y=(np.zeros(self.M), Y_diag, np.zeros(self.M)),
                        D=(D_ldiag, D_diag, D_udiag), c=c, u=u_current)
            else:
                raise NotImplementedError("Integration method not supported")

        return u_current

    def __backward_euler(self, Y: tuple[array, array, array],
                        D: tuple[array, array, array], c: array, u: array):
        """
            integrates once (self.dt) in time the equation
            (\partial_t + if)Yu - dt*Du = c
            The scheme is:
            Y(1+dt*if) - D) u_np1 = Y u + dt*c
        """
        from utils_linalg import multiply, scal_multiply as s_mult
        from utils_linalg import add_banded as add
        from utils_linalg import multiply, solve_linear
        to_inverse: tuple[array] = add(s_mult(Y, 1 + self.dt * 1j*self.f),
                                        s_mult(D, - self.dt))
        return solve_linear(to_inverse, multiply(Y, u) + self.dt*c)

