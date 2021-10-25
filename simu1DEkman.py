"""
    This module defines the class Simu1dEkman
    which simulates a 1D Ekman problem with various
    discretisations.
"""

from typing import Tuple, List
import numpy as np
from utils_linalg import multiply, scal_multiply as s_mult
from utils_linalg import add_banded as add
from utils_linalg import solve_linear

array = np.ndarray

class Simu1dEkman():
    """
        main class, instanciate it to run the simulations.
    """
    def __init__(self, z_levels: array, dt: float=1.,
            u_geostrophy: float=10., default_h_cl: float=1e4, K_mol: float=1e-7,
            C_D: float=1e-3, f: float=1e-4) -> None:
        """
            z_levels starts at 0 and contains all the full levels $z_m$.
            h_cl is the default height of the planetary boundary layer if f=0;
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
        self.defaulth_cl: float = default_h_cl
        self.dt: float = dt
        self.K_mol: float = K_mol
        self.kappa: float = 0.4
        self.u_g: float = u_geostrophy
        self.f: float = f
        self.C_D: float = C_D

    def reconstruct_FV(self, u_bar: array, phi: array,
            FV_free=False, delta_sl=0., z_star=1e-1):
        """
            spline reconstruction of the FV solution.
            u(xi) = u_bar[m] + (phi[m+1] + phi[m]) * xi/2 +
                    (phi[m+1} - phi[m]) / (2*h_half[m]) * 
                        (xi^2 - h_half[m]^2/12)
            where xi = linspace(-h_half[m]/2, h_half[m]/2, 10, endpoint=True)
        """
        h_half, z_half = np.copy(self.h_half), np.copy(self.z_half)
        if FV_free:
            h_half[0] = self.z_full[1] - delta_sl
            z_half[0] = self.z_full[1] - h_half[0]/2
            tau_sl = 1 + z_star / delta_sl - 1/np.log(1+delta_sl/z_star)
            u_bar = np.copy(u_bar) # putting u_tilde instead of u_bar[0]:
            u_bar[0] = (u_bar[0] + h_half[0] * tau_sl * \
                    (phi[0]/3 + phi[1]/6)) / (1 + tau_sl)
        xi = [np.linspace(-h/2, h/2, 10) for h in h_half[:-1]]
        sub_discrete: List[array] = [u_bar[m] + (phi[m+1] + phi[m]) * xi[m]/2 \
                + (phi[m+1] - phi[m]) / (2 * h_half[m]) * \
                (xi[m]**2 - h_half[m]**2/12) for m in range(self.M)]
        if FV_free:
            u_oversampled = np.concatenate(sub_discrete)
            z_oversampled = np.concatenate([np.array(xi[m]) + z_half[m]
                                                for m in range(self.M)])
            # xi0 = np.linspace(1e-3,delta_sl, 10)
            # z_oversampled = np.concatenate(( xi0, z_oversampled))
            # direction = u_oversampled[0] / np.abs(u_oversampled[0])
            # ustar = np.abs(u_oversampled[0]*self.kappa) / \
            #         np.log(1 + delta_sl / z_star)
            # u_oversampled = np.concatenate((
            #     direction * ustar / self.kappa * \
            #             np.log(1 + xi0 / z_star), u_oversampled))
        else:
            u_oversampled = np.concatenate(sub_discrete[1:])
            z_oversampled = np.concatenate([np.array(xi[m]) + z_half[m]
                                                for m in range(1, self.M)])

        return z_oversampled, u_oversampled


    def FV_KPP(self, u_t0: array, phi_t0: array,
            forcing: array, time_method: str="BE", sf_scheme: str="FD",
            z_star: float=1e-1, delta_sl: float=None) -> array:
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
                - "FV{1, 2, 3}" for a FV interpretation (unbiaised but delta_{sl}=z_1)
                - "FV free" for a FV interpretation with a free delta_{sl}
                    (not implemented)
            returns a numpy array of shape (M)
                                    where M is the number of space points
        """
        assert u_t0.shape[0] == self.M and phi_t0.shape[0] == self.M + 1

        N: int = forcing.shape[0] - 1 # number of time steps
        assert forcing.shape[1] == self.M
        c_1 = 0.2 # constant for h_cl in atmosphere
        if delta_sl is None:
            delta_sl = self.z_half[0] if sf_scheme in {"FD"} else self.z_full[1]

        phi_current: array = np.copy(phi_t0)

        # computing u_current for the first time
        rhs_ucur = self.h_half[0:-2] * phi_current[0:-2]/6 + \
                (self.h_half[0:-2] + self.h_half[1:-1])*phi_current[1:-1]/3 + \
                self.h_half[1:-1] * phi_current[2:] / 6
        rhs_ucur = np.concatenate((rhs_ucur,
            [self.h_half[-2] * (phi_current[-2]/6 + phi_current[-1]/3) \
                        - self.u_g]))
        to_solve = -np.ones(self.M), np.ones(self.M-1)
        if sf_scheme == "FV free":
            # getting overline{u}_{1/2} is particular:
            tilde_h: float = self.z_full[1] - delta_sl
            tau_sl = 1 + z_star/delta_sl - 1/np.log(1+delta_sl/z_star)
            rhs_ucur[0] = tilde_h *(1+3*tau_sl)/(1+tau_sl)* \
                    phi_current[0]/6 + \
                (tilde_h*(1+3*tau_sl/2)/(1+tau_sl) + \
                    self.h_half[1]) * phi_current[1]/3 + \
                self.h_half[1] * phi_current[2] / 6
            to_solve[0][0] /= 1 + tau_sl

        u_current: array = solve_linear(to_solve, rhs_ucur)

        for n in range(1,N+1):
            u_flevel_current, forcing_current = u_current[0], forcing[n]
            u_star: float
            u_hat: complex
            if sf_scheme == "FD":
                u_hat = u_flevel_current
            elif sf_scheme == "FV1":
                u_star = self.kappa * np.abs(u_flevel_current) / \
                        ((1+z_star / self.z_full[1]) * \
                        np.log(1+self.z_full[1] / z_star) - 1)
                u_hat = u_star / self.kappa * np.log(1+ self.z_full[1]/ z_star)
            elif sf_scheme == "FV2":
                u_hat = u_flevel_current + self.h_half[0] * \
                        (phi_current[0] / 6 + phi_current[1]/3)
            elif sf_scheme == "FV3":
                u_hat = u_current[1] - self.h_half[1] * \
                        (phi_current[2] / 6 + phi_current[1]/3)
            elif sf_scheme == "FV free":
                tilde_h: float = self.z_full[1] - delta_sl
                tau_sl = 1 + z_star/delta_sl - 1/np.log(1+delta_sl/z_star)
                u_hat = (u_current[0] - tilde_h * \
                        (phi_current[0]/3 + phi_current[1]/6)) / \
                            (1+tau_sl)
            u_star = self.kappa * np.abs(u_hat) \
                    / np.log(1 + delta_sl/z_star) \
                    if sf_scheme != "FV1" else u_star


            h_cl = self.defaulth_cl if np.abs(self.f) < 1e-10 \
                    else np.abs(c_1*u_star/self.f)
            # K_full = u_star * G_full + K_mol.
            G_full: array = self.kappa * self.z_full * \
                    (1 - self.z_full/h_cl)**2 \
                    * np.heaviside(1 - self.z_full/h_cl, 1)
            K_full: array = u_star * G_full + self.K_mol # viscosity
            K_full[0] = u_star * self.kappa * (delta_sl + z_star)

            Y_diag: array = np.concatenate(([0], 2/3*np.ones(self.M-1), [1/3]))
            Y_udiag: array = np.concatenate(([0],
                [self.h_half[m]/6./self.h_full[m] for m in range(1, self.M)]))
            Y_ldiag: array =  np.concatenate((
                [self.h_half[m-1]/6./self.h_full[m] for m in range(1, self.M)],
                [1/6.]))

            D_diag: array = np.concatenate(([0.],
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
                Y_udiag[0] = 0.
                Y_diag[0] = K_full[0] * np.abs(u_hat) / u_star**2
                D_diag[0] = K_full[0] / self.h_half[0]
                D_udiag[0] = - K_full[1] / self.h_half[0]
            elif sf_scheme in {"FV1", "FV2", "FV3"}:
                Y_udiag[0] = Y_diag[0] = D_udiag[0] = D_ldiag[0] = 0.
                D_diag[0] = K_full[0] * np.abs(u_hat) / u_star**2

                Y_diag[1] = K_full[1] * np.abs(u_hat) / u_star**2 \
                        - self.h_half[1]/3
                Y_udiag[1] = - self.h_half[1]/6
                Y_ldiag[0] = 0.
                D_diag[1] = K_full[1] / self.h_half[1]
                D_udiag[1] = - K_full[2] / self.h_half[1]
            elif sf_scheme == "FV free":
                # boundary condition:
                Y_diag[0] = tilde_h / 3 - K_full[0] * \
                        np.log(1+delta_sl/z_star) / \
                            (self.kappa * u_star)
                Y_udiag[0] = tilde_h / 6
                D_diag[0] = - K_full[0] / tilde_h
                D_udiag[0] = K_full[1]/ tilde_h
                # first level:
                D_diag[1] = -K_full[1] / self.h_full[1] * \
                        (1/tilde_h + 1/self.h_half[1])
                D_ldiag[0] = K_full[0]/self.h_full[1] / tilde_h
                D_udiag[1] = K_full[2]/self.h_full[1] / self.h_half[1]
                Y_diag[1] = (tilde_h + self.h_half[1]) / (3*self.h_full[1])
                Y_ldiag[0] = tilde_h/6./self.h_full[1]
                Y_udiag[1] = self.h_half[1]/6./self.h_full[1]
            else:
                raise NotImplementedError(sf_scheme + " surface flux scheme unknown")

            if time_method == "BE":
                c_I1: complex
                c_I2: complex
                if sf_scheme == "FD":
                    c_I1 = - forcing_current[0]
                    c_I2 = (forcing_current[1] - forcing_current[0]) \
                            / self.h_full[1]
                elif sf_scheme in {"FV1", "FV2", "FV3"}:
                    # in the first grid level we directly
                    # use u_hat: if sf_scheme == "FV1" it does not
                    # take the direction into account.
                    # Moreover, it's an explicit, unstable way.
                    # Fortunately, the first grid level is not
                    # really used (except for FV2)
                    c_I1 = - u_hat
                    c_I2 = - forcing_current[1]
                elif sf_scheme == "FV free":
                    c_I1 = forcing_current[0]
                    c_I2 = 1/self.h_full[1] * (forcing_current[1] - forcing_current[0])

                else:
                    raise NotImplementedError(sf_scheme + " surface flux scheme unknown")

                c_T = (1j*self.f * self.u_g - forcing_current[self.M-1]) / self.h_half[self.M-1]
                c = np.concatenate(([c_I1, c_I2], np.diff(forcing_current[1:]) / \
                                                    self.h_full[2:-1], [c_T]))

                phi_current = self.__backward_euler(
                        Y=(Y_ldiag, Y_diag, Y_udiag),
                        D=(D_ldiag, D_diag, D_udiag), c=c, u=phi_current)
                # u_current = 1/(1+self.dt*1j*self.f) * (u_current + self.dt * \
                #         (np.diff(phi_current * K_full) / self.h_half[:-1] \
                #         + forcing_current))

            else:
                raise NotImplementedError("Integration method not supported")

            # computing u_current for real at each time step (!)
            rhs_ucur = self.h_half[0:-2] * phi_current[0:-2]/6 + \
                    (self.h_half[0:-2] + self.h_half[1:-1])*phi_current[1:-1]/3 + \
                    self.h_half[1:-1] * phi_current[2:] / 6
            rhs_ucur = np.concatenate((rhs_ucur,
                [self.h_half[-2] * (phi_current[-2]/6 + phi_current[-1]/3) \
                            - self.u_g]))
            to_solve = -np.ones(self.M), np.ones(self.M-1)
            if sf_scheme == "FV free":
                # getting overline{u}_{1/2} is particular:
                rhs_ucur[0] = tilde_h *(1+3*tau_sl)/(1+tau_sl)* \
                        phi_current[0]/6 + \
                    (tilde_h*(1+3*tau_sl/2)/(1+tau_sl) + \
                        self.h_half[1]) * phi_current[1]/3 + \
                    self.h_half[1] * phi_current[2] / 6
                to_solve[0][0] /= 1 + tau_sl

            u_current = solve_linear(to_solve, rhs_ucur)


        return u_current, phi_current



    def FD_KPP(self, u_t0: array, forcing: array, method: str="BE",
            z_star: float=1e-1) -> array:
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
        N: int = forcing.shape[0] - 1 # number of time steps
        c_1 = 0.2 # constant for h_cl=c_1*u_star/f in atmosphere

        Y_diag: array = np.concatenate((np.ones(self.M), [0]))
        u_current: array = np.copy(u_t0)
        for n in range(1,N+1):
            u_flevel_current, forcing_current = u_current[0], forcing[n]
            u_star: float = self.kappa * np.abs(u_flevel_current) \
                    / np.log(1 + self.z_half[1]/z_star)

            h_cl = self.defaulth_cl if np.abs(self.f) < 1e-10 else c_1*u_star/self.f
            # K_full = u_star * G_full + K_mol.
            G_full: array = self.kappa * self.z_full * \
                    (1 - self.z_full/h_cl)**2 \
                    * np.heaviside(1 - self.z_full/h_cl, 1)
            K_full: array = u_star * G_full + self.K_mol # viscosity
            D_diag: array = np.concatenate((
                [-K_full[1] / (self.h_half[0] * self.h_full[1]) + \
                       u_star**2/np.abs(u_flevel_current)/self.h_half[0]],
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

            c: array = np.concatenate((forcing_current[:-1], [self.u_g]))

            if method == "BE":
                u_current = self.__backward_euler(
                        Y=(np.zeros(self.M), Y_diag, np.zeros(self.M)),
                        D=(D_ldiag, D_diag, D_udiag), c=c, u=u_current)
            else:
                raise NotImplementedError("Integration method not supported")

        return u_current


    def __backward_euler(self, Y: Tuple[array, array, array],
                        D: Tuple[array, array, array], c: array, u: array):
        """
            integrates once (self.dt) in time the equation
            (partial_t + if)Yu - dt*Du = c
            The scheme is:
            Y(1+dt*if) - D) u_np1 = Y u + dt*c
        """
        to_inverse: Tuple[array, array, array] = add(s_mult(Y, 1 + self.dt * 1j*self.f),
                                        s_mult(D, - self.dt))
        return solve_linear(to_inverse, multiply(Y, u) + self.dt*c)
