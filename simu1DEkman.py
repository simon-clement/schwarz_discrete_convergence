"""
    This module defines the class Simu1dEkman
    which simulates a 1D Ekman problem with various
    discretisations.
"""

from typing import Tuple, List
import bisect
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
            u_geostrophy: float=10., default_h_cl: float=1e4,
            K_mol: float=1e-7, C_D: float=1e-3, f: float=1e-4) -> None:
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
        self.e_min: float = 1e-6
        self.K_min: float = 1e-4
        self.C_m: float = 0.0667
        self.C_e: float = 0.4
        self.c_eps: float = 0.7
        self.z_star: float = 1e-1
        self.implicit_coriolis: float = 0.55 # semi-implicit coefficient
        self.lm_min: float = self.K_min / self.C_m / \
                np.sqrt(self.e_min)
        self.leps_min: float = self.K_min / self.c_eps / \
                np.sqrt(self.e_min)
        self.dictsf_scheme = {
                "FD pure" : (self.__sf_udelta_FDpure,
                                self.__sf_YDc_FDpure),
                "FD Dirichlet" : (self.__sf_udelta_FDDirichlet0,
                                self.__sf_YDc_FDDirichlet0),
                "FD2" : (self.__sf_udelta_FD2,
                                self.__sf_YDc_FD2),
                "FV pure" : (self.__sf_udelta_FVpure,
                                self.__sf_YDc_FVpure),
                "FV Dirichlet" : (self.__sf_udelta_FVDirichlet0,
                                self.__sf_YDc_FVDirichlet0),
                "FV1" : (self.__sf_udelta_FV1,
                                self.__sf_YDc_FV1),
                "FV2" : (self.__sf_udelta_FV2,
                                self.__sf_YDc_FV2),
                "FV1 free" : (self.__sf_udelta_FVfree,
                                self.__sf_YDc_FVfree),
                "FV2 free" : (self.__sf_udelta_FV2free,
                                self.__sf_YDc_FV2free) }

    def reconstruct_FV(self, u_bar: array, phi: array,
            sf_scheme: str="FV pure", delta_sl: float=None):
        """
            spline reconstruction of the FV solution.
            u(xi) = u_bar[m] + (phi[m+1] + phi[m]) * xi/2 +
                    (phi[m+1] - phi[m]) / (2*h_half[m]) *
                        (xi^2 - h_half[m]^2/12)
            where xi = linspace(-h_half[m]/2, h_half[m]/2, 10, endpoint=True)
        """
        xi = [np.linspace(-h/2, h/2, 20) for h in self.h_half[:-1]]
        sub_discrete: List[array] = [u_bar[m] + (phi[m+1] + phi[m]) * xi[m]/2 \
                + (phi[m+1] - phi[m]) / (2 * self.h_half[m]) * \
                (xi[m]**2 - self.h_half[m]**2/12) for m in range(self.M)]
        u_oversampled = np.concatenate(sub_discrete[1:])
        z_oversampled = np.concatenate([np.array(xi[m]) + self.z_half[m]
                                            for m in range(1, self.M)])
        if sf_scheme == "FV Dirichlet":
            allxi = [np.array(xi[m]) + self.z_half[m] for m in range(self.M)]
            return np.concatenate(allxi), np.concatenate(sub_discrete)

        if delta_sl is None:
            delta_sl = self.z_half[0] if sf_scheme in\
                    {"FV pure", "FV1"} else self.z_full[1]
        k1: int = bisect.bisect_right(self.z_full[1:], delta_sl)
        prognostic: array = np.concatenate((u_bar[:k1+1], phi[k1:]))

        z_log: array = np.linspace(0, delta_sl, 10)
        func_un, _ = self.dictsf_scheme[sf_scheme]
        u_delta: complex = func_un(prognostic=prognostic, delta_sl=delta_sl)
        u_star: float = self.kappa * np.abs(u_delta) \
                / np.log(1 + delta_sl/self.z_star)
        u_log: complex = u_star/self.kappa * np.log(1+z_log/self.z_star) \
                            * u_delta/np.abs(u_delta)

        k2: int = bisect.bisect_right(z_oversampled, delta_sl)
        return np.concatenate((z_log, z_oversampled[k2:])), \
                np.concatenate((u_log, u_oversampled[k2:]))


    def FV(self, u_t0: array, phi_t0: array, forcing: array,
            sf_scheme: str="FV pure", turbulence: str="TKE",
            delta_sl: float=None) -> array:
        """
            Integrates in time with Backward Euler the model with KPP
            and Finite differences.

            u_t0 should be given at half-levels (follow self.z_half)
            phi_t0 should be given at full-levels (follow self.z_full)
            forcing should be given as averaged on each volume for all times
            sf_scheme is the surface flux scheme:
                - "FV pure" for a FD interpretation (classical but biaised)
                - "FV{1, 2}" for a FV interpretation
                                (unbiaised but delta_{sl}=z_1)
                - "FV{1, 2} free" for a FV interpretation with a free delta_{sl}
                - "FV Dirichlet" for debug purpose
            turbulence should be one of:
                - "TKE" (for one-equation TKE)
                - "KPP" (for simple K-profile parametrization)
            delta_sl should be provided only with sf_scheme="FV1 free"
                (for delta_sl < z_1) or "FV2 free" (for any delta_sl)
            returns a numpy array of shape (M)
                                    where M is the number of space points
        """
        assert u_t0.shape[0] == self.M and phi_t0.shape[0] == self.M + 1

        N: int = forcing.shape[0] - 1 # number of time steps
        assert forcing.shape[1] == self.M
        c_1 = 0.2 # constant for h_cl in atmosphere
        e_sl = u_t0[0] / np.sqrt(self.C_m*self.c_eps)
        func_un, func_YDc_sf = self.dictsf_scheme[sf_scheme]
        if delta_sl is None:
            delta_sl = self.z_half[0] if sf_scheme in \
                    {"FV pure", "FV1", "FV Dirichlet"} \
                    else self.z_full[1]
        tke = np.linspace(e_sl, self.e_min, self.M + 1)
        l_m, _ = self.__mixing_lengths(delta_sl)
        k = bisect.bisect_right(self.z_full[1:], delta_sl)

        prognostic: array = np.concatenate((u_t0[:k+1], phi_t0[k:]))
        u_current: array = np.copy(u_t0)
        all_u_star = []

        for n in range(1,N+1):
            forcing_current = forcing[n]
            u_delta: complex = func_un(prognostic=prognostic,
                    delta_sl=delta_sl)
            u_star: float = self.kappa * np.abs(u_delta) \
                    / np.log(1 + delta_sl/self.z_star)
            all_u_star += [u_star]

            K_full: array
            if turbulence == "KPP":
                h_cl: float = self.defaulth_cl if np.abs(self.f) < 1e-10 \
                        else np.abs(c_1*u_star/self.f)
                G_full: array = self.kappa * self.z_full * \
                        (1 - self.z_full/h_cl)**2 \
                        * np.heaviside(1 - self.z_full/h_cl, 1)
                K_full = u_star * G_full + self.K_mol # viscosity
                K_full[0] = u_star * self.kappa * (delta_sl+self.z_star)
            elif turbulence == "TKE":
                K_full = self.C_m * l_m * np.sqrt(tke)
            else:
                raise NotImplementedError("Wrong turbulence scheme")

            Y_diag: array = np.concatenate(([0, 0.], 2/3*np.ones(self.M-1),
                                                            [1/3]))
            Y_udiag: array = np.concatenate(([0, 0.],
                [self.h_half[m]/6./self.h_full[m] for m in range(1, self.M)]))
            Y_ldiag: array =  np.concatenate(([0.],
                [self.h_half[m-1]/6./self.h_full[m] for m in range(1, self.M)],
                [1/6.]))

            D_diag: array = np.concatenate(([0., 0.],
                [-2 * K_full[m] / self.h_half[m] / self.h_half[m-1]
                                            for m in range(1, self.M)],
                [-K_full[self.M] / self.h_half[self.M - 1]**2]))
            D_udiag: array = np.concatenate(([0.,  0],
                [K_full[m+1]/self.h_full[m] / self.h_half[m]
                                            for m in range(1, self.M)]))
            D_uudiag: array = np.zeros(self.M)

            D_ldiag: array = np.concatenate(([0.],
                [K_full[m-1]/self.h_full[m] / self.h_half[m-1]
                                            for m in range(1, self.M)],
                [K_full[self.M-1] / self.h_half[self.M - 1]**2]))

            c_T = (1j*self.f * self.u_g - forcing_current[self.M-1]) \
                    / self.h_half[self.M-1]
            c = np.concatenate(([0, 0], np.diff(forcing_current) / \
                                                self.h_full[1:-1], [c_T]))
            Y = (Y_ldiag, Y_diag, Y_udiag)
            D = (D_ldiag, D_diag, D_udiag, D_uudiag)
            Y_sf, D_sf, c_sf = func_YDc_sf(K=K_full,
                    forcing=forcing_current, ustar=u_star,
                    un=u_delta, delta_sl=delta_sl)
            for y, y_sf in zip(Y, Y_sf):
                y_sf = np.array(y_sf)
                y[:y_sf.shape[0]] = y_sf
            for d, d_sf in zip(D, D_sf):
                d_sf = np.array(d_sf)
                d[:d_sf.shape[0]] = d_sf

            c_sf = np.array(c_sf)
            c[:c_sf.shape[0]] = np.array(c_sf)
            prognostic = self.__backward_euler(Y=Y, D=D, c=c,
                    u=prognostic)
            next_u = 1/(1+self.dt*1j*self.f*self.implicit_coriolis) * \
                    ((1 - self.dt*1j*self.f*(1-self.implicit_coriolis)) * \
                    u_current + self.dt * \
                    (np.diff(prognostic[1:] * K_full) / self.h_half[:-1] \
                    + forcing_current))
            next_u[:k+1] = prognostic[:k+1]
            u_current, old_u = next_u, u_current

            phi = prognostic[k+1:]
            if k > 0:
                phi = np.concatenate((K_full[k]*phi[0]*K_full[:k], phi))

            ####### TKE SCHEME #############
            if turbulence == "TKE":
                shear = np.concatenate(([0],
                    [np.abs(K_full[m]*phi[m]*(
                        old_u[m] - old_u[m-1] + u_current[m] - u_current[m-1] \
                                )/self.h_full[m]/2) \
                            for m in range(1, self.M)], [0]))
                u_delta: complex = func_un(prognostic=prognostic,
                        delta_sl=delta_sl)
                u_star: float = self.kappa * np.abs(u_delta) \
                        / np.log(1 + delta_sl/self.z_star)

                tke = self.__integrate_tke(tke, shear,
                        delta_sl=delta_sl, u_star=u_star)

        return u_current, phi, tke, all_u_star


    def FD(self, u_t0: array, forcing: array,
            turbulence: str="TKE", sf_scheme: str="FD pure") -> array:
        """
            Integrates in time with Backward Euler the model with KPP
            and Finite differences.

            u_t0 should be given at half-levels (follow self.z_half)
            forcing should be given at half-levels for all times
            turbulence should be one of:
                TKE (for one-equation TKE)
                KPP (for simple K-profile parametrization)
            sf_scheme should be one of:
                FD pure (delta_sl = z_1/2)
                FD2     (delta_sl = z_1)
                FD Dirichlet (no real delta_sl)
            returns a numpy array of shape (M)
                                    where M is the number of space points
        """
        assert u_t0.shape[0] == self.M + 1
        N: int = forcing.shape[0] - 1 # number of time steps
        c_1 = 0.2 # constant for h_cl=c_1*u_star/f in atmosphere
        func_un, func_YDc_sf = self.dictsf_scheme[sf_scheme]
        delta_sl = self.z_half[0] if sf_scheme in {"FD pure", "FD Dirichlet"} \
                else self.z_full[1]
        e_sl = u_t0[0] / np.sqrt(self.C_m*self.c_eps)
        tke = np.linspace(e_sl, self.e_min, self.M + 1)
        l_m, _ = self.__mixing_lengths(delta_sl)

        Y_diag: array = np.concatenate((np.ones(self.M), [0]))
        u_current: array = np.copy(u_t0)
        all_u_star = []
        for n in range(1,N+1):
            forcing_current = forcing[n]
            u_delta = func_un(prognostic=u_current)

            u_star: float = self.kappa * np.abs(u_delta) \
                    / np.log(1 + delta_sl/self.z_star)
            all_u_star += [u_star]

            if turbulence =="KPP":
                h_cl: float = self.defaulth_cl if np.abs(self.f) < 1e-10 \
                        else c_1*u_star/self.f
                G_full: array = self.kappa * self.z_full * \
                        (1 - self.z_full/h_cl)**2 \
                        * np.heaviside(1 - self.z_full/h_cl, 1)

                K_full: array = u_star * G_full + self.K_mol # viscosity
                K_full[0] = u_star * self.kappa * (delta_sl + self.z_star)
            elif turbulence == "TKE":
                K_full = self.C_m * l_m * np.sqrt(tke)
            else:
                raise NotImplementedError("Wrong turbulence scheme")

            D_diag: array = np.concatenate(( [0.],
                [(-K_full[m+1]/self.h_full[m+1] - K_full[m]/self.h_full[m]) \
                        / self.h_half[m] for m in range(1, self.M)],
                [-1]))
            D_udiag: array = np.concatenate(([0.],
                [K_full[m+1]/self.h_full[m+1] / self.h_half[m]
                    for m in range(1, self.M)]))
            D_ldiag: array = np.concatenate((
                [K_full[m]/self.h_full[m] / self.h_half[m]
                    for m in range(1, self.M)]
                ,[0]))

            c: array = np.concatenate((forcing_current[:-1], [self.u_g]))
            Y_sf, D_sf, c_sf = func_YDc_sf(K=K_full,
                    forcing=forcing_current, ustar=u_star,
                    un=u_delta)
            assert len(Y_sf[0]) == 0 and Y_sf[2] == (0.,)
            Y_diag[:1] = Y_sf[1] # raises an error if Y_sf[1].shape[0]>1
            for d, d_sf in zip((D_ldiag, D_diag, D_udiag), D_sf):
                d_sf = np.array(d_sf)
                d[:d_sf.shape[0]] = d_sf

            c_sf = np.array(c_sf)
            c[:c_sf.shape[0]] = np.array(c_sf)

            next_u = self.__backward_euler(
                    Y=(np.zeros(self.M), Y_diag, np.zeros(self.M)),
                    D=(D_ldiag, D_diag, D_udiag), c=c, u=u_current)
            u_current, old_u = next_u, u_current

            if turbulence == "TKE":
                ####### TKE SCHEME #############
                du = np.diff(u_current)
                du_old = np.diff(old_u)
                shear = np.concatenate(([0],
                    [np.abs(K_full[m]/self.h_full[m]**2 * du[m-1] * \
                            (du[m-1]+du_old[m-1])/2) \
                            for m in range(1, self.M)], [0]))
                u_delta = func_un(prognostic=u_current)

                u_star: float = self.kappa * np.abs(u_delta) \
                        / np.log(1 + delta_sl/self.z_star)

                tke = self.__integrate_tke(tke, shear,
                        delta_sl=delta_sl, u_star=u_star)

        return u_current, tke, all_u_star

    def __integrate_tke(self, tke, shear, delta_sl, u_star):
        l_m, l_eps = self.__mixing_lengths(delta_sl)
        Ke_half = self.C_e * (l_m[1:] + l_m[:-1])/2 * \
                np.sqrt((tke[1:] + tke[:-1])/2)
        diag_e = np.concatenate(([1],
                    [1/self.dt + self.c_eps*tke[m]/l_eps[m] \
                    + (Ke_half[m]/self.h_half[m] + \
                        Ke_half[m-1]/self.h_half[m-1]) \
                        / self.h_full[m] for m in range(1, self.M)],
                                [1]))
        ldiag_e = np.concatenate((
            [ -Ke_half[m-1] / self.h_half[m-1] / self.h_full[m] \
                    for m in range(1,self.M) ], [0]))
        udiag_e = np.concatenate(([0],
            [ -Ke_half[m] / self.h_half[m] / self.h_full[m] \
                    for m in range(1,self.M) ]))

        e_sl = u_star**2/np.sqrt(self.C_m*self.c_eps)
        rhs_e = np.concatenate(([e_sl],
            [tke[m]/self.dt + shear[m] for m in range(1, self.M)],
            [self.e_min]))

        return solve_linear((ldiag_e, diag_e, udiag_e), rhs_e)

    def __mixing_lengths(self, delta_sl):
        l_down = np.maximum(self.lm_min, self.z_full[-1] - self.z_full)
        l_up = np.maximum(self.lm_min, self.z_full - delta_sl)

        l_eps = np.abs(np.amin((l_up, l_down), axis=0))
        a = -3/2
        l_m = np.abs((0.5*(l_up**(1/a)+l_down**(1/a)))**a)
        l_m[self.z_full <= delta_sl] = np.maximum(self.lm_min,
                self.kappa * \
                (self.C_m * self.c_eps)**(1/4) / self.C_m * \
                (self.z_full[self.z_full < delta_sl] + self.z_star))
        return l_m, l_eps

    def __backward_euler(self, Y: Tuple[array, array, array],
                        D: Tuple[array, array, array], c: array, u: array):
        """
            integrates once (self.dt) in time the equation
            (partial_t + if)Yu - dt*Du = c
            The scheme is:
            Y(1+dt*if*gamma) - D) u_np1 = Y u + dt*c
        """
        to_inverse: Tuple[array, array, array] = add(s_mult(Y,
                                1 + self.implicit_coriolis * \
                            self.dt * 1j*self.f), s_mult(D, - self.dt))
        return solve_linear(to_inverse,
                (1 - (1 - self.implicit_coriolis) * self.dt*1j*self.f) * \
                                multiply(Y, u) + self.dt*c)

    ########### DEFINITION OF SF SCHEMES : VALUE OF u######""
    def __sf_udelta_FDDirichlet0(self, prognostic, **_):
        return prognostic[0]

    def __sf_udelta_FDpure(self, prognostic, **_):
        return prognostic[0]

    def __sf_udelta_FD2(self, prognostic, **_):
        return (prognostic[0] + prognostic[1])/2

    def __sf_udelta_FVDirichlet0(self, prognostic, **_):
        return prognostic[0] - self.h_half[0]* \
                (prognostic[2] - prognostic[1])/24

    def __sf_udelta_FVpure(self, prognostic, **_):
        return prognostic[0] - self.h_half[0]* \
                (prognostic[2] - prognostic[1])/24

    def __sf_udelta_FV1(self, prognostic, **_):
        return prognostic[0]

    def __sf_udelta_FV2(self, prognostic, **_):
        return prognostic[1] - self.h_half[1] * \
                (prognostic[3]/6 + prognostic[2]/3)

    def __sf_udelta_FVfree(self, prognostic, delta_sl, **_):
        tilde_h = self.z_full[1] - delta_sl
        tau_sl = 1+self.z_star/delta_sl - 1/np.log(1+delta_sl/self.z_star)
        return (prognostic[0] - tilde_h * \
                (prognostic[2]/6 + prognostic[1]/3))/(1+tau_sl)

    def __sf_udelta_FV2free(self, prognostic, delta_sl, **_):
        k = bisect.bisect_right(self.z_full[1:], delta_sl)
        zk = self.z_full[k]
        tilde_h = self.z_full[k+1] - delta_sl
        if zk > 0:
            tau_sl = 1+self.z_star/delta_sl - \
                            1/np.log(1+delta_sl/self.z_star) + zk * \
                    + (1 - (1+self.z_star/zk)*np.log(1+zk/self.z_star)) \
                    / (delta_sl * np.log(1+delta_sl/self.z_star))
        else:
            tau_sl = 1+self.z_star/delta_sl - \
                            1/np.log(1+delta_sl/self.z_star)
        return (prognostic[k] - tilde_h * \
                (prognostic[k+1] / 3 + prognostic[k+2] / 6)) \
                / (1+tau_sl)

    def __sf_YDc_FDDirichlet0(self, **_):
        Y = ((), (0.,), (0.,))
        D = ((), (1+self.h_half[0]/2/self.h_full[1],),
                (-self.h_half[0]/2/self.h_full[1],))
        c = (0.,)
        return Y, D, c

    def __sf_YDc_FDpure(self, K, forcing, ustar, un, **_):
        Y = ((), (1.,), (0.,))
        D = ((), (-K[1]/self.h_full[1]/self.h_half[1] + \
                ustar**2 / np.abs(un), ),
                (-K[1]/self.h_full[1]/self.h_half[1],))
        c = (forcing[0],)
        return Y, D, c

    def __sf_YDc_FD2(self, K, ustar, un, **_):
        Y = ((), (0.,), (0.,))
        D = ((), (-K[1]/self.h_full[1] + ustar**2 / np.abs(un) / 2,),
                (K[1]/self.h_full[1] + ustar**2 / np.abs(un) / 2,))
        c = (0.,)
        return Y, D, c

    def __sf_YDc_FVDirichlet0(self, K, forcing, **_):
        Y = ((1.,), (0., 0.), (0., 0.))
        D = ((0.,), (1, -K[0] / self.h_half[0]),
                (- self.h_half[0]/3, K[1]/self.h_half[0]),
                (-self.h_half[0]/6,))
        c = (0., forcing[0])
        return Y, D, c

    def __sf_YDc_FVpure(self, K, forcing, ustar, un, **_):
        Y = ((1.,), (0., 0.), (0., 0.))
        D = ((0.,), (1, -K[0] / self.h_half[0]),
                (K[0]*np.abs(un)/ustar**2+self.h_half[0]/24,
                    K[1]/self.h_half[0]),
                (-self.h_half[0]/24,))
        c = (0., forcing[0])
        return Y, D, c

    def __sf_YDc_FV1(self, K, forcing, ustar, un, **_):
        Y = ((1.,), (0., 0.), (0., 0.))
        D = ((0.,), (1, -K[0] / self.h_half[0]),
                (K[0]*np.abs(un)/ustar**2, K[1]/self.h_half[0]))
        c = (0., forcing[0])
        return Y, D, c

    def __sf_YDc_FV2(self, K, forcing, ustar, un, **_):
        f = lambda z: (z+self.z_star)*np.log(1+z/self.z_star) - z
        ratio_norms = (f(self.z_full[1]) - f(0)) / \
                (f(self.z_full[2]) - f(self.z_full[1]))
        Y = ((0., 1.), (0., 0., 0.), (0., 0., 0.))
        D = ((0., 0.), (1., 1., -K[1] / self.h_half[1]),
                (-ratio_norms, K[1]*np.abs(un)/ustar**2 - \
                        self.h_half[1]/3, K[2]/self.h_half[1]),
                (0., -self.h_half[1]/6))
        c = (0., 0., forcing[1])
        return Y, D, c

    def __sf_YDc_FVfree(self, K, forcing, ustar, un, delta_sl, **_):
        tau_sl = 1+self.z_star/delta_sl - 1/np.log(1+delta_sl/self.z_star)
        tilde_h = self.z_full[1] - delta_sl
        Y = ((1/(1+tau_sl), tilde_h / 6 / self.h_full[1]),
                (0., tilde_h*tau_sl/3/(1+tau_sl),
                    tilde_h/3/self.h_full[1] + \
                            self.h_half[1]/3/self.h_full[1]),
                (0., tilde_h*tau_sl/6/(1+tau_sl),
                    self.h_half[1]/6/self.h_full[1]))

        D = ((0., K[0]/tilde_h/self.h_full[1]),
                (1., -K[0] / tilde_h,
                    -K[1]/tilde_h/self.h_full[1] - \
                            K[1] / self.h_half[1] / self.h_full[1]),
                (K[0]*np.abs(un)*(1+tau_sl)/ustar**2 - tilde_h/3,
                    K[1]/tilde_h, K[2]/self.h_full[1]/self.h_half[1]),
                (-tilde_h/6, 0.))
        c = (0., forcing[0], (forcing[1] - forcing[0])/self.h_full[1])
        return Y, D, c

    def __sf_YDc_FV2free(self, K, forcing, ustar, un, delta_sl, **_):
        k = bisect.bisect_right(self.z_full[1:], delta_sl)
        zk = self.z_full[k]
        tilde_h = self.z_full[k+1] - delta_sl
        tau_sl = 1+self.z_star/delta_sl - \
                        1/np.log(1+delta_sl/self.z_star) + zk * \
                + (1 - (1+self.z_star/zk)*np.log(1+zk/self.z_star)) \
                / (delta_sl * np.log(1+delta_sl/self.z_star))

        Y = ((1/(1+tau_sl), tilde_h / 6 / self.h_full[k+1]),
                (0., tilde_h*tau_sl/3/(1+tau_sl),
                    tilde_h/3/self.h_full[k+1] + \
                            self.h_half[k+1]/3/self.h_full[k+1]),
                (0., tilde_h*tau_sl/6/(1+tau_sl),
                    self.h_half[k+1]/6/self.h_full[k+1]))

        D = ((0., K[k+0]/tilde_h/self.h_full[k+1]),
                (1., -K[k+0] / tilde_h,
                    -K[k+1]/tilde_h/self.h_full[k+1] - \
                            K[k+1] / self.h_half[k+1] / self.h_full[k+1]),
                (K[k+0]*np.abs(un)*(1+tau_sl)/ustar**2 - tilde_h/3,
                    K[k+1]/tilde_h, K[k+2]/self.h_full[k+1]/self.h_half[k+1]),
                (-tilde_h/6, 0.))
        c = (0., forcing[k+0],
                (forcing[k+1] - forcing[k+0])/self.h_full[k+1])
        Y = (np.concatenate((np.zeros(k), y)) for y in Y)
        f = lambda z: (z+self.z_star)*np.log(1+z/self.z_star) - z
        ratio_norms = ((f(self.z_full[m+1]) - f(self.z_full[m])) / \
                (f(self.z_full[m+2]) - f(self.z_full[m+1])) \
                    for m in range(k))
        D = (np.concatenate((np.zeros(k), D[0])),
                np.concatenate((np.ones(k), D[1])),
                np.concatenate((ratio_norms, D[2])),
                np.concatenate((np.zeros(k), D[3])))
        c = np.concatenate((np.zeros(k), c))
        return Y, D, c
