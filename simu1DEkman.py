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
from utils_linalg import solve_linear, full_to_half

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
        self.C_m: float = 0.126
        self.C_e: float = 0.34
        self.c_eps: float = 0.845
        self.z_star: float = 1e-1
        self.implicit_coriolis: float = 0.55 # semi-implicit coefficient
        self.lm_min: float = self.K_min / self.C_m / \
                np.sqrt(self.e_min)
        self.leps_min: float = self.K_min / self.c_eps / \
                np.sqrt(self.e_min)
        # For each name of sf_scheme, two corresponding
        # methods defined at the bottom of this class:
        # sf_udelta_* is how we compute u(delta_sl) and
        # sf_YDc_* is the definition of the bottom boundary condition
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
        func_un, func_YDc_sf = self.dictsf_scheme[sf_scheme]
        if delta_sl is None:
            delta_sl = self.z_half[0] if sf_scheme in \
                    {"FV pure", "FV1", "FV Dirichlet"} \
                    else self.z_full[1]
        if sf_scheme == "FV Dirichlet":
            delta_sl = self.z_half[1]

        tke = np.ones(self.M) * self.e_min
        dz_tke = np.zeros(self.M+1)
        l_m, _ = self.mixing_lengths(delta_sl)
        K_full: array = self.K_min + np.zeros(self.M+1)
        k = bisect.bisect_right(self.z_full[1:], delta_sl)
        if sf_scheme == "FV Dirichlet":
            k=0

        phi = phi_t0
        old_phi = np.copy(phi)

        prognostic: array = np.concatenate((u_t0[:k+1], phi_t0[k:]))
        u_current: array = np.copy(u_t0)
        all_u_star = []

        for n in range(1,N+1):
            forcing_current = forcing[n]
            u_delta: complex = func_un(prognostic=prognostic,
                    delta_sl=delta_sl, u=u_current, phi=phi)
            u_star: float = self.kappa * np.abs(u_delta) \
                    / np.log(1 + delta_sl/self.z_star)
            all_u_star += [u_star]
            K_full, tke, dz_tke = self.__visc_turb_FV(u_star,
                    delta_sl, turbulence="TKE", phi=phi,
                    old_phi=old_phi, K_full=K_full, tke=tke,
                    dz_tke=dz_tke)

            Y, D, c = self.__matrices_u_FV(K_full, forcing_current)

            self.__apply_sf_scheme(sf_scheme, Y=Y, D=D, c=c,
                    K_full=K_full, forcing=forcing_current,
                    u_star=u_star, u_delta=u_delta, delta_sl=delta_sl)

            prognostic = self.__backward_euler(Y=Y, D=D, c=c,
                    u=prognostic)
            next_u = 1/(1+self.dt*1j*self.f*self.implicit_coriolis) * \
                    ((1 - self.dt*1j*self.f*(1-self.implicit_coriolis)) * \
                    u_current + self.dt * \
                    (np.diff(prognostic[1:] * K_full) / self.h_half[:-1] \
                    + forcing_current))
            next_u[:k+1] = prognostic[:k+1]

            u_current, old_u = next_u, u_current

            old_phi, phi = phi, prognostic[k+1:]

            if k > 0: # constant flux layer : K[:k] phi[:k] = K[0] phi[0]
                phi = np.concatenate((K_full[k]*phi[0]*K_full[:k], phi))

        u_delta: complex = func_un(prognostic=prognostic,
                delta_sl=delta_sl, u=u_current, phi=phi)
        u_star: float = self.kappa * np.abs(u_delta) \
                / np.log(1 + delta_sl/self.z_star)
        all_u_star += [u_star]
        # Representation of TKE as in FD
        tke = self.__compute_tke_full(tke, dz_tke, u_star,
                delta_sl, k)
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
        assert u_t0.shape[0] == self.M
        assert forcing.shape[1] == self.M
        N: int = forcing.shape[0] - 1 # number of time steps
        func_un, func_YDc_sf = self.dictsf_scheme[sf_scheme]
        delta_sl = self.z_half[0] if sf_scheme in {"FD pure", "FD Dirichlet"} \
                else self.z_full[1]
        if sf_scheme == "FD Dirichlet":
            delta_sl = self.z_half[1] # avoid differences with FV
        tke = np.ones(self.M+1) * self.e_min
        K_full: array = self.K_min + np.zeros(self.M+1)

        u_current: array = np.copy(u_t0)
        old_u = np.copy(u_current)
        all_u_star = []
        for n in range(1,N+1):
            forcing_current = forcing[n]
            u_delta = func_un(prognostic=u_current, delta_sl=delta_sl)

            u_star: float = self.kappa * np.abs(u_delta) \
                    / np.log(1 + delta_sl/self.z_star)
            all_u_star += [u_star]

            K_full, tke = self.__visc_turb_FD(u_star=u_star,
                    delta_sl=delta_sl, turbulence="TKE",
                    u_current=u_current, old_u=old_u,
                    K_full=K_full, tke=tke)

            Y, D, c = self.__matrices_u_FD(K_full, forcing_current)
            self.__apply_sf_scheme(sf_scheme, Y=Y, D=D, c=c,
                    K_full=K_full, forcing=forcing_current,
                    u_star=u_star, u_delta=u_delta, delta_sl=delta_sl)

            next_u = self.__backward_euler(Y=Y, D=D, c=c, u=u_current)
            u_current, old_u = next_u, u_current

        return u_current, tke, all_u_star


    def __integrate_tke(self, tke, shear, K_full, delta_sl, u_star):
        """
            integrates TKE equation on one time step.
            discretisation of TKE is Finite Differences,
            located on half-points.
        """
        _, l_eps = self.mixing_lengths(delta_sl)
        Ke_half = self.C_e / self.C_m * (K_full[1:] + K_full[:-1])/2
        diag_e = np.concatenate(([1],
                    [1/self.dt + self.c_eps*np.sqrt(tke[m])/l_eps[m] \
                    + (Ke_half[m]/self.h_half[m] + \
                        Ke_half[m-1]/self.h_half[m-1]) \
                        / self.h_full[m] for m in range(1, self.M)],
                        [1/self.dt + Ke_half[self.M-1] / \
                        self.h_half[self.M-1] / self.h_full[self.M]]))
        ldiag_e = np.concatenate((
            [ -Ke_half[m-1] / self.h_half[m-1] / self.h_full[m] \
                    for m in range(1,self.M) ], [- Ke_half[self.M-1] / \
                        self.h_half[self.M-1] / self.h_full[self.M]]))
        udiag_e = np.concatenate(([0],
            [ -Ke_half[m] / self.h_half[m] / self.h_full[m] \
                    for m in range(1,self.M) ]))

        e_sl = np.maximum(u_star**2/np.sqrt(self.C_m*self.c_eps),
                                    self.e_min)
        rhs_e = np.concatenate(([e_sl],
            [tke[m]/self.dt + shear[m] for m in range(1, self.M)],
            [tke[self.M]/self.dt]))
        # if delta_sl is inside the computational domain:
        k = bisect.bisect_right(self.z_full[1:], delta_sl)
        rhs_e[:k+1] = e_sl # prescribe e=e_sl in all the ASL
        diag_e[:k+1] = 1.
        udiag_e[:k+1] = ldiag_e[:k] = 0.

        return solve_linear((ldiag_e, diag_e, udiag_e), rhs_e)

    def __integrate_tke_FV(self, tke, dz_tke, shear, K_full,
            delta_sl, u_star):
        """
            integrates TKE equation on one time step.
            discretisation of TKE is Finite Volumes,
            centered on half-points.
            tke is the array of averages,
            dz_tke is the array of space derivatives.
            shear is K_u ||dz u|| at full levels.
            K_full is K_u (at full levels)
            delta_sl is the height of the SL
            u_star is the friction scale

            WARNING DELTA_SL IS NOT SUPPOSED TO MOVE !!
            to do this, e_sl of the previous time should be known.
            (otherwise a discontinuity between the
            cell containing the SL's top and the one above
            may appear.)
            tke is the average value on the whole cell,
            except tke[k] which is the average value on
            the "subvolume" [delta_sl, z_{k+1}].
            returns tke, dz_tke
        """
        _, l_eps = self.mixing_lengths(delta_sl)
        # computing buoyancy on half levels:
        shear_half = full_to_half(shear)
        buoy_half = np.zeros_like(shear_half)

        # deciding in which cells we use Patankar's trick
        PATANKAR = (shear_half <= buoy_half)
        # turbulent viscosity of tke :
        Ke_full = K_full * self.C_e / self.C_m
        l_eps_half = full_to_half(l_eps)
        # bottom value:
        e_sl = np.maximum(u_star**2/np.sqrt(self.C_m*self.c_eps),
                                    self.e_min)
        k = bisect.bisect_right(self.z_full[1:], delta_sl)

        h_tilde = self.z_full[k+1] - delta_sl

        # definition of functions to interlace and deinterace:
        def interlace(arr1, arr2):
            """ interlace arrays of same size, beginning with arr1[0] """
            if arr1.shape[0] == arr2.shape[0]:
                return np.vstack((arr1, arr2)).T.flatten()
            assert arr1.shape[0] == arr2.shape[0] + 1
            return np.vstack((arr1,
                np.concatenate((arr2, [0])))).T.flatten()[:-1]

        def deinterlace(arr):
            """ returns a tuple of the two interlaced arrays """
            return arr[1::2], arr[::2]

        ####### ODD LINES: evolution of tke = overline{e}
        # data located at m+1/2, 0<=m<=M
        # odd_{l, ll}diag index is *not* shifted
        odd_ldiag = Ke_full[:-1] / self.h_half[:-1]
        odd_udiag = -Ke_full[1:] / self.h_half[:-1]
        odd_diag = self.c_eps * np.sqrt(tke) / l_eps_half + \
                PATANKAR * buoy_half / tke + 1/self.dt
        odd_rhs = shear_half - (1 - PATANKAR)* buoy_half + tke/self.dt
        # odd_diag[0], odd_rhs[0], odd_udiag[0] = 1., e_sl, 0.
        odd_lldiag = np.zeros(odd_ldiag.shape[0] - 1)
        odd_uudiag = np.zeros(odd_ldiag.shape[0] - 1)
        # inside the surface layer:
        odd_ldiag[:k] = 0.
        odd_udiag[:k] = 0.
        odd_ldiag[k] = Ke_full[k] / h_tilde
        odd_udiag[k] = -Ke_full[k+1] / h_tilde
        odd_rhs[:k] = e_sl
        odd_diag[:k] = 1.

        ####### EVEN LINES: evolution of dz_tke = (partial_z e)
        # data located at m, 0<m<M
        # notice that even_{l, ll}diag index is shifted
        even_diag = np.concatenate(([-self.h_half[0]*5/12],
                10*self.h_full[1:-1]/12 / self.dt + \
                Ke_full[1:-1]/self.h_half[1:-1] + \
                Ke_full[1:-1] / self.h_half[:-2],
                [1]
            ))
        even_udiag = np.concatenate(([1],
                buoy_half[1:]*PATANKAR[1:] / tke[1:] + \
                self.c_eps * np.sqrt(tke[1:]) / l_eps_half[1:]))
        even_uudiag = np.concatenate(([-self.h_half[0]/12],
                self.h_half[1:-1]/12/self.dt - \
                Ke_full[2:]/self.h_half[1:-1]))

        even_ldiag = np.concatenate((
                - buoy_half[:-1]*PATANKAR[:-1] / tke[:-1] - \
                self.c_eps * np.sqrt(tke[:-1]) / l_eps_half[:-1],
                [0]))
        even_lldiag = np.concatenate((
                self.h_half[:-2]/12/self.dt - \
                Ke_full[:-2]/self.h_half[:-2],
                [0]
                ))

        even_rhs = np.concatenate(( [e_sl],
                1/self.dt * (self.h_half[:-2]* dz_tke[:-2]/12 \
                + 10*self.h_full[1:-1]* dz_tke[1:-1]/12 \
                + self.h_half[1:-1]* dz_tke[2:]/12) \
                + np.diff(shear_half - buoy_half*(1-PATANKAR)),
                [0]))
        # e(delta_sl) = e_sl :
        even_diag[k] = -h_tilde*5/12
        even_udiag[k] = 1.
        even_uudiag[k] = -h_tilde/12
        even_rhs[k] = e_sl
        # first grid levels above delta_sl:
        even_diag[k+1] = 5*(self.h_half[k+1]+h_tilde)/12/self.dt + \
                Ke_full[k+1]/ h_tilde + \
                Ke_full[k+1] / self.h_half[k+1]
        even_lldiag[k] = h_tilde/12/self.dt - Ke_full[k]/h_tilde
        even_rhs[k+1] = 1./self.dt * (h_tilde* dz_tke[k]/12 \
                + 5*(self.h_half[k+1]+h_tilde)* dz_tke[k+1]/12 \
                + self.h_half[k+1]* dz_tke[k+2]/12) \
                + np.diff(shear_half - buoy_half*(1-PATANKAR))[k]

        # inside the surface layer: (partial_z e)_m = 0
        # we include even_ldiag[k-1] as the bd cond
        # requires even_ldiag[k-1]=even_lldiag[k-1]=0 if k>0
        even_ldiag[:k] = even_lldiag[:k] = 0.
        even_udiag[:k] = even_uudiag[:k] = even_rhs[:k] = 0.
        even_diag[:k] = 1.

        diag = interlace(even_diag, odd_diag)
        rhs = interlace(even_rhs, odd_rhs)
        udiag = interlace(even_udiag, odd_udiag)
        uudiag = interlace(even_uudiag, odd_uudiag)
        # for ldiag and lldiag the first index is shifted
        # so it begins with odd for ldiag (and not for lldiag)
        ldiag = interlace(odd_ldiag, even_ldiag)
        lldiag = interlace(even_lldiag, odd_lldiag)

        tke, dz_tke = deinterlace(\
                solve_linear((lldiag, ldiag, diag, udiag, uudiag),
                                rhs))

        tke_full = self.__compute_tke_full(tke, dz_tke, u_star,
                    delta_sl, k)
        return tke, dz_tke, tke_full

    def __compute_tke_full(self, tke, dz_tke, u_star, delta_sl, k):
        tke_full = np.concatenate((
            [tke[0] - self.h_half[0]*dz_tke[0]*5/12 - \
                    self.h_half[0]*dz_tke[1]/12],
            tke + self.h_half[:-1]*dz_tke[1:]*5/12 + \
                    self.h_half[:-1]*dz_tke[:-1]/12))
        e_sl = np.maximum(u_star**2/np.sqrt(self.C_m*self.c_eps),
                                self.e_min)
        tke_full[:k+1] = e_sl
        tke_full[k+1] = tke[k] + (self.z_full[k+1] - delta_sl) * \
                (dz_tke[k+1]*5 + dz_tke[k])/12
        return tke_full

    def mixing_lengths(self, delta_sl):
        """
            returns the mixing lengths (l_m, l_eps)
            for a given delta_sl.
            l_m is computed with (0.5(l_up^(1/a)+l_down^(1/a)))^a
            with (l_up, l_down) the raw distances to the
            (top, bottom) of the computational domain.
        """
        l_up = np.maximum(self.lm_min, self.z_full[-1] - self.z_full)
        l_down = np.maximum(self.lm_min, self.z_full)
        l_eps = np.abs(np.amin((l_up, l_down), axis=0))
        a = -(np.log(self.c_eps)-3.*np.log(self.C_m)+\
                4.*np.log(self.kappa))/np.log(16.)
        l_m = np.abs((0.5*(l_up**(1/a)+l_down**(1/a)))**a)
        l_m[self.z_full <= delta_sl] = np.maximum(self.lm_min,
                self.kappa * \
                (self.C_m * self.c_eps)**(1/4) / self.C_m * \
                (self.z_full[self.z_full <= delta_sl] + self.z_star))
        return l_m, l_eps

    def reconstruct_FV(self, u_bar: array, phi: array,
            sf_scheme: str="FV pure", delta_sl: float=None):
        """
            spline reconstruction of the FV solution.
            input: discrete representation in FV,
            output: arrays (z, u(z))
            u(xi) = u_bar[m] + (phi[m+1] + phi[m]) * xi/2 +
                    (phi[m+1] - phi[m]) / (2*h_half[m]) *
                        (xi^2 - h_half[m]^2/12)
            where xi = linspace(-h_half[m]/2, h_half[m]/2, 10, endpoint=True)
        """
        xi = [np.linspace(-h/2, h/2, 10) for h in self.h_half[:-1]]
        sub_discrete: List[array] = [u_bar[m] + (phi[m+1] + phi[m]) * xi[m]/2 \
                + (phi[m+1] - phi[m]) / (2 * self.h_half[m]) * \
                (xi[m]**2 - self.h_half[m]**2/12) for m in range(self.M)]
        u_oversampled = np.concatenate(sub_discrete[1:])
        z_oversampled = np.concatenate([np.array(xi[m]) + self.z_half[m]
                                            for m in range(1, self.M)])
        if sf_scheme in {"FV Dirichlet", "FV1", "FV pure"}:
            allxi = [np.array(xi[m]) + self.z_half[m] for m in range(self.M)]
            return np.concatenate(allxi), np.concatenate(sub_discrete)

        if delta_sl is None:
            delta_sl = self.z_half[0] if sf_scheme in\
                    {"FV pure", "FV1"} else self.z_full[1]

        # k1 is the index of the grid level containing delta_sl
        k1: int = bisect.bisect_right(self.z_full[1:], delta_sl)
        prognostic: array = np.concatenate((u_bar[:k1+1], phi[k1:]))

        # getting u(delta), from which we extrapolate the log profile
        z_log: array = np.geomspace(self.z_star, delta_sl, 10)
        func_un, _ = self.dictsf_scheme[sf_scheme]
        u_delta: complex = func_un(prognostic=prognostic, delta_sl=delta_sl)
        u_star: float = self.kappa * np.abs(u_delta) \
                / np.log(1 + delta_sl/self.z_star)

        # log profile is then:
        u_log: complex = u_star/self.kappa * np.log(1+z_log/self.z_star) \
                            * u_delta/np.abs(u_delta)

        k2: int = bisect.bisect_right(z_oversampled, self.z_full[k1+1])

        z_freepart = []
        u_freepart = []

        if sf_scheme in {"FV1 free", "FV2 free"}:
            # between the log profile and the next grid level:
            tilde_h = self.z_full[k1+1] - delta_sl
            assert 0 < tilde_h <= self.h_half[k1]
            xi = np.linspace(-tilde_h/2, tilde_h/2, 10)
            zk = self.z_full[k1]
            tau_sl = delta_sl/self.h_half[k1]*(1+self.z_star/delta_sl - \
                            1/np.log(1+delta_sl/self.z_star) \
                    + (zk - (zk+self.z_star)*np.log(1+zk/self.z_star)) \
                    / (delta_sl * np.log(1+delta_sl/self.z_star)))
            alpha_sl = tilde_h/self.h_half[k1] + tau_sl
            u_tilde = 1/alpha_sl * (u_bar[k1] + tilde_h * tau_sl * \
                    (phi[k1]/3 + phi[k1+1]/6))
            u_freepart = u_tilde + (phi[k1+1] + phi[k1]) * xi/2 \
                    + (phi[k1+1] - phi[k1]) / (2 * tilde_h) * \
                    (xi**2 - tilde_h**2/12)
            z_freepart = delta_sl + xi + tilde_h / 2

        return np.concatenate((z_log, z_freepart, z_oversampled[k2:])), \
                np.concatenate((u_log, u_freepart, u_oversampled[k2:]))


    def __backward_euler(self, Y: Tuple[array, array, array],
                        D: Tuple[array, array, array], c: array, u: array):
        """
            integrates once (self.dt) in time the equation
            (partial_t + if)Yu - dt*Du = c
            The scheme is:
            Y(1+dt*if*gamma) - D) u_np1 = Y u + dt*c + dt*if*(1-gamma)
            with gamma the coefficient of implicitation of Coriolis
        """
        to_inverse: Tuple[array, array, array] = add(s_mult(Y,
                                1 + self.implicit_coriolis * \
                            self.dt * 1j*self.f), s_mult(D, - self.dt))
        return solve_linear(to_inverse,
                (1 - (1 - self.implicit_coriolis) * self.dt*1j*self.f) * \
                                multiply(Y, u) + self.dt*c)

    def __visc_turb_FD(self, u_star, delta_sl,
            turbulence="TKE", u_current=None, old_u=None,
            K_full=None, tke=None):
        """
            Computes the turbulent viscosity on full levels K_full.
            It differs between FD and FV because of the
            shear computation (in the future, other differences
            might appear like the temperature handling).
            returns (K_full, TKE)
        """
        if turbulence =="KPP":
            c_1 = 0.2 # constant for h_cl=c_1*u_star/f
            h_cl: float = self.defaulth_cl if np.abs(self.f) < 1e-10 \
                    else c_1*u_star/self.f
            G_full: array = self.kappa * self.z_full * \
                    (1 - self.z_full/h_cl)**2 \
                    * np.heaviside(1 - self.z_full/h_cl, 1)

            K_full: array = u_star * G_full + self.K_mol # viscosity
            K_full[0] = u_star * self.kappa * (delta_sl + self.z_star)
        elif turbulence == "TKE":
            ####### TKE SCHEME #############
            l_m, _ = self.mixing_lengths(delta_sl)
            du = np.diff(u_current)
            du_old = np.diff(old_u)
            shear = np.concatenate(([0],
                [np.abs(K_full[m]/self.h_full[m]**2 * du[m-1] * \
                        (du[m-1]+du_old[m-1])/2) \
                        for m in range(1, self.M)], [0]))

            tke[:] = np.maximum(self.e_min,
                    self.__integrate_tke(tke, shear, K_full,
                    delta_sl=delta_sl, u_star=u_star))

            K_full: array = np.maximum(self.K_min,
                    self.C_m * l_m * np.sqrt(tke))
        else:
            raise NotImplementedError("Wrong turbulence scheme")
        return K_full, tke

    def __matrices_u_FD(self, K_full, forcing):
        """
            Creates the matrices D, Y, c such that the
            semi-discrete in space Ekman equation writes
            (d/dt Y - D) u = c
        """
        D_diag: array = np.concatenate(( [0.],
            [(-K_full[m+1]/self.h_full[m+1] - K_full[m]/self.h_full[m]) \
                    / self.h_half[m] for m in range(1, self.M-1)],
            [-K_full[self.M-1]/self.h_half[self.M-1]/self.h_full[self.M-1]]))
        D_udiag: array = np.concatenate(([0.],
            [K_full[m+1]/self.h_full[m+1] / self.h_half[m]
                for m in range(1, self.M-1)]))
        D_ldiag: array = np.concatenate((
            [K_full[m]/self.h_full[m] / self.h_half[m]
                for m in range(1, self.M-1)]
            , [K_full[self.M-1]/self.h_half[self.M-1]/self.h_full[self.M-1]]))

        c: array = forcing
        c[-1] += K_full[self.M] * 0. / self.h_half[self.M-1]
        Y = (np.zeros(self.M-1), np.ones(self.M), np.zeros(self.M-1))
        D = D_ldiag, D_diag, D_udiag
        return Y, D, c

    def __visc_turb_FV(self, u_star, delta_sl, dz_tke=None,
            turbulence="TKE", phi=None, old_phi=None,
            K_full=None, tke=None):
        """
            Computes the turbulent viscosity on full levels K_full.
            It differs between FD and FV because of the
            shear computation (in the future, other differences
            might appear like the temperature handling).
            returns (K_full, TKE).
            turbulence is either "TKE" or "KPP"
        """
        if turbulence == "KPP":
            c_1 = 0.2 # constant for h_cl in atmosphere
            h_cl: float = self.defaulth_cl if np.abs(self.f) < 1e-10 \
                    else np.abs(c_1*u_star/self.f)
            G_full: array = self.kappa * self.z_full * \
                    (1 - self.z_full/h_cl)**2 \
                    * np.heaviside(1 - self.z_full/h_cl, 1)
            K_full = u_star * G_full + self.K_mol # viscosity
            K_full[0] = u_star * self.kappa * (delta_sl+self.z_star)
        elif turbulence == "TKE":
            k = bisect.bisect_right(self.z_full[1:], delta_sl)
            shear = np.concatenate(([0],
                [np.abs(K_full[m]*phi[m]*(phi[m] + old_phi[m])/2) \
                        for m in range(1, self.M)], [0]))

            tke[:], dz_tke[:], tke_full = \
                self.__integrate_tke_FV(tke, dz_tke, shear,
                            K_full, delta_sl=delta_sl, u_star=u_star)

            if (tke_full < self.e_min).any() or \
                    (tke < self.e_min).any():
                tke_full = np.maximum(tke_full, self.e_min)
                tke = np.maximum(tke, self.e_min)
                dz_tke = self.__compute_dz_tke(tke_full[0],
                                    tke, delta_sl, k)

            l_m, _ = self.mixing_lengths(delta_sl)
            K_full = self.C_m * l_m * np.sqrt(tke_full)
            K_full[:k] =  self.kappa*u_star*(
                    self.z_full[:k] + self.z_star)
            K_full[k] = self.kappa*u_star*(delta_sl + self.z_star)
        else:
            raise NotImplementedError("Wrong turbulence scheme")

        return K_full, tke, dz_tke

    def __compute_dz_tke(self, e_sl, tke, delta_sl, k):
        """ solving the system of finite volumes:
        phi_{m-1}/12 + 10 phi_m / 12 + phi_{m+1} / 12 =
                (tke_{m+1/2} - tke_{m-1/2})/h
        """
        ldiag = self.h_half[:-2] / 12.
        diag = self.h_full[1:-1] * 10./12.
        udiag = self.h_half[1:-1] / 12.
        h_tilde = self.z_full[k+1] - delta_sl
        diag = np.concatenate(([h_tilde*5/12], diag, [1.]))
        udiag = np.concatenate(([h_tilde*1/12], udiag))
        ldiag = np.concatenate((ldiag, [0.]))
        rhs = np.concatenate(([tke[0]-e_sl],
            np.diff(tke), [0.]))
        old_diag, old_udiag, old_ldiag, old_rhs = diag, udiag, ldiag, rhs
        # GRID LEVEL k-1 AND BELOW: dz_tke=0:
        diag[:k], udiag[:k] = 1., 0.
        rhs[:k] = 0.
        # GRID LEVEL k : tke(z=delta_sl) = e_sl
        ldiag[:k] = 0. # ldiag[k-1] is for cell k
        diag[k], udiag[k] = h_tilde*5/12., h_tilde/12.
        rhs[k] = tke[k] - e_sl
        if k == 0:
            assert np.linalg.norm(old_diag - diag) == 0.
            assert np.linalg.norm(old_udiag - udiag) == 0.
            assert np.linalg.norm(old_ldiag - ldiag) == 0.
            assert np.linalg.norm(old_rhs - rhs) == 0.
        # GRID LEVEL k+1: h_tilde used in continuity equation
        ldiag[k] = h_tilde / 12.
        diag[k+1] = (h_tilde+self.h_half[k+1]) * 5./12.
        return solve_linear((ldiag, diag, udiag), rhs)

    def __matrices_u_FV(self, K_full, forcing):
        """
            Creates the matrices D, Y, c such that the
            semi-discrete in space Ekman equation writes
            (d/dt Y - D) phi = c
        """
        Y_diag: array = np.concatenate(([0, 0.], 2/3*np.ones(self.M-1),
                                                        [1.]))
        Y_udiag: array = np.concatenate(([0, 0.],
            [self.h_half[m]/6./self.h_full[m] for m in range(1, self.M)]))
        Y_ldiag: array =  np.concatenate(([0.],
            [self.h_half[m-1]/6./self.h_full[m] for m in range(1, self.M)],
            [0.]))

        D_diag: array = np.concatenate(([0., 0.],
            [-2 * K_full[m] / self.h_half[m] / self.h_half[m-1]
                                        for m in range(1, self.M)],
            [0.]))
        D_udiag: array = np.concatenate(([0.,  0],
            [K_full[m+1]/self.h_full[m] / self.h_half[m]
                                        for m in range(1, self.M)]))
        D_uudiag: array = np.zeros(self.M)

        D_ldiag: array = np.concatenate(([0.],
            [K_full[m-1]/self.h_full[m] / self.h_half[m-1]
                                        for m in range(1, self.M)],
            [0.]))


        c_T = 0. / K_full[self.M]
        c = np.concatenate(([0, 0], np.diff(forcing) / \
                                            self.h_full[1:-1], [c_T]))
        Y = (Y_ldiag, Y_diag, Y_udiag)
        D = (D_ldiag, D_diag, D_udiag, D_uudiag)
        return Y, D, c

    def __apply_sf_scheme(self, sf_scheme, Y, D, c, K_full,
            forcing, u_star, u_delta, delta_sl):
        """
            Changes matrices Y, D, c on the first levels
            to use the surface flux scheme.
        """
        _, func_YDc_sf = self.dictsf_scheme[sf_scheme]
        Y_sf, D_sf, c_sf = func_YDc_sf(K=K_full,
                forcing=forcing, ustar=u_star,
                un=u_delta, delta_sl=delta_sl)
        for y, y_sf in zip(Y, Y_sf):
            y_sf = np.array(y_sf)
            y[:y_sf.shape[0]] = y_sf
        for d, d_sf in zip(D, D_sf):
            d_sf = np.array(d_sf)
            d[:d_sf.shape[0]] = d_sf

        c_sf = np.array(c_sf)
        c[:c_sf.shape[0]] = c_sf

    ####### DEFINITION OF SF SCHEMES : VALUE OF u(delta_sl) #####
    # The method must use the prognostic variables and delta_sl
    # to return u(delta_sl).
    # the prognostic variables are u for FD and 
    # (u_{1/2}, ... u_{k+1/2}, phi_k, ...phi_M) for FV.

    def __sf_udelta_FDDirichlet0(self, prognostic, delta_sl, **_):
        k = bisect.bisect_right(self.z_full[1:], delta_sl)
        return prognostic[k]

    def __sf_udelta_FDpure(self, prognostic, **_):
        return prognostic[0]

    def __sf_udelta_FD2(self, prognostic, **_):
        return (prognostic[0] + prognostic[1])/2

    def __sf_udelta_FVDirichlet0(self, prognostic,
            delta_sl, u, phi, **_):
        k = bisect.bisect_right(self.z_full[1:], delta_sl)
        return u[k] - self.h_half[k]* \
                (phi[k+1] - phi[k])/24

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
        h = self.h_half[0]
        tau_sl = (delta_sl+self.z_star)/h - \
                delta_sl / h / np.log(1+delta_sl/self.z_star)
        return (prognostic[0] - tilde_h * tilde_h/h * \
                (prognostic[2]/6 + prognostic[1]/3)) \
                /(tilde_h/h + tau_sl)

    def __sf_udelta_FV2free(self, prognostic, delta_sl, **_):
        k = bisect.bisect_right(self.z_full[1:], delta_sl)
        zk = self.z_full[k]
        tilde_h = self.z_full[k+1] - delta_sl
        tau_sl = delta_sl/self.h_half[k]*(1+self.z_star/delta_sl - \
                        1/np.log(1+delta_sl/self.z_star) + \
                (zk - (zk+self.z_star)*np.log(1+zk/self.z_star)) \
                / (delta_sl * np.log(1+delta_sl/self.z_star)))
        return (prognostic[k] - tilde_h*tilde_h/self.h_half[k] * \
                (prognostic[k+1] / 3 + prognostic[k+2] / 6)) \
                / (tilde_h/self.h_half[k]+tau_sl)

    ####### DEFINITION OF SF SCHEMES : FIRST LINES OF Y,D,c #####
    # each method must return Y, D, c:
    # Y: 3-tuple of tuples of sizes (j-1, j, j)
    # D: 3-tuple of tuples of sizes (j-1, j, j)
    # c: tuple of size j
    # they represent the first j lines of the matrices.
    # for Y and D, the tuples are (lower diag, diag, upper diag)

    def __sf_YDc_FDDirichlet0(self, **_):
        Y = ((), (0.,), (0.,))
        D = ((), (1+self.h_half[0]/2/self.h_full[1],),
                (-self.h_half[0]/2/self.h_full[1],))
        c = (0.,)
        return Y, D, c

    def __sf_YDc_FDpure(self, K, forcing, ustar, un, **_):
        Y = ((), (1.,), (0.,))
        D = ((), (-K[1]/self.h_full[1]/self.h_half[0] - \
                ustar**2 / np.abs(un)/self.h_half[0], ),
                (K[1]/self.h_full[1]/self.h_half[0],))
        c = (forcing[0],)
        return Y, D, c

    def __sf_YDc_FD2(self, K, ustar, un, **_):
        Y = ((), (0.,), (0.,))
        D = ((), (-K[1]/self.h_full[1] - ustar**2 / np.abs(un) / 2,),
                (K[1]/self.h_full[1] - ustar**2 / np.abs(un) / 2,))
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
                (-K[0]*np.abs(un)/ustar**2+self.h_half[0]/24,
                    K[1]/self.h_half[0]),
                (-self.h_half[0]/24,))
        c = (0., forcing[0])
        return Y, D, c

    def __sf_YDc_FV1(self, K, forcing, ustar, un, **_):
        Y = ((1.,), (0., 0.), (0., 0.))
        D = ((0.,), (1, -K[0] / self.h_half[0]),
                (-K[0]*np.abs(un)/ustar**2, K[1]/self.h_half[0]))
        c = (0., forcing[0])
        return Y, D, c

    def __sf_YDc_FV2(self, K, forcing, ustar, un, **_):
        f = lambda z: (z+self.z_star)*np.log(1+z/self.z_star) - z
        ratio_norms = (f(self.z_full[1]) - f(0)) / \
                (f(self.z_full[2]) - f(self.z_full[1]))
        Y = ((0., 1.), (0., 0., 0.), (0., 0., 0.))
        D = ((0., 0.), (1., 1., -K[1] / self.h_half[1]),
                (-ratio_norms, -K[1]*np.abs(un)/ustar**2 - \
                        self.h_half[1]/3, K[2]/self.h_half[1]),
                (0., -self.h_half[1]/6))
        c = (0., 0., forcing[1])
        return Y, D, c

    def __sf_YDc_FVfree(self, K, forcing, ustar, un, delta_sl, **_):
        tilde_h = self.z_full[1] - delta_sl
        tau_sl = (delta_sl+self.z_star)/self.h_half[0] - \
                delta_sl / self.h_half[0] / np.log(1+delta_sl/self.z_star)
        alpha_sl = tilde_h/self.h_half[0] + tau_sl
        Y = ((1, tilde_h / 6 / self.h_full[1]),
                (0., tilde_h*tau_sl/3, tilde_h/3/self.h_full[1] + \
                            self.h_half[1]/3/self.h_full[1]),
                (0., tilde_h*tau_sl/6,
                    self.h_half[1]/6/self.h_full[1]))

        D = ((0., K[0]/tilde_h/self.h_full[1]),
                (-1., -K[0]*alpha_sl / tilde_h,
                    -K[1]/tilde_h/self.h_full[1] - \
                            K[1] / self.h_half[1] / self.h_full[1]),
                (K[0]*alpha_sl*np.abs(un)/ustar**2 + \
                        tilde_h**2/3/self.h_half[0],
                    K[1]*alpha_sl/tilde_h, K[2]/self.h_full[1]/self.h_half[1]),
                (tilde_h**2/6/self.h_half[0], 0.))
        c = (0., forcing[0]*alpha_sl,
                (forcing[1] - forcing[0])/self.h_full[1])
        return Y, D, c

    def __sf_YDc_FV2free(self, K, forcing, ustar, un, delta_sl, **_):
        k = bisect.bisect_right(self.z_full[1:], delta_sl)
        zk = self.z_full[k]
        tilde_h = self.z_full[k+1] - delta_sl
        tau_sl = delta_sl/self.h_half[k]*(1+self.z_star/delta_sl - \
                        1/np.log(1+delta_sl/self.z_star) + \
                (zk - (zk+self.z_star)*np.log(1+zk/self.z_star)) \
                / (delta_sl * np.log(1+delta_sl/self.z_star)))
        alpha_sl = tilde_h/self.h_half[k] + tau_sl

        Y = ((1, tilde_h / 6 / self.h_full[k+1]),
                (0., tilde_h*tau_sl/3,
                    tilde_h/3/self.h_full[k+1] + \
                            self.h_half[k+1]/3/self.h_full[k+1]),
                (0., tilde_h*tau_sl/6,
                    self.h_half[k+1]/6/self.h_full[k+1]))

        D = ((0., K[k+0]/tilde_h/self.h_full[k+1]),
                (-1., -alpha_sl*K[k+0] / tilde_h,
                    -K[k+1]/tilde_h/self.h_full[k+1] - \
                            K[k+1] / self.h_half[k+1] / self.h_full[k+1]),
                (K[k+0]*np.abs(un)*alpha_sl/ustar**2 + \
                        tilde_h**2 / 3 / self.h_half[k],
                    alpha_sl * K[k+1]/tilde_h, K[k+2]/self.h_full[k+1]/self.h_half[k+1]),
                (tilde_h**2 / 6 / self.h_half[k], 0.))
        c = (0., forcing[k+0]*alpha_sl,
                (forcing[k+1] - forcing[k+0])/self.h_full[k+1])
        Y = (np.concatenate((np.zeros(k), y)) for y in Y)
        f = lambda z: (z+self.z_star)*np.log(1+z/self.z_star) - z
        ratio_norms = [(f(self.z_full[m+1]) - f(self.z_full[m])) / \
                (f(self.z_full[m+2]) - f(self.z_full[m+1])) \
                    for m in range(k)]
        D = (np.concatenate((np.zeros(k), D[0])),
                np.concatenate((-np.ones(k), D[1])),
                np.concatenate((ratio_norms, D[2])),
                np.concatenate((np.zeros(k), D[3])))
        c = np.concatenate((np.zeros(k), c))
        return Y, D, c

