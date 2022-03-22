"""
    This module defines the class TkeOcean1D which
    contains everything for the turbulent kinetic energy.
"""
from typing import List
import bisect
import numpy as np
import ocean1DStratified as oce1D
from utils_linalg import solve_linear, full_to_half

coeff_FV_big = 1/3. # 1/3 or 5/12
coeff_FV_small = 1/6. # 1/6 or 1/12
array = np.ndarray

class TkeOcean1D:
    """
        Integrates the TKE in time with self.integrate_tke(),
        yields the TKE with self.tke_full.
    """
    def __init__(self, M, discretization="FV", TEST_CASE: int=0):
        """
            M: number of cells
            discretization: "FV" or "FD"
        """
        self.e_min: float = 1e-6 # min value of tke
        self.e0_min: float = 1e-4 # min value of surface tke
        self.M: int = M # number of cells
        self.tke_full: array = np.ones(M+1) * self.e_min
        self.tke: array = np.ones(M) * self.e_min
        self.dz_tke: array = np.zeros(M+1) * self.e_min
        assert discretization in {"FV", "FD"}
        self.discretization: str = discretization
        self.Patankar = False
        self.TEST_CASE = TEST_CASE

    def integrate_tke(self, ocean: oce1D.Ocean1dStratified,
            SL: oce1D.SurfaceLayerData,
            universal_funcs,
            shear: array, K_full: array, l_eps: array,
            Ktheta_full: array, N2_full: array, tau_m: float,
            tau_b: float=0.):
        """
            integrates TKE equation on one time step.
            discretization of TKE is defined by self.discretization.
            Ocean is the Ocean1dStratified instance
                which calls this method.
            shear is K_u ||dz u|| at full levels.
            K_full is K_u (at full levels)
            l_eps is the tke mixing length (at full levels)
            Ktheta_full is K_theta (at full levels)
            N2_full is the BF frequency (at full levels).
        """
        if self.discretization == "FD":
            self.__integrate_tke_FD(ocean, shear, K_full, l_eps,
            Ktheta_full, N2_full, tau_m, tau_b)
        elif self.discretization == "FV":
            buoy_half = full_to_half(Ktheta_full*N2_full)
            shear_half = full_to_half(shear)
            self.__integrate_tke_FV(ocean, shear_half, K_full,
                SL, l_eps, buoy_half, True,
                universal_funcs, tau_m, tau_b)

    def __integrate_tke_FD(self, ocean: oce1D.Ocean1dStratified,
            shear: array, K_full: array, l_eps: array,
            Ktheta_full: array, N2: array, tau_m: float,
            tau_b: float=0.):
        """
            integrates TKE equation on one time step.
            discretization of TKE is Finite Differences,
            located on half-points.
        """
        if Ktheta_full is None or N2 is None:
            Ktheta_full = N2 = np.zeros(self.M)
        Ke_half = ocean.C_e / ocean.C_m * (K_full[1:] + K_full[:-1])/2
        diag_e = np.concatenate(([1],
                    [1/ocean.dt + \
                    ocean.c_eps*np.sqrt(self.tke_full[m])/l_eps[m] \
                    + (Ke_half[m]/ocean.h_half[m] + \
                        Ke_half[m-1]/ocean.h_half[m-1]) \
                        / ocean.h_full[m] for m in range(1, self.M)],
                        [1/ocean.dt + Ke_half[self.M-1] / \
                        ocean.h_half[self.M-1] / ocean.h_full[self.M]]))
        ldiag_e = np.concatenate((
            [ -Ke_half[m-1] / ocean.h_half[m-1] / ocean.h_full[m] \
                    for m in range(1,self.M) ], [- Ke_half[self.M-1] / \
                        ocean.h_half[self.M-1] / ocean.h_full[self.M]]))
        udiag_e = np.concatenate(([0],
            [ -Ke_half[m] / ocean.h_half[m] / ocean.h_full[m] \
                    for m in range(1,self.M) ]))

        ebb = 67.83
        if self.TEST_CASE > 0:
            e_sl = max(self.e0_min, ebb*np.abs(tau_m/ocean.rho0))
        e_bottom = max(self.e_min, ebb*tau_b)

        rhs_e = np.concatenate(([e_bottom],
            [self.tke_full[m]/ocean.dt + shear[m]
                for m in range(1, self.M)], [e_sl]))
        for m in range(1, self.M): # Patankar trick ?
            if shear[m] <= Ktheta_full[m] * N2[m] and self.Patankar:
                diag_e[m] += Ktheta_full[m] * N2[m] / self.tke_full[m]
            else: # normal handling of buoyancy
                rhs_e[m] -=  Ktheta_full[m] * N2[m]
        ldiag_e[-1] = 0.
        diag_e[-1] = 1.

        self.tke_full = np.maximum(self.e_min,
                solve_linear((ldiag_e, diag_e, udiag_e), rhs_e))
        return self.tke_full

    def __integrate_tke_FV(self, ocean: oce1D.Ocean1dStratified,
            shear_half: array, K_full: array,
            SL: oce1D.SurfaceLayerData, l_eps: array,
            buoy_half: array, ignore_sl: bool,
            universal_funcs, tau_m: float, tau_b: float=0.):
        """
            integrates TKE equation on one time step.
            discretization of TKE is Finite Volumes,
            centered on half-points.
            tke is the array of averages,
            dz_tke is the array of space derivatives.
            shear is K_u ||dz u|| at half levels.
            K_full is K_u (at full levels)
            l_eps is the tke mixing length (at full levels)
            Ktheta_full is K_theta (at full levels)
            buoy_half is the buoyancy (Ktheta*N2) (at half levels).

            WARNING DELTA_SL IS NOT SUPPOSED TO MOVE !!
            to do this, e_sl of the previous time should be known.
            (otherwise a discontinuity between the
            cell containing the SL's top and the one above
            may appear.)
        """
        # deciding in which cells we use Patankar's trick
        PATANKAR = (shear_half <= buoy_half) if self.Patankar else \
                np.zeros_like(shear_half)
        # Turbulent viscosity of tke :
        Ke_full = K_full * ocean.C_e / ocean.C_m
        # Bottom value:
        phi_m, *_ = universal_funcs
        k = SL.k
        z_sl = np.copy(ocean.z_full[k:])
        z_sl[0] = z_sl[0] if ignore_sl else SL.delta_sl
        # buoyancy and shear in surface layer:
        KN2_sl = 9.81/283. * SL.t_star * SL.u_star
        shear_sl = SL.u_star**3 * phi_m(z_sl * SL.inv_L_MO) / \
                ocean.kappa / (-z_sl + SL.z_0M)
        e_sl = np.maximum(((l_eps[k:]/ocean.c_eps * \
                (shear_sl - KN2_sl))**2)**(1/3), ocean.e_min)
        ebb = 67.83
        if self.TEST_CASE > 0:
            e_sl = np.maximum(self.e0_min,
                    ebb*np.abs(tau_m/ocean.rho0)*np.ones_like(e_sl))
        e_top = e_sl[0]
        e_bottom = np.maximum(self.e_min, ebb*tau_b)

        l_eps_half = full_to_half(l_eps)

        h_tilde = SL.delta_sl - ocean.z_full[k-1]
        if ignore_sl:
            h_tilde = ocean.h_half[k-1]

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
        odd_ldiag = Ke_full[:-1] / ocean.h_half[:-1]
        odd_udiag = -Ke_full[1:] / ocean.h_half[:-1]
        odd_diag = ocean.c_eps * np.sqrt(self.tke) / l_eps_half + \
                PATANKAR * buoy_half / self.tke + 1/ocean.dt
        odd_rhs = shear_half - (1 - PATANKAR)* buoy_half + self.tke/ocean.dt
        odd_lldiag = np.zeros(odd_ldiag.shape[0] - 1)
        odd_uudiag = np.zeros(odd_ldiag.shape[0] - 1)
        # inside the surface layer:
        odd_ldiag[k:] = 0.
        odd_udiag[k:] = 0.
        odd_ldiag[k-1] = Ke_full[k-1] / h_tilde
        odd_udiag[k-1] = -Ke_full[k] / h_tilde
        odd_rhs[k:] = e_sl[:self.M-k]
        odd_diag[k:] = 1.

        ####### EVEN LINES: evolution of dz_tke = (partial_z e)
        # data located at m, 0<m<M
        # notice that even_{l, ll}diag index is shifted.
        even_diag = np.concatenate(([-coeff_FV_big*ocean.h_half[0]],
                2*coeff_FV_big*ocean.h_full[1:-1] / ocean.dt + \
                Ke_full[1:-1]/ocean.h_half[1:-1] + \
                Ke_full[1:-1] / ocean.h_half[:-2],
                [0.] # will be overriden
            ))
        even_udiag = np.concatenate(([1],
                buoy_half[1:]*PATANKAR[1:] / self.tke[1:] + \
                ocean.c_eps * np.sqrt(self.tke[1:]) / l_eps_half[1:]))
        even_uudiag = np.concatenate(([-coeff_FV_small*ocean.h_half[0]],
                ocean.h_half[1:-1]*coeff_FV_small/ocean.dt - \
                Ke_full[2:]/ocean.h_half[1:-1]))

        even_ldiag = np.concatenate((
                - buoy_half[:-1]*PATANKAR[:-1] / self.tke[:-1] - \
                ocean.c_eps * np.sqrt(self.tke[:-1]) / l_eps_half[:-1],
                [0])) # will be overriden
        even_lldiag = np.concatenate((
                ocean.h_half[:-2]*coeff_FV_small/ocean.dt - \
                Ke_full[:-2]/ocean.h_half[:-2],
                [0] # will be overriden
                ))

        even_rhs = np.concatenate(( [e_bottom],
                1/ocean.dt * (ocean.h_half[:-2]* self.dz_tke[:-2]*coeff_FV_small \
                + 2*ocean.h_full[1:-1]* self.dz_tke[1:-1]*coeff_FV_big \
                + ocean.h_half[1:-1]* self.dz_tke[2:]*coeff_FV_small) \
                + np.diff(shear_half - buoy_half*(1-PATANKAR)),
                [0])) # will be overriden
        # e(delta_sl) = e_top
        even_diag[k] = h_tilde*coeff_FV_big
        even_ldiag[k-1] = 1.
        even_lldiag[k-1] = h_tilde*coeff_FV_small
        even_rhs[k] = e_top
        # first grid levels below delta_sl:
        even_diag[k-1] = (ocean.h_half[k-2]+h_tilde)*coeff_FV_big/ocean.dt + \
                Ke_full[k-1]/ h_tilde + \
                Ke_full[k-1] / ocean.h_half[k-2]
        even_uudiag[k-1] = h_tilde*coeff_FV_small/ocean.dt - Ke_full[k]/h_tilde
        even_rhs[k-1] = (h_tilde* self.dz_tke[k]*coeff_FV_small \
                +(ocean.h_half[k-2]+h_tilde)* self.dz_tke[k-1]*coeff_FV_big \
                +ocean.h_half[k-2]*self.dz_tke[k-2]*coeff_FV_small)/ocean.dt \
                + np.diff(shear_half - buoy_half*(1-PATANKAR))[k-2]

        # inside the surface layer: (partial_z e)_m = 0
        # we include even_ldiag[k-1] because if k>0
        # the bd cond requires even_ldiag[k-1]=even_lldiag[k-1]=0
        even_ldiag[k:] = even_lldiag[k:] = 0.
        even_udiag[k:] = even_uudiag[k:] = even_rhs[k+1:] = 0.
        even_diag[k+1:] = 1.

        diag = interlace(even_diag, odd_diag)
        rhs = interlace(even_rhs, odd_rhs)
        udiag = interlace(even_udiag, odd_udiag)
        uudiag = interlace(even_uudiag, odd_uudiag)
        # for ldiag and lldiag the first index is shifted
        # so it begins with odd for ldiag (and not for lldiag)
        ldiag = interlace(odd_ldiag, even_ldiag)
        lldiag = interlace(even_lldiag, odd_lldiag)

        self.tke, self.dz_tke = deinterlace(\
                solve_linear((lldiag, ldiag, diag, udiag, uudiag),
                                rhs))

        if (self.tke < self.e_min).any():
            self.tke = np.maximum(self.tke, self.e_min)
            self.__compute_dz_tke(ocean, e_top,
                    SL.delta_sl, k, ignore_sl)

        self.tke_full = np.maximum(self.__compute_tke_full_FV(ocean,
            SL, ignore_sl, l_eps, universal_funcs), self.e_min)
        return self.tke_full

    def __compute_dz_tke(self, ocean: oce1D.Ocean1dStratified,
            e_sl: float, delta_sl: float, k: int, ignore_sl: bool):
        """ solving the system of finite volumes:
                phi_{m-1}/12 + 10 phi_m / 12 + phi_{m+1} / 12 =
                    (tke_{m+1/2} - tke_{m-1/2})/h
            The coefficients 1/12, 5/12 are replaced
                by coeff_FV_small, coeff_FV_big.
        """
        ldiag = ocean.h_half[:-2] *coeff_FV_small
        diag = ocean.h_full[1:-1] * 2*coeff_FV_big
        udiag = ocean.h_half[1:-1] *coeff_FV_small
        h_tilde = delta_sl - ocean.z_full[k-1]
        if ignore_sl: # If the tke is reconstructed until z=0:
            assert k == self.M
            h_tilde = ocean.h_half[k-1]
        diag = np.concatenate(([1.], diag, [-h_tilde*coeff_FV_big]))
        udiag = np.concatenate(([0.], udiag))
        ldiag = np.concatenate((ldiag, [-h_tilde*coeff_FV_small]))
        rhs = np.concatenate(([0.],
            np.diff(self.tke), [self.tke[-1]-e_sl]))

        # GRID LEVEL k+1 AND ABOVE: dz_tke=0:
        diag[k+1:], ldiag[k:] = 1., 0.
        rhs[k+1:] = 0.
        udiag[k:] = 0. # udiag[k] is for cell k:
        # GRID level k: tke(z=delta_sl) = e_sl
        diag[k] = -h_tilde*coeff_FV_big
        ldiag[k-1] = -h_tilde*coeff_FV_small
        rhs[k] = self.tke[k-1] - e_sl
        # GRID LEVEL k-1: h_tilde used in continuity equation
        udiag[k-1] = h_tilde *coeff_FV_small
        diag[k-1] = (h_tilde+ocean.h_half[k-2]) * coeff_FV_big
        self.dz_tke = solve_linear((ldiag, diag, udiag), rhs)

    def __compute_tke_full_FV(self, ocean: oce1D.Ocean1dStratified,
            SL: oce1D.SurfaceLayerData, ignore_sl: bool,
            l_eps: array, universal_funcs):
        """
            projection of the tke reconstruction on z_full
            if ignore_sl is False, the closest z level below
            delta_sl is replaced with delta_sl.
        """
        h_half, k = np.copy(ocean.h_half), SL.k
        z_sl = np.copy(ocean.z_full[k+1:])
        if not ignore_sl:
            h_half[k-1] = SL.delta_sl - ocean.z_full[k-1]

        phi_m, *_ = universal_funcs

        KN2 = 9.81/283. * SL.t_star * SL.u_star
        shear = SL.u_star**3 * phi_m(z_sl* SL.inv_L_MO) / \
                ocean.kappa / (-z_sl + SL.z_0M)
        # TKE inside the surface layer:
        tke_sl = np.maximum(((l_eps[k+1:]/ocean.c_eps * \
                (shear - KN2))**2)**(1/3), self.e_min)
        # TKE at the bottom of the surface layer:
        tke_k = [self.tke[k-1] + h_half[k-1]*self.dz_tke[k-1]*coeff_FV_small + \
                    h_half[k-1]*self.dz_tke[k]*coeff_FV_big]
        # Combining both with the TKE above the surface layer:
        return np.concatenate((self.tke[:k] - \
                h_half[:k]*self.dz_tke[:k]*coeff_FV_big - \
                    h_half[:k]*self.dz_tke[1:k+1]*coeff_FV_small,
                    tke_k, tke_sl))

    def reconstruct_TKE(self, ocean: oce1D.Ocean1dStratified,
            SL: oce1D.SurfaceLayerData, sf_scheme: str,
            universal_funcs, l_eps: array):
        """
            returns (z, tke(z)), the reconstruction
            of the tke.
            SL and l_eps are outputs of the integration in time.
            universal_funcs can be businger()
        """
        ignore_tke_sl = sf_scheme in {"FV pure", "FV1"}
        z_min = SL.delta_sl / 2.
        # We first compute the reconstruction above the SL:
        xi = [np.linspace(-h/2, h/2, 15) for h in ocean.h_half[:-1]]
        xi[1] = np.linspace(-ocean.h_half[1]/2, ocean.h_half[1]/2, 40)
        xi[0] = np.linspace(z_min-ocean.h_half[0]/2, ocean.h_half[0]/2, 40)
        sub_discrete: List[array] = [self.tke[m] + \
                (self.dz_tke[m+1] + self.dz_tke[m]) * xi[m]/2 \
                + (self.dz_tke[m+1] - self.dz_tke[m]) / (2 * ocean.h_half[m]) * \
                (xi[m]**2 - ocean.h_half[m]**2/12) \
                for m in range(self.M)]

        # if the coefficients are 1/12, 5/12:
        if abs(coeff_FV_small - 1./12) < 1e-4:
            assert abs(coeff_FV_big - 5./12) < 1e-4
            coefficients = 1. / 32. * np.array((
                (60, -1, 1, -14, -14),
                (0,-8, -8, -48, 48),
                (-480, 24, -24, 240, 240),
                (0, 32, 32, 64, -64),
                (960, -80, 80, -480, -480)))
            for m in range(self.M):
                h = ocean.h_half[m]
                values = np.array((self.tke[m], h*self.dz_tke[m],
                    h*self.dz_tke[m+1], \
                    self.tke[m] - 5/12*h*self.dz_tke[m] - h*self.dz_tke[m+1]/12,
                    self.tke[m] + h*self.dz_tke[m]/12 + h*self.dz_tke[m+1]*5/12))
                polynomial = coefficients @ values
                sub_discrete[m] = np.sum([(rihi/h**i) * xi[m]**i\
                    for i, rihi in enumerate(polynomial)], axis=0)

        # If the SL is not handled, we can stop now:
        if ignore_tke_sl:
            tke_oversampled = np.concatenate(sub_discrete)
            z_oversampled = np.concatenate([np.array(xi[m]) + \
                    ocean.z_half[m] for m in range(self.M)])

            return z_oversampled, tke_oversampled
        tke_oversampled = np.concatenate(sub_discrete[1:])
        z_oversampled = np.concatenate([np.array(xi[m]) + \
                ocean.z_half[m] for m in range(1, self.M)])
        # The part above the SL starts at z_oversampled[k2:].
        k2: int = bisect.bisect_right(z_oversampled,
                ocean.z_full[SL.k+1])

        # We now compute the reconstruction inside the SL:
        phi_m, *_ = universal_funcs
        z_log: array = np.array((z_min, SL.delta_sl))
        # buoyancy and shear inside the SL:
        KN2 = 9.81/283. * SL.t_star * SL.u_star
        shear = SL.u_star**3 * phi_m(z_log * SL.inv_L_MO) / \
                ocean.kappa / (z_log + SL.z_0M)
        from scipy.interpolate import interp1d
        z_levels = np.copy(ocean.z_full)
        if not ignore_tke_sl:
            z_levels[SL.k] = SL.delta_sl
        f = interp1d(z_levels, l_eps, fill_value="extrapolate")
        l_eps_log = np.maximum(f(z_log), 0.)
        # With the quasi-equilibrium hypothesis, we get the tke:
        tke_log = np.maximum(((l_eps_log/ocean.c_eps * \
                (shear - KN2))**2)**(1/3), ocean.e_min)

        # We finally compute the reconstruction just above SL:
        z_freepart = [] # we call "freepart" the zone
        k = SL.k # between the log profile and the next grid level:
        tilde_h = ocean.z_full[k+1] - SL.delta_sl
        assert 0 < tilde_h <= ocean.h_half[k]
        xi = np.linspace(-tilde_h/2, tilde_h/2, 15)

        tke_freepart = self.tke[k] + (self.dz_tke[k+1] + self.dz_tke[k]) * xi/2 \
                + (self.dz_tke[k+1] - self.dz_tke[k]) / (2 * tilde_h) * \
                (xi**2 - tilde_h**2/12)

        if abs(coeff_FV_small - 1./12) < 1e-4:
            values = np.array((self.tke[k], tilde_h*self.dz_tke[k],
                tilde_h*self.dz_tke[k+1], \
                self.tke[k] - 5/12*tilde_h*self.dz_tke[k] - tilde_h*self.dz_tke[k+1]/12,
                self.tke[k] + tilde_h*self.dz_tke[k]/12 + tilde_h*self.dz_tke[k+1]*5/12))
            polynomial = coefficients @ values
            tke_freepart = np.sum([(rihi/tilde_h**i) * xi**i\
                for i, rihi in enumerate(polynomial)], axis=0)

        z_freepart = SL.delta_sl + xi + tilde_h / 2
        # we return the concatenation of the 3 zones:
        return np.concatenate((z_log,
                            z_freepart, z_oversampled[k2:])), \
                np.concatenate((tke_log,
                            tke_freepart, tke_oversampled[k2:]))

