"""
    This module defines the class TkeAtm1D which
    contains everything for the turbulent kinetic energy.
"""
import bisect
import numpy as np
import atm1DStratified as atm1D
from utils_linalg import solve_linear, full_to_half
from bulk import SurfaceLayerData

coeff_FV_big = 1/3. # 1/3 or 5/12
coeff_FV_small = 1/6. # 1/6 or 1/12
array = np.ndarray

class TkeAtm1D:
    """
        Integrates the TKE in time with self.integrate_tke(),
        yields the TKE with self.tke_full.
    """
    def __init__(self, atm, discretization="FV", ignore_sl: bool=True,
            Neutral_case: bool=False, SL=None):
        """
            M: number of cells
            discretization: "FV" or "FD"
            ignore_sl: if there's a special treatment of FV free,
                put ignore_sl to False.
            SL: in the stratified case for FV, we need a
                atm1DStratified.SurfaceLayerData for initialization
        """
        self.e_min: float = 1e-6 # min value of tke
        self.e0_min: float = 1e-4 # min value of surface tke
        self.M: int = atm.M # number of cells
        self.tke_full: array = np.ones(atm.M+1) * self.e_min
        self.tke: array = np.ones(atm.M) * self.e_min
        self.dz_tke: array = np.zeros(atm.M+1) * self.e_min
        assert discretization in {"FV", "FD"}
        self.discretization: str = discretization
        self.Patankar = False
        self.ignore_sl = ignore_sl
        self.neutral_case = Neutral_case
        self.__initialize_tke(atm, SL)

    def integrate_tke(self, atm: atm1D.Atm1dStratified,
            SL: SurfaceLayerData,
            universal_funcs,
            shear: array, K_full: array, l_eps: array,
            Ktheta_full: array, N2_full: array):
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
            self.__integrate_tke_FD(atm, shear, K_full, SL, l_eps,
            Ktheta_full, N2_full, universal_funcs)
        elif self.discretization == "FV":
            buoy_half = full_to_half(Ktheta_full*N2_full)
            shear_half = full_to_half(shear)
            self.__integrate_tke_FV(atm, shear_half, K_full,
                SL, l_eps, buoy_half, universal_funcs)

    def __integrate_tke_FV(self, atm, shear_half: array, K_full: array,
            SL: SurfaceLayerData, l_eps: array,
            buoy_half: array, universal_funcs):
        """
            integrates TKE equation on one time step.
            discretisation of TKE is Finite Volumes,
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
            tke is the average value on the whole cell,
            except tke[k] which is the average value on
            the "subcell" [delta_sl, z_{k+1}].
            returns tke, dz_tke
        """
        # deciding in which cells we use Patankar's trick
        PATANKAR = (shear_half <= buoy_half)
        # Turbulent viscosity of tke :
        Ke_full = K_full * atm.C_e / atm.C_m
        # Bottom value:
        phi_m, *_ = universal_funcs
        k = bisect.bisect_right(atm.z_full[1:], SL.delta_sl)
        z_sl = np.copy(atm.z_full[:k+1])
        z_sl[-1] = z_sl[-1] if self.ignore_sl else SL.delta_sl
        # buoyancy and shear in surface layer:
        KN2_sl = 9.81/283. * SL.t_star * SL.u_star
        shear_sl = SL.u_star**3 * phi_m(z_sl * SL.inv_L_MO) / \
                atm.kappa / (z_sl + SL.z_0M)
        e_sl = np.maximum(((l_eps[:k+1]/atm.c_eps * \
                (shear_sl - KN2_sl))**2)**(1/3), atm.e_min)

        l_eps_half = full_to_half(l_eps)

        h_tilde = atm.z_full[k+1] - SL.delta_sl
        if self.ignore_sl:
            h_tilde = atm.h_half[k]

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
        odd_ldiag = Ke_full[:-1] / atm.h_half[:-1]
        odd_udiag = -Ke_full[1:] / atm.h_half[:-1]
        odd_diag = atm.c_eps * np.sqrt(self.tke) / l_eps_half + \
                PATANKAR * buoy_half / self.tke + 1/atm.dt
        odd_rhs = shear_half - (1 - PATANKAR)* buoy_half + \
                self.tke/atm.dt
        odd_lldiag = np.zeros(odd_ldiag.shape[0] - 1)
        odd_uudiag = np.zeros(odd_ldiag.shape[0] - 1)
        # inside the surface layer:
        odd_ldiag[:k] = 0.
        odd_udiag[:k] = 0.
        odd_ldiag[k] = Ke_full[k] / h_tilde
        odd_udiag[k] = -Ke_full[k+1] / h_tilde
        odd_rhs[:k] = e_sl[:k]
        odd_diag[:k] = 1.

        ####### EVEN LINES: evolution of dz_tke = (partial_z e)
        # data located at m, 0<m<M
        # notice that even_{l, ll}diag index is shifted
        even_diag = np.concatenate(([-atm.h_half[0]*coeff_FV_big],
                2*coeff_FV_big*atm.h_full[1:-1] / atm.dt + \
                Ke_full[1:-1]/atm.h_half[1:-1] + \
                Ke_full[1:-1] / atm.h_half[:-2],
                [1]
            ))
        even_udiag = np.concatenate(([1],
                buoy_half[1:]*PATANKAR[1:] / self.tke[1:] + \
                atm.c_eps * np.sqrt(self.tke[1:]) / l_eps_half[1:]))
        even_uudiag = np.concatenate(([-atm.h_half[0]*coeff_FV_small],
                atm.h_half[1:-1]*coeff_FV_small/atm.dt - \
                Ke_full[2:]/atm.h_half[1:-1]))

        even_ldiag = np.concatenate((
                - buoy_half[:-1]*PATANKAR[:-1] / self.tke[:-1] - \
                atm.c_eps * np.sqrt(self.tke[:-1]) / l_eps_half[:-1],
                [0]))
        even_lldiag = np.concatenate((
                atm.h_half[:-2]*coeff_FV_small/atm.dt - \
                Ke_full[:-2]/atm.h_half[:-2],
                [0]
                ))

        even_rhs = np.concatenate(( [e_sl[0]],
                1/atm.dt * (atm.h_half[:-2]* self.dz_tke[:-2]*coeff_FV_small \
                + 2*atm.h_full[1:-1]* self.dz_tke[1:-1]*coeff_FV_big \
                + atm.h_half[1:-1]* self.dz_tke[2:]*coeff_FV_small) \
                + np.diff(shear_half - buoy_half*(1-PATANKAR)),
                [0]))
        # e(delta_sl) = e_sl :
        even_diag[k] = -h_tilde*coeff_FV_big
        even_udiag[k] = 1.
        even_uudiag[k] = -h_tilde*coeff_FV_small
        even_rhs[k] = e_sl[k]
        # first grid levels above delta_sl:
        even_diag[k+1] = (atm.h_half[k+1]+h_tilde)*coeff_FV_big/atm.dt + \
                Ke_full[k+1]/ h_tilde + \
                Ke_full[k+1] / atm.h_half[k+1]
        even_lldiag[k] = h_tilde*coeff_FV_small/atm.dt - Ke_full[k]/h_tilde
        even_rhs[k+1] = (h_tilde* self.dz_tke[k]*coeff_FV_small \
                +(atm.h_half[k+1]+h_tilde)* self.dz_tke[k+1]*coeff_FV_big \
                +atm.h_half[k+1]*self.dz_tke[k+2]*coeff_FV_small)/atm.dt \
                + np.diff(shear_half - buoy_half*(1-PATANKAR))[k]

        # inside the surface layer: (partial_z e)_m = 0
        # we include even_ldiag[k-1] because if k>0
        # the bd cond requires even_ldiag[k-1]=even_lldiag[k-1]=0
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

        self.tke, self.dz_tke = deinterlace(\
                solve_linear((lldiag, ldiag, diag, udiag, uudiag),
                                rhs))

        self.tke_full = self.__compute_tke_full(atm, SL, l_eps,
                universal_funcs)

        if (self.tke_full < self.e_min).any() or \
                (self.tke < self.e_min).any():
            self.tke_full = np.maximum(self.tke_full, self.e_min)
            self.tke = np.maximum(self.tke, self.e_min)
            self.dz_tke = self.__compute_dz_tke(atm, self.tke_full[k],
                                SL.delta_sl, k)
        return

    def __integrate_tke_FD(self, atm, shear: array,
            K_full: array, SL: SurfaceLayerData,
            l_eps: array, Ktheta_full: array,
            N2: array, universal_funcs):
        """
            integrates TKE equation on one time step.
            discretisation of TKE is Finite Differences,
            located on half-points.
        """
        if Ktheta_full is None or N2 is None:
            Ktheta_full = N2 = np.zeros(atm.M)
        Ke_half = atm.C_e / atm.C_m * (K_full[1:] + K_full[:-1])/2
        diag_e = np.concatenate(([1],
                    [1/atm.dt + atm.c_eps*np.sqrt(self.tke[m])/l_eps[m] \
                    + (Ke_half[m]/atm.h_half[m] + \
                        Ke_half[m-1]/atm.h_half[m-1]) \
                        / atm.h_full[m] for m in range(1, atm.M)],
                        [1/atm.dt + Ke_half[atm.M-1] / \
                        atm.h_half[atm.M-1] / atm.h_full[atm.M]]))
        ldiag_e = np.concatenate((
            [ -Ke_half[m-1] / atm.h_half[m-1] / atm.h_full[m] \
                    for m in range(1,atm.M) ], [- Ke_half[atm.M-1] / \
                        atm.h_half[atm.M-1] / atm.h_full[atm.M]]))
        udiag_e = np.concatenate(([0],
            [ -Ke_half[m] / atm.h_half[m] / atm.h_full[m] \
                    for m in range(1,atm.M) ]))

        phi_m, *_ = universal_funcs
        KN2_sl = 9.81/283. * SL.t_star * SL.u_star
        shear_sl = SL.u_star**3 * phi_m(SL.delta_sl * SL.inv_L_MO) / \
                atm.kappa / (SL.delta_sl + SL.z_0M)

        e_sl = np.maximum(((l_eps[SL.k]/atm.c_eps * \
                (shear_sl - KN2_sl))**2)**(1/3), self.e_min)

        rhs_e = np.concatenate(([e_sl], [self.tke[m]/atm.dt + shear[m]
                for m in range(1, atm.M)],
            [self.tke[atm.M]/atm.dt]))
        for m in range(1, atm.M):
            if shear[m] <= Ktheta_full[m] * N2[m]: # Patankar trick
                diag_e[m] += Ktheta_full[m] * N2[m] / self.tke[m]
            else: # normal handling of buoyancy
                rhs_e[m] -=  Ktheta_full[m] * N2[m]
        # if delta_sl is inside the computational domain:
        k = bisect.bisect_right(atm.z_full[1:], SL.delta_sl)
        rhs_e[:k+1] = e_sl # prescribe e=e(delta_sl)
        diag_e[:k+1] = 1. # because lower levels are not used
        udiag_e[:k+1] = ldiag_e[:k] = 0.
        self.tke_full = solve_linear((ldiag_e, diag_e, udiag_e), rhs_e)

    def __compute_tke_full(self, atm, SL: SurfaceLayerData,
            l_eps: array, universal_funcs):
        """
            projection of the tke reconstruction on z_full
            if ignore_sl is False, the closest z level below
            delta_sl is replaced with delta_sl.
        """
        h_half, k = np.copy(atm.h_half), SL.k
        z_sl = np.copy(atm.z_full[:k])
        if not self.ignore_sl:
            h_half[k] = atm.z_full[k+1] - SL.delta_sl

        phi_m, *_ = universal_funcs

        KN2 = 9.81/283. * SL.t_star * SL.u_star
        shear = SL.u_star**3 * phi_m(z_sl* SL.inv_L_MO) / \
                atm.kappa / (z_sl + SL.z_0M)
        # TKE inside the surface layer:
        tke_sl = np.maximum(((l_eps[:k]/atm.c_eps * \
                (shear - KN2))**2)**(1/3), self.e_min)
        # TKE at the top of the surface layer:
        tke_k = [self.tke[k] - h_half[k]*self.dz_tke[k]*coeff_FV_big - \
                    h_half[k]*self.dz_tke[k+1]*coeff_FV_small]
        # Combining both with the TKE above the surface layer:
        return np.concatenate((tke_sl, tke_k,
            self.tke[k:] + h_half[k:-1]*self.dz_tke[k+1:]*coeff_FV_big + \
                    h_half[k:-1]*self.dz_tke[k:-1]*coeff_FV_small))

    def __compute_dz_tke(self, atm, e_sl: float, delta_sl: float,
            k: int):
        """ solving the system of finite volumes:
                phi_{m-1}/12 + 10 phi_m / 12 + phi_{m+1} / 12 =
                    (tke_{m+1/2} - tke_{m-1/2})/h
            The coefficients 1/12, 5/12 are replaced
                by coeff_FV_small, coeff_FV_big.
        """
        ldiag = atm.h_half[:-2] *coeff_FV_small
        diag = atm.h_full[1:-1] * 2*coeff_FV_big
        udiag = atm.h_half[1:-1] *coeff_FV_small
        h_tilde = atm.z_full[k+1] - delta_sl
        if self.ignore_sl: # If the tke is reconstructed until z=0:
            assert k == 0
            h_tilde = atm.h_half[k]
        diag = np.concatenate(([h_tilde*coeff_FV_big], diag, [1.]))
        udiag = np.concatenate(([h_tilde*coeff_FV_small], udiag))
        ldiag = np.concatenate((ldiag, [0.]))
        rhs = np.concatenate(([self.tke[0]-e_sl],
            np.diff(self.tke), [0.]))

        # GRID LEVEL k-1 AND BELOW: dz_tke=0:
        diag[:k], udiag[:k] = 1., 0.
        rhs[:k] = 0.
        ldiag[:k] = 0. # ldiag[k-1] is for cell k
        # GRID level k: tke(z=delta_sl) = e_sl
        diag[k] = h_tilde*coeff_FV_big
        udiag[k] = h_tilde*coeff_FV_small
        rhs[k] = self.tke[k] - e_sl
        # GRID LEVEL k+1: h_tilde used in continuity equation
        ldiag[k] = h_tilde *coeff_FV_small
        diag[k+1] = (h_tilde+atm.h_half[k+1]) * coeff_FV_big
        return solve_linear((ldiag, diag, udiag), rhs)

    def __initialize_tke(self, atm, SL):
        """
        Initialization method for the tke.
        It is not crucial to have a perfect initial
        profile, but dz_tke is computed with care to
        have a continuous profile.
        """
        if self.discretization == "FV":
            self.tke = np.ones(self.M) * self.e_min
            if not self.neutral_case:
                assert SL is not None
                # N2 and inv_L_MO are 0 at initialization
                # so we use a neutral e_sl assuming l_m=l_eps
                e_sl = np.maximum(SL.u_star**2 / \
                        np.sqrt(atm.C_m*atm.c_eps), self.e_min)
                z_half, k = np.copy(atm.z_half), SL.k
                z_half[k] = z_half[k] if self.ignore_sl \
                        else SL.delta_sl
                self.tke[z_half[:-1] <= 250] = self.e_min + e_sl*(1 - \
                    (z_half[:-1][z_half[:-1] <= 250] - \
                    SL.delta_sl) / (250.- SL.delta_sl))**3
                # inversion of a system to find dz_tke:
                self.dz_tke = self.__compute_dz_tke(atm, e_sl,
                        SL.delta_sl, SL.k)
            self.dz_tke = np.zeros(self.M+1)
        elif self.discretization == "FD":
            self.tke = np.ones(self.M+1) * self.e_min
            if not self.neutral_case:
                self.tke[atm.z_half <= 250] = self.e_min + 0.4*(1 - \
                        atm.z_half[atm.z_half <= 250] / 250)**3
        else:
            raise NotImplementedError("Wrong discretization in tke")


    def reconstruct_TKE(self, atm,
            SL: SurfaceLayerData, sf_scheme: str,
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
        xi = [np.linspace(-h/2, h/2, 15) for h in atm.h_half[:-1]]
        xi[1] = np.linspace(-atm.h_half[1]/2, atm.h_half[1]/2, 40)
        xi[0] = np.linspace(z_min-atm.h_half[0]/2, atm.h_half[0]/2, 40)
        sub_discrete = [self.tke[m] + \
                (self.dz_tke[m+1] + self.dz_tke[m]) * xi[m]/2 \
                + (self.dz_tke[m+1] - self.dz_tke[m]) / (2 * atm.h_half[m]) * \
                (xi[m]**2 - atm.h_half[m]**2/12) \
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
                h = atm.h_half[m]
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
                    atm.z_half[m] for m in range(self.M)])

            return z_oversampled, tke_oversampled
        tke_oversampled = np.concatenate(sub_discrete[1:])
        z_oversampled = np.concatenate([np.array(xi[m]) + \
                atm.z_half[m] for m in range(1, self.M)])
        # The part above the SL starts at z_oversampled[k2:].
        k2: int = bisect.bisect_right(z_oversampled,
                atm.z_full[SL.k+1])

        # We now compute the reconstruction inside the SL:
        phi_m, *_ = universal_funcs
        z_log: array = np.array((z_min, SL.delta_sl))
        # buoyancy and shear inside the SL:
        KN2 = 9.81/283. * SL.t_star * SL.u_star
        shear = SL.u_star**3 * phi_m(z_log * SL.inv_L_MO) / \
                atm.kappa / (z_log + SL.z_0M)
        from scipy.interpolate import interp1d
        z_levels = np.copy(atm.z_full)
        if not self.ignore_tke_sl:
            z_levels[SL.k] = SL.delta_sl
        f = interp1d(z_levels, l_eps, fill_value="extrapolate")
        l_eps_log = np.maximum(f(z_log), 0.)
        # With the quasi-equilibrium hypothesis, we get the tke:
        tke_log = np.maximum(((l_eps_log/atm.c_eps * \
                (shear - KN2))**2)**(1/3), self.e_min)

        # We finally compute the reconstruction just above SL:
        z_freepart = [] # we call "freepart" the zone
        k = SL.k # between the log profile and the next grid level:
        tilde_h = atm.z_full[k+1] - SL.delta_sl
        assert 0 < tilde_h <= atm.h_half[k]
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
