"""
    This module defines the class Simu1dStratified
    which simulates a 1D Ekman stratified problem with various
    discretisations.
"""

from typing import Tuple, List
import bisect
import numpy as np
from utils_linalg import multiply, scal_multiply as s_mult
from utils_linalg import add_banded as add
from utils_linalg import solve_linear
from utils_linalg import full_to_half
from universal_functions import Businger_et_al_1971

array = np.ndarray

class Simu1dStratified():
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
        self.Ktheta_min: float = 1e-5
        self.C_m: float = 0.126
        self.C_s: float = 0.143
        self.C_1: float = 0.143
        self.C_e: float = 0.34
        self.c_eps: float = 0.845
        self.z_star: float = 1e-1
        self.implicit_coriolis: float = 0.55 # semi-implicit coefficient
        self.lm_min: float = self.K_min / self.C_m / \
                np.sqrt(self.e_min)
        self.leps_min: float = self.K_min / self.C_m / \
                np.sqrt(self.e_min)
        # For each name of sf_scheme, two corresponding
        # methods defined at the bottom of this class:
        # sf_udelta_* is how we compute u(delta_sl) and
        # sf_YDc_* is the definition of the bottom boundary condition
        self.dictsf_scheme = {
                "FD pure" : (self.__sf_udelta_FDpure,
                                self.__sf_YDc_FDpure),
                "FD2" : (self.__sf_udelta_FD2,
                                self.__sf_YDc_FD2),
                "FV pure" : (self.__sf_udelta_FVpure,
                                self.__sf_YDc_FVpure),
                "FV1" : (self.__sf_udelta_FV1,
                                self.__sf_YDc_FV1),
                "FV2" : (self.__sf_udelta_FV2,
                                self.__sf_YDc_FV2),
                "FV free" : (self.__sf_udelta_FVfree,
                                self.__sf_YDc_FVfree)}
        self.dictsf_scheme_theta = {
                "FD pure" : (self.__sf_thetadelta_FDpure,
                                self.__sf_YDc_FDpure_theta),
                "FD2" : (self.__sf_thetadelta_FD2,
                                self.__sf_YDc_FD2_theta),
                "FV pure" : (self.__sf_thetadelta_FVpure,
                                self.__sf_YDc_FVpure_theta),
                "FV2" : (self.__sf_thetadelta_FV2,
                                self.__sf_YDc_FV2_theta),
                "FV free" : (self.__sf_thetadelta_FVfree,
                                self.__sf_YDc_FVfree_theta),
                }

    def FV(self, u_t0: array, phi_t0: array, forcing: array,
            SST:array, sf_scheme: str="FV pure",
            turbulence: str="TKE", delta_sl: float=None) -> array:
        """
            Integrates in time with Backward Euler the model with TKE
            and Finite volumes.

            u_t0 should be given at half-levels (follow self.z_half)
            phi_t0 should be given at full-levels (follow self.z_full)
            forcing should be given as averaged on each volume for all times
            sf_scheme is the surface flux scheme:
                - "FV pure" for a FD interpretation (classical but biaised)
                - "FV{1, 2}" for a FV interpretation
                                (unbiaised but delta_{sl}=z_1)
                - "FV{1, 2} free" for a FV interpretation with a free delta_{sl}
            turbulence should be one of:
                - "TKE" (for one-equation TKE)
                - "KPP" (for simple K-profile parametrization)
            delta_sl should be provided only with sf_scheme="FV1 free"
                (for delta_sl < z_1) or "FV free" (for any delta_sl)
            returns a numpy array of shape (M)
                                    where M is the number of space points
        """
        assert u_t0.shape[0] == self.M and phi_t0.shape[0] == self.M + 1
        assert forcing.shape[1] == self.M
        N: int = forcing.shape[0] - 1 # number of time steps
        func_un, _ = self.dictsf_scheme[sf_scheme]
        func_theta, _ = self.dictsf_scheme_theta[sf_scheme]
        if delta_sl is None:
            delta_sl = self.z_half[0] if sf_scheme in \
                    {"FV pure", "FV1"} \
                    else self.z_full[1]
        k = bisect.bisect_right(self.z_full[1:], delta_sl)

        tke = np.ones(self.M) * self.e_min
        tke[self.z_half[:-1] <= 250] = self.e_min + 0.4*(1 - \
                self.z_half[:-1][self.z_half[:-1] <= 250] / 250)**3
        # inversion of a system to find dz_tke:
        tke_full = np.ones(self.M+1)*self.e_min
        tke_full[self.z_full <= 250] = self.e_min + 0.4*(1 - \
                self.z_full[self.z_full <= 250] / 250)**3
        dz_tke = self.__compute_dz_tke(tke_full, tke, delta_sl, k)
        h_tilde = self.z_full[k+1] - delta_sl

        businger = Businger_et_al_1971()

        # approximate profile of theta: 265 then linear increasing
        theta: array = 265 + np.maximum(0, 0.01 * (self.z_half[:-1]-100))
        # But there's an angle, which is not a quadratic spline:
        index_angle = bisect.bisect_right(theta, theta[0] + 1e-4)
        theta[index_angle] = 265 + 0.01 * \
                (self.z_full[index_angle+1] - 100.)/2 * \
                (self.z_full[index_angle+1] - 100.)/ \
                (self.h_half[index_angle])
        # inversion of a system to find phi:
        ldiag, diag, udiag, rhs = np.ones(self.M)/6, \
                np.ones(self.M+1)*2/3, np.ones(self.M)/6, \
                np.diff(theta, prepend=0, append=0.)/ self.h_half
        udiag[0] = rhs[0] = 0. # bottom Neumann condition
        ldiag[-1], diag[-1], rhs[-1] = 0., 1., 0.01 # top Neumann
        dz_theta: array = solve_linear((ldiag, diag, udiag), rhs)

        Ku_full: array = self.K_min + np.zeros(self.M+1)
        Ktheta_full: array = self.Ktheta_min + np.zeros(self.M+1)
        l_m = self.lm_min*np.ones(self.M+1)
        l_eps = self.leps_min*np.ones(self.M+1)

        phi = phi_t0
        old_phi, old_theta = np.copy(phi), np.copy(theta)
        l_eps = self.leps_min * np.ones(self.M+1)
        l_m = self.lm_min * np.ones(self.M+1)

        prognostic: array = np.concatenate((u_t0[:k+1], phi_t0[k:]))
        prognostic_theta: array = np.concatenate((
            theta[:k+1], dz_theta[k:]))
        u_current: array = np.copy(u_t0)
        all_u_star = []
        u_delta, t_delta = 8. + 0j, 265. #TODO better initialisation

        if N == 0: # when we want to visualize the initial condition
            u_star, t_star = self.__friction_scales(
                    u_delta=u_delta, t_delta=t_delta,
                    delta_sl=delta_sl, SST=SST[0],
                    universal_funcs=businger)
            inv_L_MO = t_star / t_delta / u_star**2 * \
                    self.kappa * 9.81


        for n in range(1,N+1):
            forcing_current, SST_current = forcing[n], SST[n]
            SST_derivative = (SST[n] - SST[n-1]) / self.dt
            forcing_theta = np.zeros_like(forcing_current)
            u_star, t_star = self.__friction_scales(
                    u_delta=u_delta, t_delta=t_delta,
                    delta_sl=delta_sl, SST=SST_current,
                    universal_funcs=businger)
            all_u_star += [u_star]
            inv_L_MO = t_star / t_delta / u_star**2 * \
                    self.kappa * 9.81

            Ku_full, Ktheta_full, tke, dz_tke = self.__visc_turb_FV(u_star,
                    delta_sl, turbulence=turbulence, phi=phi,
                    old_phi=old_phi, l_m=l_m, l_eps=l_eps,
                    K_full=Ku_full, tke=tke, dz_tke=dz_tke,
                    dz_theta=dz_theta, Ktheta_full=Ktheta_full,
                    universal_funcs=businger, inv_L_MO=inv_L_MO)


            Y, D, c = self.__matrices_u_FV(Ku_full, forcing_current)

            Y_theta, D_theta, c_theta = self.__matrices_theta_FV(
                    Ktheta_full, forcing_theta)

            self.__apply_sf_scheme(\
                    func=self.dictsf_scheme_theta[sf_scheme][1],
                    Y=Y_theta, D=D_theta, c=c_theta,
                    SST=SST_current, K_theta=Ktheta_full,
                    forcing=forcing_theta, u_star=u_star,
                    t_star=t_star, t_delta=t_delta,
                    delta_sl=delta_sl, universal_funcs=businger,
                    inv_L_MO=inv_L_MO,
                    SST_derivative=SST_derivative)

            self.__apply_sf_scheme(\
                    func=self.dictsf_scheme[sf_scheme][1],
                    Y=Y, D=D, c=c, K_u=Ku_full,
                    forcing=forcing_current, u_star=u_star,
                    u_delta=u_delta, delta_sl=delta_sl,
                    t_star=t_star, t_delta=t_delta,
                    universal_funcs=businger,
                    inv_L_MO=inv_L_MO)

            prognostic_theta = np.real(self.__backward_euler(Y=Y_theta,
                    D=D_theta, c=c_theta, u=prognostic_theta, f=0.))

            prognostic = self.__backward_euler(Y=Y, D=D, c=c,
                    u=prognostic, f=self.f)

            next_u = 1/(1+self.dt*1j*self.f*self.implicit_coriolis) * \
                    ((1 - self.dt*1j*self.f*(1-self.implicit_coriolis)) * \
                    u_current + self.dt * \
                    (np.diff(prognostic[1:] * Ku_full) / self.h_half[:-1] \
                    + forcing_current))

            next_theta = theta + self.dt * \
                    np.diff(prognostic_theta[1:] * Ktheta_full) \
                    / self.h_half[:-1]


            next_theta[:k+1] = prognostic_theta[:k+1]
            next_u[:k+1] = prognostic[:k+1]

            u_current, old_u = next_u, u_current
            theta, old_theta = next_theta, theta

            old_phi, phi = phi, prognostic[k+1:]
            old_dz_theta, dz_theta = dz_theta, prognostic_theta[k+1:]

            if k > 0: # constant flux layer : K[:k] phi[:k] = K[0] phi[0]
                phi = np.concatenate((Ku_full[k]*phi[0]*Ku_full[:k], phi))
                dz_theta = np.concatenate(( Ktheta_full[k]* \
                        dz_theta[0]*Ktheta_full[:k], dz_theta))

            # self.debug_seul(theta, dz_theta, delta_sl, sf_scheme)
            u_delta: complex = func_un(prognostic=prognostic,
                    delta_sl=delta_sl, inv_L_MO=inv_L_MO,
                    universal_funcs=businger)
            t_delta: float = func_theta(prognostic=prognostic_theta,
                    delta_sl=delta_sl, inv_L_MO=inv_L_MO,
                    universal_funcs=businger,
                    SST=SST_current)

        tke = self.__compute_tke_full(tke, dz_tke, u_star,
                delta_sl, k)
        return u_current, phi, tke, all_u_star, theta, \
                dz_theta, l_m, inv_L_MO


    def FD(self, u_t0: array, forcing: array, SST:array,
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
            returns a numpy array of shape (M)
                                    where M is the number of space points
        """
        assert u_t0.shape[0] == self.M
        assert forcing.shape[1] == self.M
        N: int = forcing.shape[0] - 1 # number of time steps
        func_un, _ = self.dictsf_scheme[sf_scheme]
        delta_sl = self.z_half[0] if sf_scheme in {"FD pure"} \
                else self.z_full[1]

        tke = np.ones(self.M+1) * self.e_min
        tke[self.z_half <= 250] = self.e_min + 0.4*(1 - \
                self.z_half[self.z_half <= 250] / 250)**3
        businger = Businger_et_al_1971()

        theta: array = 265 + np.maximum(0, 0.01 * (self.z_half[:-1]-100))
        Ku_full: array = self.K_min + np.zeros_like(tke)
        Ktheta_full: array = self.Ktheta_min + np.zeros_like(tke)
        l_m = self.lm_min*np.ones(self.M+1)
        l_eps = self.leps_min*np.ones(self.M+1)

        u_current: array = np.copy(u_t0)
        old_theta: array = np.copy(theta)
        old_u = np.copy(u_current)
        all_u_star = []
        for n in range(1,N+1):
            forcing_current, SST_current = forcing[n], SST[n]
            u_delta = func_un(prognostic=u_current, delta_sl=delta_sl)
            t_delta = self.__sf_thetadelta_FDpure(prognostic=theta)

            u_star, t_star = self.__friction_scales(u_delta=u_delta,
                    t_delta=t_delta, delta_sl=delta_sl, SST=SST_current,
                    universal_funcs=businger)
            all_u_star += [u_star]

            Ku_full, Ktheta_full, tke = self.__visc_turb_FD(u_star=u_star,
                    delta_sl=delta_sl, turbulence="TKE",
                    u_current=u_current, old_u=old_u,
                    K_full=Ku_full, tke=tke, theta=theta,
                    Ktheta_full=Ktheta_full, l_m=l_m, l_eps=l_eps)
            Y, D, c = self.__matrices_u_FD(Ku_full, forcing_current)
            # lam_s, lam_m = self.__lambdas_reaction(Pr)
            Y_theta, D_theta, c_theta = self.__matrices_theta_FD(
                    Ktheta_full, np.zeros(self.M))

            _, func = self.dictsf_scheme[sf_scheme]
            self.__apply_sf_scheme(func=self.dictsf_scheme[sf_scheme][1],
                    Y=Y, D=D, c=c, K_u=Ku_full,
                    forcing=forcing_current, u_star=u_star,
                    u_delta=u_delta, delta_sl=delta_sl)

            self.__apply_sf_scheme(\
                    func=self.dictsf_scheme_theta[sf_scheme][1],
                    Y=Y_theta, D=D_theta, SST=SST_current,
                    c=c_theta, K_theta=Ktheta_full,
                    forcing=forcing_current, u_star=u_star,
                    t_star=t_star, t_delta=t_delta,
                    delta_sl=delta_sl)

            next_theta = np.real(self.__backward_euler(Y=Y_theta,
                    D=D_theta, c=c_theta, u=theta, f=0.))
            next_u = self.__backward_euler(Y=Y, D=D, c=c,
                    u=u_current, f=self.f)
            u_current, old_u = next_u, u_current
            theta, old_theta = next_theta, theta

        return u_current, tke, all_u_star, theta, l_m

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

    def __compute_dz_tke(self, tke_full, tke, delta_sl, k):
        """ solving the system:
        tke-h*5/12phi[:-1]-h*phi[1:]/12 = tke_full[:-1]
        """
        ldiag = self.h_half[:-2] / 12.
        diag = self.h_full[1:-1] * 10./12.
        udiag = self.h_half[1:-1] / 12.
        h_tilde = self.z_full[k+1] - delta_sl
        diag = np.concatenate(([h_tilde*5/12], diag, [1.]))
        udiag = np.concatenate(([h_tilde*1/12], udiag))
        ldiag = np.concatenate((ldiag, [0.]))
        rhs = np.concatenate(([tke[0]-tke_full[0]],
            np.diff(tke), [0.]))
        old_diag, old_udiag, old_ldiag, old_rhs = diag, udiag, ldiag, rhs
        # GRID LEVEL k-1 AND BELOW: dz_tke=0:
        diag[:k], udiag[:k] = 1., 0.
        rhs[:k] = 0.
        # GRID LEVEL k : tke(z=delta_sl) = tke_full[k]
        ldiag[:k] = 0. # ldiag[k-1] is for cell k
        diag[k], udiag[k] = h_tilde*5/12., h_tilde/12.
        rhs[k] = tke[k] - tke_full[k]
        if k == 0:
            assert np.linalg.norm(old_diag - diag) == 0.
            assert np.linalg.norm(old_udiag - udiag) == 0.
            assert np.linalg.norm(old_ldiag - ldiag) == 0.
            assert np.linalg.norm(old_rhs - rhs) == 0.
        # GRID LEVEL k+1: h_tilde used in continuity equation
        ldiag[k] = h_tilde / 12.
        diag[k+1] = (h_tilde+self.h_half[k+1]) * 5./12.
        return solve_linear((ldiag, diag, udiag), rhs)

    def __integrate_tke_FV(self, tke, dz_tke, shear, K_full, delta_sl, u_star,
            l_eps, Ktheta_full=None, N2=None):
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
            l_eps is the tke mixing length (at full levels)
            Ktheta_full is K_theta (at full levels)
            N2 is the Brunt–Vaisälä frequency (at full levels).

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
        # computing buoyancy on half levels:
        buoy_half = full_to_half(Ktheta_full*N2)
        shear_half = full_to_half(shear)

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

    def __integrate_tke(self, tke, shear, K_full, delta_sl, u_star,
            l_eps, Ktheta_full=None, N2=None):
        """
            integrates TKE equation on one time step.
            discretisation of TKE is Finite Differences,
            located on half-points.
        """
        if Ktheta_full is None or N2 is None:
            Ktheta_full = N2 = np.zeros(self.M)
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
        rhs_e = np.concatenate(([e_sl], [tke[m]/self.dt + shear[m]
                for m in range(1, self.M)],
            [tke[self.M]/self.dt]))
        for m in range(1, self.M):
            if shear[m] <= Ktheta_full[m] * N2[m]: # Patankar trick
                diag_e[m] += Ktheta_full[m] * N2[m] / tke[m]
            else: # normal handling of buoyancy
                rhs_e[m] -=  Ktheta_full[m] * N2[m]
        # if delta_sl is inside the computational domain:
        k = bisect.bisect_right(self.z_full[1:], delta_sl)
        rhs_e[:k+1] = e_sl # prescribe e=e_sl in all the ASL
        diag_e[:k+1] = 1.
        udiag_e[:k+1] = ldiag_e[:k] = 0.

        return solve_linear((ldiag_e, diag_e, udiag_e), rhs_e)

    def mixing_lengths(self, tke, dzu2, N2):
        """
            returns the mixing lengths (l_m, l_eps)
            for given entry parameters.
            dzu2 =||du/dz||^2 should be computed similarly
            to the shear
            l_m, l_eps, l_up and l_down are
            computed on full levels.
        """
        l_up = np.zeros(self.M+1) + \
                (self.kappa / self.C_m) * \
                (self.C_m*self.c_eps)**.25 * self.z_star
        l_down = np.copy(l_up)
        l_up[-1] = l_down[-1] = self.lm_min

        #  Mixing length computation
        Rod = 0.2
        buoyancy = np.maximum(1e-12, N2)
        mxlm = np.maximum(self.lm_min, 2.*np.sqrt(tke) / \
                (Rod*np.sqrt(dzu2) + np.sqrt(Rod**2*dzu2+2.*buoyancy)))
        mxlm[0] = mxlm[-1] = self.lm_min

        # should not exceed linear mixing lengths:
        for j in range(self.M - 1, -1, -1):
            l_up[j] = min(l_up[j+1] + self.h_half[j], mxlm[j])
        for j in range(1, self.M+1):
            l_down[j] = min(l_down[j-1] + self.h_half[j-1], mxlm[j])

        a = -(np.log(self.c_eps)-3.*np.log(self.C_m)+ \
                4.*np.log(self.kappa))/np.log(16.)
        l_m = np.maximum((0.5*(l_down**(1./a) + l_up**(1./a)))**a,
                    self.lm_min)
        l_eps = np.minimum(l_down, l_up) # l_eps

        return l_m, l_eps

    def reconstruct_FV(self, u_bar: array, phi: array, theta: array,
            dz_theta: array, inv_L_MO: float, SST: float, sf_scheme: str="FV pure",
            delta_sl: float=None,
            ignore_loglaw: bool=False):
        """
            spline reconstruction of the FV solution.
            input: discrete representation in FV,
            output: arrays (z, u(z))
            u(xi) = u_bar[m] + (phi[m+1] + phi[m]) * xi/2 +
                    (phi[m+1] - phi[m]) / (2*h_half[m]) *
                        (xi^2 - h_half[m]^2/12)
            where xi = linspace(-h_half[m]/2, h_half[m]/2, 10, endpoint=True)
        """
        xi = [np.linspace(-h/2, h/2, 5) for h in self.h_half[:-1]]
        xi[1] = np.linspace(-self.h_half[1]/2, self.h_half[1]/2, 40)
        sub_discrete: List[array] = [u_bar[m] + (phi[m+1] + phi[m]) * xi[m]/2 \
                + (phi[m+1] - phi[m]) / (2 * self.h_half[m]) * \
                (xi[m]**2 - self.h_half[m]**2/12) for m in range(self.M)]

        sub_discrete_theta: List[array] = [theta[m] + (dz_theta[m+1] + dz_theta[m]) * xi[m]/2 \
                + (dz_theta[m+1] - dz_theta[m]) / (2 * self.h_half[m]) * \
                (xi[m]**2 - self.h_half[m]**2/12) for m in range(self.M)]
        u_oversampled = np.concatenate(sub_discrete[1:])
        theta_oversampled = np.concatenate(sub_discrete_theta[1:])
        z_oversampled = np.concatenate([np.array(xi[m]) + self.z_half[m]
                                            for m in range(1, self.M)])
        if sf_scheme in {"FV1", "FV pure"} or ignore_loglaw:
            allxi = [np.array(xi[m]) + self.z_half[m] for m in range(self.M)]
            return np.concatenate(allxi), \
                    np.concatenate(sub_discrete), \
                    np.concatenate(sub_discrete_theta), \


        if delta_sl is None:
            delta_sl = self.z_half[0] if sf_scheme in\
                    {"FV pure", "FV1"} else self.z_full[1]

        # k1 is the index of the grid level containing delta_sl
        k1: int = bisect.bisect_right(self.z_full[1:], delta_sl)
        prognostic: array = np.concatenate((u_bar[:k1+1], phi[k1:]))
        prognostic_theta: array = np.concatenate((theta[:k1+1], dz_theta[k1:]))

        # getting information of the surface layer (from which
        # we get the MOST profiles)
        z_log: array = np.geomspace(self.z_star, delta_sl, 20)

        func_un, _ = self.dictsf_scheme[sf_scheme]
        func_theta, _ = self.dictsf_scheme_theta[sf_scheme]

        businger = Businger_et_al_1971()

        u_delta: complex = func_un(prognostic=prognostic,
                delta_sl=delta_sl, inv_L_MO=inv_L_MO,
                universal_funcs=businger)
        t_delta: float = func_theta(prognostic=prognostic_theta,
                delta_sl=delta_sl, inv_L_MO=inv_L_MO,
                universal_funcs=businger,
                SST=SST)

        u_star, t_star = self.__friction_scales(
                u_delta=u_delta, t_delta=t_delta,
                delta_sl=delta_sl, SST=SST,
                universal_funcs=businger)

        _, _, psi_m, psi_h, *_ = businger
        Pr = 1.# 4.8/7.8
        u_log: complex = u_star/self.kappa * \
                (np.log(1+z_log/self.z_star) - \
                psi_m(z_log*inv_L_MO) + psi_m(self.z_star*inv_L_MO)) \
                            * u_delta/np.abs(u_delta)
        theta_log: complex = SST + Pr * t_star / self.kappa * \
                (np.log(1+z_log/self.z_star) - \
                psi_h(z_log*inv_L_MO) + psi_h(self.z_star*inv_L_MO)) \
                            * np.sign(t_delta - SST)

        k2: int = bisect.bisect_right(z_oversampled, self.z_full[k1+1])

        z_freepart = []
        u_freepart = []
        theta_freepart = []

        if sf_scheme in {"FV1 free", "FV free", "FV2 free"}:
            # between the log profile and the next grid level:
            tilde_h = self.z_full[k1+1] - delta_sl
            assert 0 < tilde_h <= self.h_half[k1]
            xi = np.linspace(-tilde_h/2, tilde_h/2, 10)
            tau_slu, tau_slt = self.__tau_sl(delta_sl=delta_sl,
                    universal_funcs=businger, inv_L_MO=inv_L_MO)
            alpha_slu = tilde_h/self.h_half[k1] + tau_slu
            alpha_slt = tilde_h/self.h_half[k1] + tau_slt

            u_tilde = 1/alpha_slu * (u_bar[k1] + tilde_h * tau_slu * \
                    (phi[k1]/3 + phi[k1+1]/6))
            theta_tilde = 1/alpha_slt * (theta[k1] + tilde_h * tau_slt * \
                    (dz_theta[k1]/3 + dz_theta[k1+1]/6) - (1-alpha_slt)*SST)

            u_freepart = u_tilde + (phi[k1+1] + phi[k1]) * xi/2 \
                    + (phi[k1+1] - phi[k1]) / (2 * tilde_h) * \
                    (xi**2 - tilde_h**2/12)

            theta_freepart = theta_tilde + (dz_theta[k1+1] + dz_theta[k1]) * xi/2 \
                    + (dz_theta[k1+1] - dz_theta[k1]) / (2 * tilde_h) * \
                    (xi**2 - tilde_h**2/12)

            z_freepart = delta_sl + xi + tilde_h / 2
        elif sf_scheme in {"FV2"}: # link log cell with profile
            k2: int = bisect.bisect_right(z_oversampled,
                    self.z_full[k1])

        return np.concatenate((z_log, z_freepart, z_oversampled[k2:])), \
                np.concatenate((u_log, u_freepart, u_oversampled[k2:])), \
                np.concatenate((theta_log, theta_freepart, theta_oversampled[k2:]))


    def __backward_euler(self, Y: Tuple[array, array, array],
                        D: Tuple[array, array, array], c: array,
                        u: array, f:float=0.):
        """
            if it's for $u$, set f=self.f otherwise f=0
            integrates once (self.dt) in time the equation
            (partial_t + if)Yu - dt*Du = c
            The scheme is:
            Y(1+dt*if*gamma) - D) u_np1 = Y u + dt*c + dt*if*(1-gamma)
            with gamma the coefficient of implicitation of Coriolis
        """
        to_inverse: Tuple[array, array, array] = add(s_mult(Y,
                                1 + self.implicit_coriolis * \
                            self.dt * 1j*f), s_mult(D, - self.dt))
        return solve_linear(to_inverse,
                (1 - (1 - self.implicit_coriolis) * self.dt*1j*f) * \
                                multiply(Y, u) + self.dt*c)

    def __visc_turb_FD(self, u_star, delta_sl,
            turbulence="TKE", u_current=None, old_u=None,
            K_full=None, tke=None, theta=None,
            Ktheta_full=None, l_m=None, l_eps=None):
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
            du = np.diff(u_current)
            du_old = np.diff(old_u)
            shear = np.concatenate(([0],
                [np.abs(K_full[m]/self.h_full[m]**2 * du[m-1] * \
                        (du[m-1]+du_old[m-1])/2) \
                        for m in range(1, self.M)], [0]))
            dz_theta = np.diff(theta) / self.h_full[1:-1]
            g, theta_ref = 9.81, 283.
            N2 = np.concatenate(([1e-12], g/theta_ref * dz_theta, [1e-12]))
            tke[:] = np.maximum(self.e_min,
                    self.__integrate_tke(tke, shear, K_full,
                    delta_sl=delta_sl, u_star=u_star,
                    l_eps=l_eps, N2=N2, Ktheta_full=Ktheta_full))

            l_m[:], l_eps[:] = self.mixing_lengths(tke,
                    shear/K_full, N2)

            phi_z = self.__stability_temperature_phi_z(z=self.z_full,
                    C_1=self.C_1, l_m=l_m, l_eps=l_eps, N2=N2,
                    TKE=tke)


            Ktheta_full: array = np.maximum(self.Ktheta_min,
                    self.C_s * phi_z * l_m * np.sqrt(tke))

            Ku_full: array = np.maximum(self.K_min,
                    self.C_m * l_m * np.sqrt(tke))

        else:
            raise NotImplementedError("Wrong turbulence scheme")
        return Ku_full, Ktheta_full, tke

    def __stability_temperature_phi_z(self, z, C_1, l_m, l_eps, N2, TKE):
        return 1/(1+np.maximum(-0.5455,
                C_1*l_m*l_eps*np.maximum(1e-12, N2)/TKE))

    def __matrices_u_FD(self, K_full, forcing):
        """
            Creates the matrices D, Y, c such that the
            semi-discrete in space Ekman stratified equation 
            for the momentum writes ((d/dt+if) Y - D) u = c
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

    def __matrices_theta_FD(self, K_full, forcing):
        """
            Creates the matrices D, Y, c such that the
            semi-discrete in space Ekman Stratified equation
            for the temperature writes (d/dt Y - D) u = c
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

    def __visc_turb_FV(self, u_star, delta_sl,
            turbulence="TKE", phi=None, old_phi=None,
            K_full=None, tke=None, dz_tke=None,
            l_m=None, l_eps=None,
            dz_theta=None, Ktheta_full=None,
            universal_funcs=None, inv_L_MO=None):
        """
            Computes the turbulent viscosity on full levels K_full.
            It differs between FD and FV because of the
            shear computation (in the future, other differences
            might appear like the temperature handling).
            returns (K_full, TKE).
            turbulence is either "TKE" or "KPP"
            inv_L_MO is the 1/(Obukhov length),
            l_m, l_eps the mixing lengths
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
            shear = np.concatenate(([0],
                [np.abs(K_full[m]*phi[m]*(phi[m] + old_phi[m])/2) \
                        for m in range(1, self.M)], [0]))

            k = bisect.bisect_right(self.z_full[1:], delta_sl)

            g, theta_ref = 9.81, 283.

            tke[:], dz_tke[:], tke_full = \
                    self.__integrate_tke_FV(tke, dz_tke,
                    shear=shear, K_full=K_full,
                    delta_sl=delta_sl, u_star=u_star,
                    l_eps=l_eps, N2=g/theta_ref * dz_theta,
                    Ktheta_full=Ktheta_full)

            if (tke_full < self.e_min).any() or \
                    (tke < self.e_min).any():
                tke_full = np.maximum(tke_full, self.e_min)
                tke = np.maximum(tke, self.e_min)
                dz_tke = self.__compute_dz_tke(tke_full, tke,
                                delta_sl, k)

            l_m[:], l_eps[:] = self.mixing_lengths(tke_full,
                    shear/K_full, g/theta_ref*dz_theta)

            phi_z = self.__stability_temperature_phi_z(z=self.z_full,
                    C_1=self.C_1, l_m=l_m, l_eps=l_eps,
                    N2=g/theta_ref*dz_theta, TKE=tke_full)

            Ktheta_full: array = np.maximum(self.Ktheta_min,
                    self.C_s * phi_z * l_m * np.sqrt(tke_full))

            phi_m, phi_h, *_ = universal_funcs
            Ktheta_full[:k] = (self.kappa * u_star*(self.z_full[:k]+self.z_star))\
                    / phi_h(self.z_full[:k]*inv_L_MO)
            Ktheta_full[k] = (self.kappa * u_star*(delta_sl+self.z_star))\
                        / phi_h(delta_sl*inv_L_MO)

            K_full = self.C_m * l_m * np.sqrt(tke_full)
            # K_full[:k] =  self.kappa*u_star*(
            #         self.z_full[:k] + self.z_star)
            # K_full[k] = self.kappa*u_star*(delta_sl + self.z_star)
            K_full[:k] = (self.kappa * u_star*(self.z_full[:k]+self.z_star)\
                    ) / phi_m(self.z_full[:k]*inv_L_MO)
            K_full[k] = (self.kappa * u_star*(delta_sl+self.z_star)\
                    ) / phi_m(delta_sl*inv_L_MO)
        else:
            raise NotImplementedError("Wrong turbulence scheme")

        return K_full, Ktheta_full, tke, dz_tke

    def __matrices_u_FV(self, K_full, forcing):
        """
            Creates the matrices D, Y, c such that the
            semi-discrete in space Ekman Stratified equation writes
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


    def __matrices_theta_FV(self, K_full, forcing):
        """
            Creates the matrices D, Y, c such that the
            semi-discrete in space Ekman Stratified equation writes
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

    # def __lambdas_reaction(self, ):
    #     beta_min = 0.3
    #     beta_max = 1.
    #     h_bl = 1000 # WARNING this should be characterized by
    #                 # something like c_1 u* / f
    #     # make sure that beta_min h_bl > z_full[2]
    #     # make sure that beta_max h_bl < z_full[-3]
    #     # z_full[2] / beta_min < h_bl < z_full[-3]/beta_max
    #     assert self.z_full[2] / beta_min < self.z_full[-3]/beta_max
    #     h_bl = np.clip(h_bl, self.z_full[2] / beta_min, self.z_full[-3]/beta_max)

    #     lambda_s_min = 0.2 / self.dt
    #     lambda_s_max = 
    #     lambda_m_min = 
    #     lambda_m_max = 
    #     alpha = [ 
    #             (3*beta_max - beta_min) * beta_min**2 * lambda_s_max \
    #             + (beta_max - 3*beta_min) * beta_max**2 * lambda_s_min,
    #             -6*beta_max*beta_min*(lambda_s_max - lambda_s_min),
    #             3*(beta_max+beta_min)*(lambda_s_max - lambda_s_min),
    #             -2 * (lambda_s_max - lambda_s_min) ]
    #     lambda_s = np.sum(np.array([
    #         alpha[m] / (beta_max - beta_min)**3 * (z / h_bl)**m 
    #             for m in range(4)]))
    #     return lambda_s, lambda_m

    def __apply_sf_scheme(self, func, Y, D, c, **kwargs):
        """
            Changes matrices Y, D, c on the first levels
            to use the surface flux scheme.
            _, func = self.dictsf_scheme[sf_scheme]
        """
        Y_sf, D_sf, c_sf = func(**kwargs)
        for y, y_sf in zip(Y, Y_sf):
            y_sf = np.array(y_sf)
            y[:y_sf.shape[0]] = y_sf
        for d, d_sf in zip(D, D_sf):
            d_sf = np.array(d_sf)
            d[:d_sf.shape[0]] = d_sf

        c_sf = np.array(c_sf)
        c[:c_sf.shape[0]] = c_sf

    def __friction_scales(self, u_delta, delta_sl, t_delta, SST,
            universal_funcs):
        """
        return (u*, t*) with a fixed point algorithm.
        universal_funcs is the tuple (phim, phih, psim, psih, Psim, Psih)
        defined in universal_functions.py
        """
        _, _, psim, psis, *_ = universal_funcs
        t_star: float = (t_delta-SST) * \
                (0.0180 if t_delta > SST else 0.0327)
        u_star: float = (self.kappa *np.abs(u_delta) / \
                np.log(1 + delta_sl/self.z_star ) )
        for _ in range(5):
            zeta = self.kappa * delta_sl * 9.81*\
                    (t_star / t_delta) / u_star**2
            Cd    = self.kappa**2 / \
                    (np.log(1+delta_sl/self.z_star) - psim(zeta))**2
            Ch    = self.kappa * np.sqrt(Cd) / \
                    (np.log(1+delta_sl/self.z_star) - psis(zeta))
            u_star = np.sqrt(Cd) * np.abs(u_delta)
            t_star = ( Ch / np.sqrt(Cd) ) * (t_delta - SST)
            # self.z_star = K_mol / self.kappa / u_star ?
        return u_star, t_star

    def __tau_sl(self, delta_sl, universal_funcs, inv_L_MO):
        _, _, psi_m, psi_h, Psi_m, Psi_h = universal_funcs
        k = bisect.bisect_right(self.z_full[1:], delta_sl)
        zk = self.z_full[k]
        def brackets_u(z):
            return (z+self.z_star)*np.log(1+z/self.z_star) - z + \
                    z*Psi_m(z*inv_L_MO)
        def brackets_theta(z):
            return (z+self.z_star)*np.log(1+z/self.z_star) - z + \
                    z*Psi_h(z*inv_L_MO)
        log_delta = np.log(1+delta_sl/self.z_star)
        denom_u = log_delta - psi_m(delta_sl*inv_L_MO)
        denom_theta = log_delta - psi_h(delta_sl*inv_L_MO)

        tau_slu = (brackets_u(delta_sl) - brackets_u(zk)) / \
                self.h_half[k] / denom_u
        tau_slt = (brackets_theta(delta_sl)-brackets_theta(zk)) / \
                self.h_half[k] / denom_theta

        return tau_slu, tau_slt

    ####### DEFINITION OF SF SCHEMES : VALUE OF u(delta_sl) #####
    # The method must use the prognostic variables and delta_sl
    # to return u(delta_sl).
    # the prognostic variables are u for FD and 
    # (u_{1/2}, ... u_{k+1/2}, phi_k, ...phi_M) for FV.

    def __sf_udelta_FDpure(self, prognostic, **_):
        return prognostic[0]

    def __sf_udelta_FD2(self, prognostic, **_):
        return (prognostic[0] + prognostic[1])/2

    def __sf_udelta_FVpure(self, prognostic, **_):
        return prognostic[0] - self.h_half[0]* \
                (prognostic[2] - prognostic[1])/24

    def __sf_udelta_FV1(self, prognostic, **_):
        return prognostic[0]

    def __sf_udelta_FV2(self, prognostic, **_):
        return prognostic[1] - self.h_half[1] * \
                (prognostic[3]/6 + prognostic[2]/3)

    def __sf_udelta_FVfree(self, prognostic, delta_sl,
            universal_funcs, inv_L_MO, **_):
        tau_slu, _ = self.__tau_sl(delta_sl=delta_sl,
                universal_funcs=universal_funcs, inv_L_MO=inv_L_MO)
        k = bisect.bisect_right(self.z_full[1:], delta_sl)
        tilde_h = self.z_full[k+1] - delta_sl
        return (prognostic[k] - tilde_h*tilde_h/self.h_half[k] * \
                (prognostic[k+1] / 3 + prognostic[k+2] / 6)) \
                / (tilde_h/self.h_half[k]+tau_slu)

    ####### DEFINITION OF SF SCHEMES : FIRST LINES OF Y,D,c #####
    # each method must return Y, D, c:
    # Y: 3-tuple of tuples of sizes (j-1, j, j)
    # D: 3-tuple of tuples of sizes (j-1, j, j)
    # c: tuple of size j
    # they represent the first j lines of the matrices.
    # for Y and D, the tuples are (lower diag, diag, upper diag)

    def __sf_YDc_FDpure(self, K_u, forcing, u_star, u_delta, **_):
        Y = ((), (1.,), (0.,))
        D = ((), (-K_u[1]/self.h_full[1]/self.h_half[0] - \
                u_star**2 / np.abs(u_delta)/self.h_half[0], ),
                (K_u[1]/self.h_full[1]/self.h_half[0],))
        c = (forcing[0],)
        return Y, D, c

    def __sf_YDc_FD2(self, K_u, u_star, u_delta, **_):
        Y = ((), (0.,), (0.,))
        D = ((), (-K_u[1]/self.h_full[1] - u_star**2 / np.abs(u_delta) / 2,),
                (K_u[1]/self.h_full[1] - u_star**2 / np.abs(u_delta) / 2,))
        c = (0.,)
        return Y, D, c

    def __sf_YDc_FVpure(self, K_u, forcing, u_star, u_delta, **_):
        Y = ((1.,), (0., 0.), (0., 0.))
        D = ((0.,), (1, -K_u[0] / self.h_half[0]),
                (-K_u[0]*np.abs(u_delta)/u_star**2+self.h_half[0]/24,
                    K_u[1]/self.h_half[0]),
                (-self.h_half[0]/24,))
        c = (0., forcing[0])
        return Y, D, c

    def __sf_YDc_FV1(self, K_u, forcing, u_star, u_delta, **_):
        Y = ((1.,), (0., 0.), (0., 0.))
        D = ((0.,), (1, -K_u[0] / self.h_half[0]),
                (-K_u[0]*np.abs(u_delta)/u_star**2, K_u[1]/self.h_half[0]))
        c = (0., forcing[0])
        return Y, D, c

    def __sf_YDc_FV2(self, K_u, forcing, u_star, u_delta, t_star,
            t_delta, universal_funcs, **_):
        _, _, _, _, Psi_m, _ = universal_funcs
        inv_L_MO = t_star / t_delta / u_star**2 * self.kappa * 9.81
        def f(z):
            return (z+self.z_star)*np.log(1+z/self.z_star) - z + \
                    z * Psi_m(z*inv_L_MO)

        try:
            ratio_norms = (f(self.z_full[1]) - f(0)) / \
                    (f(self.z_full[2]) - f(self.z_full[1]))
        except ZeroDivisionError:
            ratio_norms = 0.
        Y = ((0., 1.), (0., 0., 0.), (0., 0., 0.))
        D = ((0., 0.), (1., 1., -K_u[1] / self.h_half[1]),
                (-ratio_norms, -K_u[1]*np.abs(u_delta)/u_star**2 - \
                        self.h_half[1]/3, K_u[2]/self.h_half[1]),
                (0., -self.h_half[1]/6))
        c = (0., 0., forcing[1])
        return Y, D, c

    def __sf_YDc_FVfree(self, K_u, forcing, u_star, u_delta,
            delta_sl, universal_funcs, inv_L_MO, **_):
        k = bisect.bisect_right(self.z_full[1:], delta_sl)
        tilde_h = self.z_full[k+1] - delta_sl
        tau_slu, _ = self.__tau_sl(delta_sl=delta_sl,
                universal_funcs=universal_funcs, inv_L_MO=inv_L_MO)
        alpha_sl = tilde_h/self.h_half[k] + tau_slu

        Y = ((1, tilde_h / 6 / self.h_full[k+1]),
                (0., tilde_h*tau_slu/3,
                    tilde_h/3/self.h_full[k+1] + \
                            self.h_half[k+1]/3/self.h_full[k+1]),
                (0., tilde_h*tau_slu/6,
                    self.h_half[k+1]/6/self.h_full[k+1]))

        D = ((0., K_u[k+0]/tilde_h/self.h_full[k+1]),
                (-1., -alpha_sl*K_u[k+0] / tilde_h,
                    -K_u[k+1]/tilde_h/self.h_full[k+1] - \
                            K_u[k+1] / self.h_half[k+1] / self.h_full[k+1]),
                (K_u[k+0]*np.abs(u_delta)*alpha_sl/u_star**2 + \
                        tilde_h**2 / 3 / self.h_half[k],
                    alpha_sl * K_u[k+1]/tilde_h, K_u[k+2]/self.h_full[k+1]/self.h_half[k+1]),
                (tilde_h**2 / 6 / self.h_half[k], 0.))
        c = (0.+0j, forcing[k+0]*alpha_sl,
                (forcing[k+1] - forcing[k+0])/self.h_full[k+1])
        Y = (np.concatenate((np.zeros(k), y)) for y in Y)

        *_, Psi_m, _ = universal_funcs
        def f(z):
            return (z+self.z_star)*np.log(1+z/self.z_star) - z + \
                    z * Psi_m(z*inv_L_MO)

        try:
            ratio_norms = (f(self.z_full[1]) - f(0)) / \
                    (f(self.z_full[2]) - f(self.z_full[1]))
            ratio_norms = [(f(self.z_full[m+1]) - f(self.z_full[m])) / \
                    (f(self.z_full[m+2]) - f(self.z_full[m+1])) \
                        for m in range(k)]
        except ZeroDivisionError:
            ratio_norms = np.zeros(k)

        D = (np.concatenate((np.zeros(k), D[0])),
                np.concatenate((-np.ones(k), D[1])),
                np.concatenate((ratio_norms, D[2])),
                np.concatenate((np.zeros(k), D[3])))
        c = np.concatenate((np.zeros(k), c))
        return Y, D, c


    ####### DEFINITION OF SF SCHEMES : VALUE OF theta(delta_sl) ##
    # The method must use the prognostic variables and delta_sl
    # to return theta(delta_sl).
    # the prognostic variables are theta for FD and 
    # (theta_{1/2}, ... theta_{k+1/2}, phit_k, ...phit_M) for FV.
    def __sf_thetadelta_FDpure(self, prognostic, **_):
        return prognostic[0]

    def __sf_thetadelta_FD2(self, prognostic, **_):
        return (prognostic[0] + prognostic[1])/2

    def __sf_thetadelta_FVpure(self, prognostic, **_):
        return prognostic[0] - self.h_half[0]* \
                (prognostic[2] - prognostic[1])/24

    def __sf_thetadelta_FV2(self, prognostic, **_):
        return prognostic[1] - self.h_half[1] * \
                (prognostic[3]/6 + prognostic[2]/3)

    def __sf_thetadelta_FVfree(self, prognostic, delta_sl, SST,
            universal_funcs, inv_L_MO, **_):
        _, tau_slt = self.__tau_sl(delta_sl=delta_sl,
                universal_funcs=universal_funcs, inv_L_MO=inv_L_MO)
        k = bisect.bisect_right(self.z_full[1:], delta_sl)
        zk = self.z_full[k]
        tilde_h = self.z_full[k+1] - delta_sl
        alpha = tilde_h/self.h_half[k]+tau_slt
        return (prognostic[k] - tilde_h*tilde_h/self.h_half[k] * \
                (prognostic[k+1] / 3 + prognostic[k+2] / 6) - \
                (1 - alpha) * SST) \
                / alpha


    ####### DEFINITION OF SF SCHEMES : FIRST LINES OF Y,D,c (theta)
    # each method must return Y, D, c:
    # Y: 3-tuple of tuples of sizes (j-1, j, j)
    # D: 3-tuple of tuples of sizes (j-1, j, j)
    # c: tuple of size j
    # they represent the first j lines of the matrices.
    # for Y and D, the tuples are (lower diag, diag, upper diag)
    def __sf_YDc_FDpure_theta(self, K_theta, SST, u_star, t_star,
            t_delta, delta_sl, **_):
        phi_stab = -7.8 * self.kappa * delta_sl * 9.81*\
                    (t_star / t_delta) / u_star**2
        ch_du = u_star * self.kappa / (np.log(1+delta_sl/self.z_star)-phi_stab)
        Y = ((), (1.,), (0.,))
        D = ((), (-K_theta[1]/self.h_full[1]/self.h_half[0] - \
                ch_du /self.h_half[0],),
                (K_theta[1]/self.h_full[1]/self.h_half[0],))
        c = (SST*ch_du / self.h_half[0],)
        return Y, D, c

    def __sf_YDc_FD2_theta(self, K_theta, SST, u_star, t_star, t_delta,
            delta_sl, **_):
        phi_stab = -7.8 * self.kappa * delta_sl * 9.81*\
                    (t_star / t_delta) / u_star**2
        ch_du = u_star * self.kappa / (np.log(1+delta_sl/self.z_star)-phi_stab)
        Y = ((), (0.,), (0.,))
        D = ((), (-K_theta[1]/self.h_full[1] - ch_du / 2,),
                (K_theta[1]/self.h_full[1] - ch_du / 2,))
        c = (ch_du * SST,)
        return Y, D, c

    def __sf_YDc_FVpure_theta(self, K_theta, SST, u_star, t_star,
            t_delta, delta_sl, universal_funcs, **_):
        _, _, _, psi_h, _, _ = universal_funcs
        inv_L_MO = t_star / t_delta / u_star**2 * self.kappa * 9.81
        ch_du = u_star * self.kappa / \
                (np.log(1+delta_sl/self.z_star)-psi_h(delta_sl*inv_L_MO))

        Y = ((1.,), (0., 0.), (0., 0.))
        D = ((0.,), (-1, -K_theta[0] / self.h_half[0]),
                (K_theta[0]/ch_du-self.h_half[0]/24,
                    K_theta[1]/self.h_half[0]),
                (self.h_half[0]/24,))
        c = (SST, 0.)
        return Y, D, c

    def __sf_YDc_FV2_theta(self, K_theta, SST, u_star, inv_L_MO,
            delta_sl, universal_funcs, **_):
        _, _, _, psi_h, _, Psi_h = universal_funcs
        def f(z):
            return (z+self.z_star)*np.log(1+z/self.z_star) \
                - z + z* Psi_h(z*inv_L_MO)

        ch_du = u_star * self.kappa / \
                (np.log(1+delta_sl/self.z_star)-psi_h(delta_sl*inv_L_MO))

        try:
            ratio_norms = (f(self.z_full[1]) - f(0)) / \
                    (f(self.z_full[2]) - f(self.z_full[1]))
        except:
            ratio_norms = 0.
        Y = ((0., 1.), (0., 0., 0.), (0., 0., 0.))
        D = ((0., 0.), (1., -1., -K_theta[1] / self.h_half[1]),
                (-ratio_norms, K_theta[1]/ch_du + \
                        self.h_half[1]/3, K_theta[2]/self.h_half[1]),
                (0., self.h_half[1]/6))
        c = (SST*(ratio_norms-1), SST, 0.)
        return Y, D, c

    def __sf_YDc_FVfree_theta(self, K_theta, SST, u_star,
            t_delta, delta_sl, universal_funcs, inv_L_MO, forcing,
            SST_derivative, **_):
        k = bisect.bisect_right(self.z_full[1:], delta_sl)
        tilde_h = self.z_full[k+1] - delta_sl
        _, _, _, psi_h, _, Psi_h = universal_funcs
        _, tau_slt = self.__tau_sl(delta_sl=delta_sl,
                universal_funcs=universal_funcs, inv_L_MO=inv_L_MO)
        alpha_slt = tilde_h/self.h_half[k] + tau_slt
        ch_du = u_star * self.kappa / \
                (np.log(1+delta_sl/self.z_star)-psi_h(delta_sl*inv_L_MO))

        Y = ((1, tilde_h / 6 / self.h_full[k+1]),
                (0., tilde_h*tau_slt/3,
                    tilde_h/3/self.h_full[k+1] + \
                            self.h_half[k+1]/3/self.h_full[k+1]),
                (0., tilde_h*tau_slt/6,
                    self.h_half[k+1]/6/self.h_full[k+1]))

        D = ((0., K_theta[k+0]/tilde_h/self.h_full[k+1]),
                (-1., -alpha_slt*K_theta[k+0] / tilde_h,
                    -K_theta[k+1]/tilde_h/self.h_full[k+1] - \
                            K_theta[k+1] / self.h_half[k+1] / self.h_full[k+1]),
                (K_theta[k+0]*alpha_slt/ch_du + \
                        tilde_h**2 / 3 / self.h_half[k],
                    alpha_slt * K_theta[k+1]/tilde_h, K_theta[k+2]/self.h_full[k+1]/self.h_half[k+1]),
                (tilde_h**2 / 6 / self.h_half[k], 0.))
        c = (SST, forcing[k+0]*alpha_slt + (1 - alpha_slt) * SST_derivative,
                (forcing[k+1] - forcing[k+0])/self.h_full[k+1])
        Y = (np.concatenate((np.zeros(k), y)) for y in Y)

        def f(z):
            return (z+self.z_star)*np.log(1+z/self.z_star) \
                - z + z* Psi_h(z*inv_L_MO)

        try:
            ratio_norms = np.array([(f(self.z_full[m+1]) - f(self.z_full[m])) / \
                    (f(self.z_full[m+2]) - f(self.z_full[m+1])) \
                        for m in range(k)])
        except ZeroDivisionError:
            ratio_norms = np.zeros(k)

        D = (np.concatenate((np.zeros(k), D[0])),
                np.concatenate((np.ones(k), D[1])),
                np.concatenate((-ratio_norms, D[2])),
                np.concatenate((np.zeros(k), D[3])))
        c = np.concatenate(((ratio_norms - 1)*SST, c))
        return Y, D, c
