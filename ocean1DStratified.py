"""
    This module defines the class Ocean1dStratified
    which simulates a 1D Ekman stratified problem with various
    discretisations.
"""

from typing import Tuple, List, NamedTuple
import bisect
import numpy as np
from tqdm import tqdm
from scipy import integrate
from utils_linalg import multiply, scal_multiply as s_mult
from utils_linalg import add_banded as add
from utils_linalg import solve_linear, orientation
from utils_linalg import full_to_half
from universal_functions import Businger_et_al_1971 as businger
from universal_functions import Large_et_al_2019 as large_ocean
from shortwave_absorption import shortwave_fractional_decay, \
        Qsw_E, shortwave_frac_sl
from bulk import SurfaceLayerData, friction_scales, \
        process_friction_scales_oce

array = np.ndarray

class Ocean1dStratified():
    """
        main class, instanciate it to run the simulations.
    """
    def __init__(self, z_levels: array,
            N0: float, alpha: float, dt: float=30.,
            u_geostrophy: float=10.,
            K_mol: float=1e-4, C_p: float=3985., f: float=1e-4,
            loading_bar: bool=False) -> None:
        """
            z_levels starts at bottom of ocean and contains
            all the full levels $z_m$ until $z_M=0$.
            dt is the time step
            u_geostrophy is used as a top boundary condition
            K_mol is the background diffusivity
            f is the Coriolis parameter
        """
        assert z_levels.shape[0] > 2 and abs(z_levels[-1]) < 1e-3

        # X_full[m] represents in LaTeX $X_m$ whereas
        # X_half[m] represents in LaTeX $X_{m+1/2}$.
        # both have M+1 points (0<=m<=M).
        self.z_full: array = np.copy(z_levels)
        self.h_half: array = np.diff(self.z_full,
                append=2*self.z_full[-1] - self.z_full[-2])
        self.z_half: array = self.z_full + self.h_half/2
        self.h_full: array = np.diff(self.z_half,
                prepend=2*z_levels[0] - self.z_half[0])
        self.M: int = self.z_full.shape[0] - 1
        self.loading_bar: bool = loading_bar
        self.dt: float = dt
        self.K_mol: float = K_mol
        self.kappa: float = 0.4
        self.u_g: float = u_geostrophy
        self.f: float = f
        self.C_p: float = C_p
        self.e_min: float = 1e-6 # min value of tke
        self.e0_min: float = 1e-4 # min value of surface tke
        self.Ku_min: float = 1e-4
        self.Ktheta_min: float = 1e-5
        self.C_m: float = 0.1
        self.C_s: float = 0.143
        self.C_1: float = 0.143
        self.C_e: float = self.C_m # used to be 0.34
        self.c_eps: float = np.sqrt(2.)/2.
        self.rho0: float = 1024.
        self.N0: float = N0
        self.alpha: float = alpha
        self.dTdz_bot: float = N0*N0/(alpha*9.81)
        self.implicit_coriolis: float = 0.55 # semi-implicit coefficient
        self.mxl_min: float = self.K_mol / ( self.C_m * np.sqrt(self.e_min) )
        self.dict_tau_sl = {}
        # For each name of sf_scheme, two corresponding
        # methods defined at the bottom of this class:
        # sf_udelta_* is how we compute u(delta_sl) and
        # sf_YDc_* is the definition of the bottom boundary condition
        self.dictsf_scheme = {
                "FD pure" : (self.__sf_udelta_FDpure,
                                self.__sf_YDc_FDpure),
                "FD2" : (self.__sf_udelta_FD2,
                                self.__sf_YDc_FD2),
                "FD test" : (self.__sf_udelta_FDtest,
                                self.__sf_YDc_FDtest),
                "FV pure" : (self.__sf_udelta_FVpure,
                                self.__sf_YDc_FVpure),
                "FV1" : (self.__sf_udelta_FV1,
                                self.__sf_YDc_FV1),
                "FV test" : (self.__sf_udelta_FVtest,
                                self.__sf_YDc_FVtest),
                "FV free" : (self.__sf_udelta_FVfree,
                                self.__sf_YDc_FVfree)}
        self.dictsf_scheme_theta = {
                "FD pure" : (self.__sf_thetadelta_FDpure,
                                self.__sf_YDc_FDpure_theta),
                "FD2" : (self.__sf_thetadelta_FD2,
                                self.__sf_YDc_FD2_theta),
                "FD test" : (self.__sf_thetadelta_FDtest,
                                self.__sf_YDc_FDtest_theta),
                "FV pure" : (self.__sf_thetadelta_FVpure,
                                self.__sf_YDc_FVpure_theta),
                "FV1" : (self.__sf_thetadelta_FV1,
                                self.__sf_YDc_FV1_theta),
                "FV test" : (self.__sf_thetadelta_FVtest,
                                self.__sf_YDc_FVtest_theta),
                "FV free" : (self.__sf_thetadelta_FVfree,
                                self.__sf_YDc_FVfree_theta)
                }

    def FV(self, u_t0: array, phi_t0: array, theta_t0: array,
            dz_theta_t0: array, Q_sw: array, Q_lw: array,
            u_delta: float, t_delta: float, delta_sl_a: float,
            u_star: array, t_star: array,
            wind_10m: array, temp_10m: array,
            delta_sl: float=None,
            sf_scheme: str="FV test", Neutral_case: bool=False,
            turbulence: str="TKE", store_all: bool=False):
        """
            Integrates in time with Backward Euler the model with TKE
            and Finite volumes.

            u_t0: average of cells (centered on self.z_half[:-1])
            phi_t0: derivative of u (given at self.z_full)
            theta_t0: average of cells (centered like u_t0)
            dz_theta_t0: derivative of theta (given like phi_t0)
            u_delta: for FV free, it is necessay to have u(delta)
            t_delta: for FV free, it is necessay to have theta(delta)
            Q_sw: SW radiation (from the sun) positive downward
            Q_lw: LW radiation (blackbody radiation) positive downward
            wind_10m, temp_10m: wind, temperature at 10m in atmosphere
            delta_sl: height of the surface layer.
                Make sure it is coherent with sf_scheme.
            sf_scheme is the surface flux scheme:
                "FV pure", "FV1", "FV2" or "FV free"
            {u, t}_delta: value of {u, t} at z=delta_sl
            turbulence should be one of:
                - "TKE" (for 1.5-equation TKE)
                - "KPP" (for simple K-profile parametrization)
            If Neutral_case is True, no temperature profile
            is computed and t_star = 0
        """
        assert u_t0.shape[0] == self.M
        assert phi_t0.shape[0] == self.M + 1
        assert sf_scheme in self.dictsf_scheme_theta
        assert sf_scheme in self.dictsf_scheme
        if delta_sl is None:
            delta_sl = self.z_full[-2] if sf_scheme == "FV2" \
                    else self.z_half[-2]
        if sf_scheme in {"FV2",}:
            assert abs(delta_sl - self.z_full[-2]) < 1e-10
        elif sf_scheme in {"FV1", "FV pure"}:
            assert abs(delta_sl - self.z_full[-2]/2) < 1e-10
        assert turbulence in {"TKE", "KPP"}
        N: int = wind_10m.shape[0] - 1 # number of time steps
        assert temp_10m.shape[0] == N + 1
        assert Q_sw.shape[0] == Q_lw.shape[0] == N + 1

        forcing = 1j*self.u_g*np.ones((N+1, self.M))

        k = bisect.bisect_left(self.z_full, delta_sl)

        SL = process_friction_scales_oce(ua_delta=wind_10m[0],
                delta_sl_a=delta_sl_a, ta_delta=temp_10m[0],
                univ_funcs_a=businger(),
                u_star=u_star[0], t_star=t_star[0],
                uo_delta=u_delta, delta_sl_o=delta_sl,
                to_delta=t_delta, univ_funcs_o=large_ocean(),
                sf_scheme=sf_scheme, Q_sw=Q_sw[0], Q_lw=Q_lw[0],
                k=k, is_atm=False)

        ignore_tke_sl = sf_scheme in {"FV pure", "FV1", "FV test"}

        import tkeOcean1D
        wave_breaking = False
        tke = tkeOcean1D.TkeOcean1D(self.M, "FV",
                ignore_sl=ignore_tke_sl,
                wave_breaking=wave_breaking)

        theta, dz_theta = np.copy(theta_t0), np.copy(dz_theta_t0)

        Ku_full: array = self.Ku_min + np.zeros(self.M+1)
        Ktheta_full: array = self.Ktheta_min + np.zeros(self.M+1)

        z_levels_sl = np.copy(self.z_full)
        z_levels_sl[k] = self.z_full[k] if ignore_tke_sl else delta_sl
        l_m = self.mxl_min*np.ones(self.M+1)
        l_eps = self.mxl_min*np.ones(self.M+1)

        phi, old_phi = phi_t0, np.copy(phi_t0)
        u_current: array = np.copy(u_t0)
        all_u_star = []
        ret_u_delta, ret_t_delta = [u_delta], [t_delta]
        ret_u_current, ret_tke, ret_SL = [], [], []
        ret_phi, ret_theta, ret_dz_theta, ret_leps = [], [], [], []

        for n in tqdm(range(1,N+1), total=N, leave=False) \
                if self.loading_bar else range(1,N+1):
            # Compute friction scales:
            SL_nm1 = SL
            SL = process_friction_scales_oce(ua_delta=wind_10m[n],
                    delta_sl_a=delta_sl_a, ta_delta=temp_10m[n],
                    univ_funcs_a=businger(),
                    u_star=u_star[n], t_star=t_star[n],
                    uo_delta=u_delta, delta_sl_o=delta_sl,
                    to_delta=t_delta, univ_funcs_o=large_ocean(),
                    sf_scheme=sf_scheme, Q_sw=Q_sw[n], Q_lw=Q_lw[n],
                    k=k, is_atm=False)
            all_u_star += [SL.u_star]

            # Compute viscosities
            Ku_full, Ktheta_full, = self.__visc_turb_FV(
                    SL, turbulence=turbulence, phi=phi,
                    old_phi=old_phi, l_m=l_m, l_eps=l_eps,
                    K_full=Ku_full, tke=tke,
                    dz_theta=dz_theta, Ktheta_full=Ktheta_full,
                    universal_funcs=large_ocean(),
                    ignore_tke_sl=ignore_tke_sl, tau_b=0.,
                    wave_breaking=wave_breaking)

            old_phi = phi
            # integrate in time momentum
            u_current, phi = self.__step_u(u=u_current,
                    phi=phi, Ku_full=Ku_full,
                    forcing=forcing[n], SL=SL, SL_nm1=SL_nm1)

            if not Neutral_case:
                # integrate in time potential temperature
                absorption_sl = shortwave_frac_sl(delta_sl)
                if sf_scheme != "FV free":
                    absorption_sl = 1. # for FV pure we want
                    # to have a forcing that takes effect in the SL
                swr_frac = shortwave_fractional_decay(self.M,
                        np.diff(z_levels_sl)) * absorption_sl
                forcing_theta = np.diff(swr_frac * Q_sw[n] \
                        / self.rho0 / self.C_p) / self.h_half[:-1]
                theta, dz_theta = self.__step_theta(theta,
                        dz_theta, Ktheta_full, forcing_theta,
                        SL, SL_nm1)

            # Refreshing u_delta, t_delta for next friction scales:
            func_un, _ = self.dictsf_scheme[sf_scheme]
            func_theta, _ = self.dictsf_scheme_theta[sf_scheme]
            prognostic_u: array = np.concatenate((phi[0:SL.k+1],
                u_current[SL.k-1:]))
            prognostic_theta: array = np.concatenate((dz_theta[0:SL.k+1],
                theta[SL.k-1:]))
            u_delta = func_un(prognostic=prognostic_u,
                    delta_sl=delta_sl, SL=SL,
                    universal_funcs=large_ocean())
            t_delta = func_theta(prognostic=prognostic_theta, SL=SL,
                    universal_funcs=large_ocean())
            ret_u_delta += [u_delta]
            ret_t_delta += [t_delta]

            if store_all:
                ret_u_current += [np.copy(u_current)]
                ret_tke += [np.copy(tke.tke_full)]
                ret_phi += [np.copy(phi)]
                ret_theta += [np.copy(theta)]
                ret_dz_theta += [np.copy(dz_theta)]
                ret_leps += [np.copy(l_eps)]
                ret_SL += [SL]

        ret_dict = {'u_delta' : ret_u_delta,
                't_delta': ret_t_delta,}
        if store_all:
            ret_dict['all_u'] = ret_u_current
            ret_dict['all_phi'] = ret_phi
            ret_dict['all_tke'] = ret_tke
            ret_dict['all_theta'] = ret_theta
            ret_dict['all_dz_theta'] = ret_dz_theta
            ret_dict['all_leps'] = ret_leps
            ret_dict['all_SL'] = ret_SL

        ret_dict['u'] = u_current
        ret_dict['phi'] = phi
        ret_dict['tke'] = tke.tke_full
        ret_dict['all_u_star'] = all_u_star
        ret_dict['theta'] = theta
        ret_dict['dz_theta'] = dz_theta
        ret_dict['l_eps'] = l_eps
        ret_dict['SL'] = SL
        ret_dict['Ktheta'] = Ktheta_full
        return ret_dict

    def FD(self, u_t0: array, theta_t0: array,
            Q_sw: array, Q_lw: array,
            wind_10m: array, temp_10m: array,
            u_star: array, t_star: array,
            delta_sl_a: float=10.,
            turbulence: str="TKE", sf_scheme: str="FD pure",
            Neutral_case: bool=False, store_all: bool=False):
        """
            Integrates in time with Backward Euler the model with KPP
            and Finite differences.

            u_t0 should be given at half-levels (follow self.z_half)
            forcing should be given at half-levels for all times
            wind_10m, temp_10m: wind, temperature at 10m in atmosphere
            SST: Surface Temperature for all times
            turbulence should be one of:
                "TKE" (for 1.5-equation TKE)
                "KPP" (for simple K-profile parametrization)
            sf_scheme is the surface flux scheme:
                "FD pure" (delta_sl = z_1/2) or
                "FD2"     (delta_sl = z_1)
            If Neutral_case is True, no temperature profile
            is computed and t_star = 0
        """
        assert u_t0.shape[0] == self.M
        assert sf_scheme in self.dictsf_scheme_theta
        assert sf_scheme in self.dictsf_scheme
        assert sf_scheme in {"FD2", "FD pure", "FD test"}
        assert turbulence in {"TKE", "KPP"}
        N: int = wind_10m.shape[0] - 1 # number of time steps
        assert Q_sw.shape[0] == N + 1
        assert Q_lw.shape[0] == N + 1
        assert temp_10m.shape[0] == N + 1

        forcing = 1j*self.u_g*np.ones((N+1, self.M))
        # methods to get u(delta) and theta(delta):
        func_un, _ = self.dictsf_scheme[sf_scheme]
        func_theta, _ = self.dictsf_scheme_theta[sf_scheme]
        # height of the surface layer:
        delta_sl = self.z_half[self.M-1] if sf_scheme == "FD pure" \
                else self.z_full[self.M-1]
        ###### Initialization #####
        import tkeOcean1D
        wave_breaking = False
        tke = tkeOcean1D.TkeOcean1D(self.M, "FD",
                wave_breaking=wave_breaking)
        theta: array = np.copy(theta_t0)
        # Initializing viscosities and mixing lengths:
        Ku_full: array = self.Ku_min + np.zeros(self.M+1)
        Ktheta_full: array = self.Ktheta_min + np.zeros(self.M+1)
        l_m = self.mxl_min*np.ones(self.M+1)
        l_eps = self.mxl_min*np.ones(self.M+1)

        u_current: array = np.copy(u_t0)
        old_u: array = np.copy(u_current)
        all_u_star = []
        all_u, all_tke, all_theta, all_leps = [], [], [], []
        ret_u_delta, ret_t_delta = [], []
        for n in tqdm(range(1,N+1), total=N, leave=False) \
                if self.loading_bar else range(1,N+1):
            u_delta = func_un(prognostic=u_current, delta_sl=delta_sl)
            t_delta = func_theta(prognostic=theta)
            SL = process_friction_scales_oce(ua_delta=wind_10m[n],
                    delta_sl_a=delta_sl_a, ta_delta=temp_10m[n],
                    univ_funcs_a=businger(),
                    u_star=u_star[n], t_star=t_star[n],
                    uo_delta=u_delta, delta_sl_o=delta_sl,
                    to_delta=t_delta, univ_funcs_o=large_ocean(),
                    sf_scheme=sf_scheme, Q_sw=Q_sw[n], Q_lw=Q_lw[n],
                    k=self.M, is_atm=False)
            all_u_star += [SL.u_star]

            # Compute viscosities:
            Ku_full, Ktheta_full = self.__visc_turb_FD(SL=SL,
                    u_current=u_current, old_u=old_u,
                    K_full=Ku_full, tke=tke, theta=theta,
                    Ktheta_full=Ktheta_full, l_m=l_m, l_eps=l_eps,
                    universal_funcs=large_ocean(), tau_b=0.,
                    wave_breaking=wave_breaking)

            # integrate in time momentum:
            Y, D, c = self.__matrices_u_FD(Ku_full, forcing[n])
            self.__apply_sf_scheme(Y=Y, D=D, c=c, K_u=Ku_full,
                    func=self.dictsf_scheme[sf_scheme][1],
                    forcing=forcing[n], SL=SL)
            next_u = self.__backward_euler(Y=Y, D=D, c=c,
                    u=u_current, f=self.f)
            u_current, old_u = next_u, u_current

            if not Neutral_case:
                swr_frac = shortwave_fractional_decay(self.M,
                        self.h_half) # Q_sw is positive downward
                forcing_theta = np.diff(swr_frac * Q_sw[n] \
                        / self.rho0 / self.C_p) / self.h_half[:-1]
                # integrate in time potential temperature:
                Y_theta, D_theta, c_theta = self.__matrices_theta_FD(
                        Ktheta_full, forcing_theta)
                self.__apply_sf_scheme(\
                        func=self.dictsf_scheme_theta[sf_scheme][1],
                        Y=Y_theta, D=D_theta, c=c_theta, SL=SL,
                        K_theta=Ktheta_full, forcing_theta=forcing_theta,
                        universal_funcs=large_ocean(),
                        universal_funcs_a=businger())

                theta = np.real(self.__backward_euler(Y=Y_theta,
                        D=D_theta, c=c_theta, u=theta, f=0.))
            ret_u_delta += [u_delta]
            ret_t_delta += [t_delta]

            if store_all:
                all_u += [np.copy(u_current)]
                all_tke += [np.copy(tke.tke_full)]
                all_theta += [np.copy(theta)]
                all_leps += [np.copy(l_eps)]

        ret_u_delta += [func_un(prognostic=u_current,
                                delta_sl=delta_sl)]
        ret_t_delta += [func_theta(prognostic=theta)]

        ret_dict = {'u_delta' : ret_u_delta,
                't_delta': ret_t_delta,}
        if store_all:
            ret_dict['all_u'] = all_u
            ret_dict['all_tke'] = all_tke
            ret_dict['all_theta'] = all_theta
            ret_dict['all_leps'] = all_leps

        ret_dict['u'] = u_current
        ret_dict['tke'] = tke.tke_full
        ret_dict['all_u_star'] = all_u_star
        ret_dict['theta'] = theta
        ret_dict['l_eps'] = l_eps
        ret_dict['SL'] = SL
        ret_dict['Ktheta'] = Ktheta_full
        return ret_dict

    def __step_u(self, u: array, phi: array,
            Ku_full: array, forcing: array,
            SL: SurfaceLayerData, SL_nm1: SurfaceLayerData):
        """
        One step of integration in time for the momentum (FV).
        u: average on the cells (centered at z_half),
        phi: space derivative of u (located at z_full),
        Ku_full: viscosity for u (located at z_full),
        forcing: average of the forcing on each cell (z_half)
        SL: Data of the surface layer at current time
        SL_nm1: Data of the surface layer at previous time
        """
        func_un, func_YDc = self.dictsf_scheme[SL.sf_scheme]
        prognostic: array = np.concatenate((phi[0:SL.k+1],
            u[SL.k-1:]))
        Y, D, c = self.__matrices_u_FV(Ku_full, forcing)
        Y_nm1 = tuple(np.copy(y) for y in Y)
        self.__apply_sf_scheme(\
                func=func_YDc, Y=Y, D=D, c=c, K_u=Ku_full,
                forcing=forcing, SL=SL, SL_nm1=SL_nm1, Y_nm1=Y_nm1,
                universal_funcs=large_ocean())

        prognostic = self.__backward_euler(Y=Y, D=D, c=c,
                u=prognostic, f=self.f, Y_nm1=Y_nm1)

        next_u = np.zeros_like(u) + 0j
        next_u[:SL.k] = 1/(1+self.dt*1j*self.f*self.implicit_coriolis) * \
                ((1 - self.dt*1j*self.f*(1-self.implicit_coriolis)) * \
                u[:SL.k] + self.dt * \
                (np.diff(prognostic[:SL.k+1] * Ku_full[:SL.k+1]) / self.h_half[:SL.k] \
                + forcing[:SL.k]))

        next_u[SL.k-1:] = prognostic[SL.k+1:]
        phi = prognostic[:SL.k+1]
        if SL.k < self.M: # constant flux layer : K[:k] phi[:k] = K0 phi0
            phi = np.concatenate(( phi, Ku_full[SL.k]* \
                    prognostic[SL.k]/Ku_full[SL.k+1:]))

        return next_u, phi

    def __step_theta(self, theta: array, dz_theta: array,
            Ktheta_full: array, forcing_theta: array,
            SL: SurfaceLayerData, SL_nm1: SurfaceLayerData):
        """
        One step of integration in time for potential temperature (FV)
        theta: average on the cells (centered at z_half),
        dz_theta: space derivative of theta (located at z_full),
        Ktheta_full: viscosity for theta (located at z_full),
        forcing_theta: average of the forcing on each cell (z_half)
        SL: Data of the surface layer at current time
        SL_nm1: Data of the surface layer at previous time
        """
        prognostic_theta: array = np.concatenate((
            dz_theta[0:SL.k+1], theta[SL.k-1:]))
        Y_theta, D_theta, c_theta = self.__matrices_theta_FV(
                Ktheta_full, forcing_theta)
        Y_nm1 = tuple(np.copy(y) for y in Y_theta)
        self.__apply_sf_scheme(\
                func=self.dictsf_scheme_theta[SL.sf_scheme][1],
                Y=Y_theta, D=D_theta, c=c_theta, Y_nm1=Y_nm1,
                K_theta=Ktheta_full, forcing_theta=forcing_theta,
                universal_funcs=large_ocean(),
                universal_funcs_a=businger(), SL=SL, SL_nm1=SL_nm1)
        prognostic_theta[:] = np.real(self.__backward_euler(Y=Y_theta,
                D=D_theta, c=c_theta, u=prognostic_theta, f=0.,
                Y_nm1=Y_nm1))

        next_theta = np.zeros_like(theta)
        next_theta[:SL.k] = theta[:SL.k] + self.dt * \
                (np.diff(prognostic_theta[:SL.k+1] * \
                        Ktheta_full[:SL.k+1]) \
                        / self.h_half[:SL.k] + forcing_theta[:SL.k])

        next_theta[SL.k-1:] = prognostic_theta[SL.k+1:]
        dz_theta = prognostic_theta[:SL.k+1]
        if SL.k < self.M: # const flux layer : K[:k] phi[:k] = K[0] phi[0]
            dz_theta = np.concatenate(( dz_theta, Ktheta_full[SL.k]* \
                    prognostic_theta[SL.k]/Ktheta_full[SL.k+1:]))
        func_theta, _ = self.dictsf_scheme_theta[SL.sf_scheme]
        return next_theta, dz_theta

    def __mixing_lengths(self, tke: array, dzu2: array, N2: array,
            z_levels: array, SL: SurfaceLayerData, universal_funcs,
            wave_breaking: bool):
        """
            returns the mixing lengths (l_m, l_eps)
            for given entry parameters.
            dzu2 =||du/dz||^2 should be computed similarly
            to the shear
            all input and output are given on full levels.
            z_levels[k] can be set to delta_sl
            for a better link with the surface layer.
        """
        l_down = self.mxl_min * np.ones_like(z_levels)
        l_up = np.copy(l_down)
        assert z_levels[-1] >= SL.delta_sl
        # to take into account surface layer we allow to change
        # z levels in this method
        h_half = np.diff(z_levels)
        buoyancy = np.maximum(1e-20, N2)
        mxlm = np.maximum(self.mxl_min,
                np.sqrt(2.*tke) / np.sqrt(buoyancy))
        mxlm[0] = self.mxl_min
        # should not exceed linear mixing lengths:
        for j in range(1, self.M+1):
            l_down[j] = min(l_down[j-1] + h_half[j-1], mxlm[j])

        g = 9.81
        z_sl = z_levels[SL.k:]
        phi_m, *_ = universal_funcs
        mxlm[SL.k:] = 1/l_down[SL.k:] / tke[SL.k:] * (SL.u_star * \
                self.kappa * (-z_sl + SL.z_0M) / self.C_m / \
                phi_m(-z_sl * SL.inv_L_MO))**2
        # surface wave breaking parameterization:
        if wave_breaking:
            mxl0 = 0.04
            l_up[SL.k:] = max(mxl0,
                    np.abs(SL.u_star**2)*self.kappa*2e5/9.81)
        else:
            l_up[SL.k:] = mxlm[SL.k:]
        # limiting l_up with the distance to the surface:
        for j in range(SL.k - 1, -1, -1):
            l_up[j] = min(l_up[j+1] + h_half[j], mxlm[j])

        l_m = np.sqrt(l_up*l_down)
        l_eps = np.minimum(l_down, l_up)
        return l_m, l_eps


    def reconstruct_FV(self, u_bar: array, phi: array, theta: array,
            dz_theta: array, SL: SurfaceLayerData,
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
        z_max = -0.02
        xi = [np.linspace(-h/2, h/2, 15) for h in self.h_half[:-1]]
        sub_discrete: List[array] = [u_bar[m] + (phi[m+1] + phi[m]) * xi[m]/2 \
                + (phi[m+1] - phi[m]) / (2 * self.h_half[m]) * \
                (xi[m]**2 - self.h_half[m]**2/12) for m in range(self.M)]

        sub_discrete_theta: List[array] = [theta[m] + (dz_theta[m+1] + dz_theta[m]) * xi[m]/2 \
                + (dz_theta[m+1] - dz_theta[m]) / (2 * self.h_half[m]) * \
                (xi[m]**2 - self.h_half[m]**2/12) for m in range(self.M)]
        u_star, t_star, _, _, inv_L_MO, _, _, u_zM, t_zM, \
                delta_sl, k1, sf_scheme, Q_sw, Q_lw, _ = SL
        if sf_scheme in {"FV1", "FV pure", "FV test"} \
                or ignore_loglaw or delta_sl >= z_max:
            allxi = [np.array(xi[m]) + self.z_half[m] for m in range(self.M)]
            k_1m: int = bisect.bisect_left(allxi[-1], z_max)
            allxi[-1] = allxi[-1][:k_1m]
            sub_discrete_theta[-1] = sub_discrete_theta[-1][:k_1m]
            sub_discrete[-1] = sub_discrete[-1][:k_1m]
            return np.concatenate(allxi), \
                    np.concatenate(sub_discrete), \
                    np.concatenate(sub_discrete_theta), \

        if delta_sl is None:
            delta_sl = self.z_full[-2] if sf_scheme == "FV2" \
                    else self.z_half[-2]

        u_oversampled = np.concatenate(sub_discrete[:-1])
        theta_oversampled = np.concatenate(sub_discrete_theta[:-1])
        z_oversampled = np.concatenate([np.array(xi[m]) +\
                self.z_half[m] for m in range(self.M)])

        prognostic_u: array = np.concatenate((phi[0:SL.k+1],
            u_bar[SL.k-1:]))
        prognostic_t: array = np.concatenate((dz_theta[0:SL.k+1],
            theta[SL.k-1:]))
        # getting information of the surface layer (from which
        # we get the MOST profiles)
        z_log: array = np.geomspace(delta_sl, z_max, 20)

        _, phi_h, psi_m, psi_h, *_ = large_ocean()

        func_un, _ = self.dictsf_scheme[SL.sf_scheme]
        func_theta, _ = self.dictsf_scheme_theta[SL.sf_scheme]

        u_delta: complex = func_un(prognostic=prognostic_u,
                SL=SL, universal_funcs=large_ocean())
        t_delta: complex = func_theta(prognostic=prognostic_t,
                SL=SL, universal_funcs=large_ocean())

        abs_uzM_m_uz: array = u_star/self.kappa * \
                (np.log(1-z_log/SL.z_0M) - psi_m(-z_log*inv_L_MO))
        abs_uzM_m_udelta: array = u_star/self.kappa * \
                (np.log(1-delta_sl/SL.z_0M) - \
                psi_m(-delta_sl*inv_L_MO))
        u_log: array = u_zM - abs_uzM_m_uz / abs_uzM_m_udelta * \
                (u_zM - u_delta)

        def tzM_m_t(z: float):
            QH = t_star * u_star * self.rho0 * self.C_p
            term_lw = 1 + SL.Q_lw / QH
            term_sw = Qsw_E(z, SL) / QH
            return t_star/self.kappa * term_lw * \
                    (np.log(1-z/SL.z_0H) - psi_h(-z*inv_L_MO)) \
                    - term_sw
        tzM_m_t = np.vectorize(tzM_m_t, otypes=[float])

        theta_log: array = SL.t_0 - tzM_m_t(z_log) / \
                tzM_m_t(delta_sl) * (t_zM - t_delta)

        # index of z_{k-1} in z_oversampled:
        k2: int = bisect.bisect_left(z_oversampled, self.z_full[k1-1])

        z_freepart = []
        u_freepart = []
        theta_freepart = []

        if sf_scheme in {"FV free", }:
            # between the log profile and the next grid level:
            tilde_h = delta_sl - self.z_full[k1-1]
            assert 0 < tilde_h <= self.h_half[k1-1]
            xi = np.linspace(-tilde_h/2, tilde_h/2, 15)
            tau_slu, tau_slt = self.__tau_sl(SL, large_ocean())
            alpha_slu = tilde_h/self.h_half[k1-1] + tau_slu
            alpha_slt = tilde_h/self.h_half[k1-1] + tau_slt

            u_tilde = 1/alpha_slu * (u_bar[k1-1] - tilde_h * tau_slu * \
                    (phi[k1]/3 + phi[k1-1]/6) - (1-alpha_slu)*u_zM)
            theta_tilde = 1/alpha_slt * (theta[k1-1] - \
                    tilde_h * tau_slt * \
                    (dz_theta[k1]/3 + dz_theta[k1-1]/6) - \
                    (1-alpha_slt)*t_zM)

            u_freepart = u_tilde + (phi[k1] + phi[k1-1]) * xi/2 \
                    + (phi[k1] - phi[k1-1]) / (2 * tilde_h) * \
                    (xi**2 - tilde_h**2/12)

            theta_freepart = theta_tilde \
                    + (dz_theta[k1] + dz_theta[k1-1]) * xi/2 \
                    + (dz_theta[k1] - dz_theta[k1-1]) / (2 * tilde_h)\
                    * (xi**2 - tilde_h**2/12)

            z_freepart = delta_sl + xi - tilde_h / 2
        elif sf_scheme in {"FV2"}: # link log cell with profile
            k2: int = bisect.bisect_left(z_oversampled,
                    self.z_full[k1])

        return np.concatenate((z_oversampled[:k2], z_freepart, z_log)), \
    np.concatenate((u_oversampled[:k2], u_freepart, u_log)), \
                np.concatenate((theta_oversampled[:k2],
                    theta_freepart, theta_log))


    def __backward_euler(self, Y: Tuple[array, array, array],
                        D: Tuple[array, array, array], c: array,
                        u: array, f:float=0., Y_nm1=None):
        """
            if it's for $u$, set f=self.f otherwise f=0
            integrates once (self.dt) in time the equation
            (partial_t + if)Yu - dt*Du = c
            The scheme is:
            Y(1+dt*if*gamma) - D) u_np1 = Y u + dt*c + dt*if*(1-gamma)
            with gamma the coefficient of implicitation of Coriolis.
            If partial_t Y is not zero, then Y_nm1 != Y can be given.
        """
        if Y_nm1 is None:
            Y_nm1=Y
        to_inverse: Tuple[array, array, array] = add(s_mult(Y,
                                1 + self.implicit_coriolis * \
                            self.dt * 1j*f), s_mult(D, - self.dt))
        return solve_linear(to_inverse,
                (1 - (1 - self.implicit_coriolis) * self.dt*1j*f) * \
                                multiply(Y_nm1, u) + self.dt*c)

    def __visc_turb_FD(self, SL: SurfaceLayerData,
            turbulence: str="TKE", u_current: array=None,
            old_u: array=None, K_full: array=None,
            tke=None, theta: array=None,
            Ktheta_full: array=None, l_m: array=None,
            l_eps: array=None, universal_funcs=None,
            tau_b: float=None, wave_breaking: bool=False):
        """
            Computes the turbulent viscosity on full levels K_full.
            It differs between FD and FV because of the
            shear computation (in the future, other differences
            might appear like the temperature handling).
            returns (K_full, TKE)
        """
        u_star, delta_sl = SL.u_star, SL.delta_sl
        if turbulence =="KPP":
            c_1 = 0.2 # constant for h_cl=c_1*u_star/f
            assert abs(self.f) > 1e-10
            h_cl: float = c_1*u_star/self.f
            G_full: array = self.kappa * np.abs(self.z_full) * \
                    (1 - np.abs(self.z_full)/h_cl)**2 \
                    * np.heaviside(1 - np.abs(self.z_full)/h_cl, 1)

            K_full: array = u_star * G_full + self.K_mol # viscosity
            K_full[0] = u_star * self.kappa * (-delta_sl + SL.z_0M)
        elif turbulence == "TKE":
            ####### TKE SCHEME #############
            du = np.diff(u_current)
            du_old = np.diff(old_u)
            dzu2 = np.concatenate(([0],
                [np.abs(du[m-1] / self.h_full[m]**2  * \
                        (du[m-1]+du_old[m-1])/2) \
                        for m in range(1, self.M)], [0]))
            dz_theta = np.diff(theta) / self.h_full[1:-1]
            # linear equation of state:
            # rho1 = rho0 * (1 - alpha * theta_full - T0)
            # dzrho1 = rho0 * - alpha * dz_theta
            g = 9.81
            dz_rho = - self.rho0 * self.alpha * dz_theta
            N2 = -g/self.rho0 * dz_rho
            N2 = np.concatenate(([N2[0]], -g/self.rho0 * dz_rho,
                [N2[-1]]))

            tke.integrate_tke(self, SL, universal_funcs,
                    dzu2*K_full, K_full, l_eps, Ktheta_full, N2,
                    self.rho0*SL.u_star**2, tau_b)
            l_m[:], l_eps[:] = self.__mixing_lengths(tke.tke_full,
                    dzu2, N2, self.z_full, SL, universal_funcs,
                    wave_breaking)

            apdlr = self.__stability_temperature_phi_z(\
                    N2, K_full, dzu2*K_full)

            Ktheta_full: array = np.maximum(self.Ktheta_min,
                    self.C_m * apdlr * l_m * np.sqrt(tke.tke_full))
            Ku_full: array = np.maximum(self.Ku_min,
                    self.C_m * l_m * np.sqrt(tke.tke_full))
        else:
            raise NotImplementedError("Wrong turbulence scheme")
        return Ku_full, Ktheta_full

    def __stability_temperature_phi_z(self, N2_full: array,
            Ku_full: array, shear_full: array):
        # local Richardson number:
        zri = np.maximum(N2_full, 0.) * Ku_full / \
                (shear_full + 1e-20)
        # critical Richardson number:
        ri_cri = 2. / ( 2. + self.c_eps / self.C_m )
        return np.maximum(0.1, ri_cri / np.maximum(ri_cri , zri))


    def __matrices_u_FD(self, K_full: array, forcing: array):
        """
            Creates the matrices D, Y, c such that the
            semi-discrete in space Ekman stratified equation 
            for the momentum writes ((d/dt+if) Y - D) u = c
        """
        D_diag: array = np.concatenate((
            [-K_full[1] / self.h_full[1] / self.h_half[0]],
            [(-K_full[m+1]/self.h_full[m+1] - K_full[m]/self.h_full[m]) \
                    / self.h_half[m] for m in range(1, self.M-1)],
            [-K_full[self.M-1]/self.h_half[self.M-1]/self.h_full[self.M-1]]))
        D_udiag: array = np.array(
            [K_full[m+1]/self.h_full[m+1] / self.h_half[m]
                for m in range(self.M-1)])
        D_ldiag: array = np.array(
            [K_full[m]/self.h_full[m] / self.h_half[m]
                for m in range(1, self.M)])

        c: array = np.copy(forcing)
        dzu_bottom: float = 0.
        c[0] += K_full[0] * dzu_bottom / self.h_half[0]
        Y = (np.zeros(self.M-1), np.ones(self.M), np.zeros(self.M-1))
        D = D_ldiag, D_diag, D_udiag
        return Y, D, c

    def __matrices_theta_FD(self, K_full: array, forcing: array):
        """
            Creates the matrices D, Y, c such that the
            semi-discrete in space Ekman Stratified equation
            for the temperature writes (d/dt Y - D) u = c
        """
        D_diag: array = np.concatenate((
            [-K_full[1] / self.h_full[1] / self.h_half[0]],
            [(-K_full[m+1]/self.h_full[m+1] - \
                    K_full[m]/self.h_full[m]) \
                    / self.h_half[m] for m in range(1, self.M-1)],
            [-K_full[self.M-1] / self.h_half[self.M-1] / \
                    self.h_full[self.M-1]]))
        D_udiag: array = np.array([K_full[m+1]/self.h_full[m+1] / \
                self.h_half[m] for m in range(self.M-1)])
        D_ldiag: array = np.array(
            [K_full[m]/self.h_full[m] / self.h_half[m]
                for m in range(1, self.M)])

        c: array = np.copy(forcing)
        dztheta_bottom = 0. # Gamma_T
        c[0] -= K_full[0] * dztheta_bottom/self.h_half[0]
        Y = (np.zeros(self.M-1), np.ones(self.M), np.zeros(self.M-1))
        D = D_ldiag, D_diag, D_udiag
        return Y, D, c

    def __visc_turb_FV(self, SL: SurfaceLayerData,
            turbulence: str="TKE", phi: array=None,
            old_phi: array=None, K_full: array=None,
            tke=None,
            l_m: array=None, l_eps: array=None,
            dz_theta: array=None, Ktheta_full: array=None,
            universal_funcs: tuple=None,
            ignore_tke_sl: bool=False,
            tau_b=None, wave_breaking: bool=False):
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
        u_star, delta_sl, inv_L_MO = SL.u_star, SL.delta_sl, SL.inv_L_MO
        if turbulence == "KPP":
            c_1 = 0.2 # constant for h_cl in atmosphere
            assert abs(self.f) > 1e-10
            h_cl: float = np.abs(c_1*u_star/self.f)
            G_full: array = self.kappa * np.abs(self.z_full) * \
                    (1 - np.abs(self.z_full)/h_cl)**2 \
                    * np.heaviside(1 - np.abs(self.z_full)/h_cl, 1)
            K_full = u_star * G_full + self.K_mol # viscosity
            K_full[self.M] = u_star * self.kappa * (-delta_sl+SL.z_0M)
        elif turbulence == "TKE":
            k = SL.k
            z_levels = np.copy(self.z_full)
            if not ignore_tke_sl:
                z_levels[k] = delta_sl
            h_half = np.diff(z_levels)
            phi_prime = phi[1:] * ( \
                    17./48 + K_full[1:]*self.dt/2/h_half**2 \
                    ) + phi[:-1] * ( \
                    7./48 - K_full[:-1]*self.dt/2/h_half**2 \
                    ) + np.diff(old_phi) / 48
            phi_second = phi[:-1] * ( \
                    17./48 + K_full[:-1]*self.dt/2/h_half**2 \
                    ) + phi[1:] * ( \
                    7./48 - K_full[1:]*self.dt/2/h_half**2 \
                    ) - np.diff(old_phi) / 48

            # For energy conservation: shear from Burchard (2002)
            shear_half = np.abs(K_full[1:] * phi[1:] * phi_prime \
                        + K_full[:-1] * phi[:-1] * phi_second)

            # linear equation of state:
            # rho1 = rho0 * (1 - alpha * theta_full - T0)
            # dzrho1 = rho0 * - alpha * dz_theta
            g = 9.81
            dz_rho = - self.rho0 * self.alpha * dz_theta
            N2_full = -g/self.rho0 * dz_rho
            shear_full = np.abs(K_full*phi*phi)

            tke.integrate_tke(self, SL, universal_funcs,
                    shear_full, K_full, l_eps, Ktheta_full,
                    N2_full, self.rho0*SL.u_star**2, tau_b)

            # MOST profiles cause troubles with FV free (high res):
            # dzu2_full[:k+1] = SL.u_star**2 * \
            #         phi_m(z_levels[:k+1]*SL.inv_L_MO)**2 / \
            #         self.kappa**2 * (z_levels[:k+1] + SL.z_0M)**2
            # without this, We don't have equivalence between
            # FV2 and FV free because of dzu2_full[k].

            l_m[:], l_eps[:] = self.__mixing_lengths(tke.tke_full,
                    shear_full / K_full, N2_full,
                    z_levels, SL, universal_funcs, wave_breaking)

            apdlr = self.__stability_temperature_phi_z(\
                    N2_full, K_full, shear_full)

            phi_m, phi_h, *_ = large_ocean() # for MOST:
            apdlr[SL.k:] = phi_m(-z_levels[SL.k:]*SL.inv_L_MO) / \
                    phi_h(-z_levels[SL.k:]*SL.inv_L_MO)

            Ktheta_full: array = np.maximum(self.Ktheta_min,
                    self.C_m * apdlr * l_m * np.sqrt(tke.tke_full))
            K_full = self.C_m * l_m * np.sqrt(tke.tke_full)
            # NOW we want to compare K_full, Ktheta_full
            # to their MOST equivalent.
            if SL.sf_scheme == "FV free":
                # remark that we don't need replacement,
                # we adjusted l_m, l_eps to ensure this.
                K_full_replacement = (self.kappa * \
                        SL.u_star*(-SL.delta_sl \
                       + SL.z_0M)) / \
                       phi_m(-SL.delta_sl*SL.inv_L_MO)
                assert abs(K_full_replacement - K_full[SL.k])<1e-10
                Ktheta_full_replacement = np.maximum((self.kappa * \
                        SL.u_star*(-SL.delta_sl \
                       + SL.z_0M)) / \
                       phi_h(-SL.delta_sl*SL.inv_L_MO),
                       self.Ktheta_min)
                assert abs(Ktheta_full_replacement - Ktheta_full[SL.k])<1e-10
        else:
            raise NotImplementedError("Wrong turbulence scheme")

        return K_full, Ktheta_full

    def __matrices_u_FV(self, K_full, forcing):
        """
            Creates the matrices D, Y, c such that the
            semi-discrete in space Ekman Stratified equation writes
            (d/dt Y - D) phi = c
        """
        # last two lines of diag, ldiag will be replaced by sf_scheme
        # last line of udiag will be replaced as well
        Y_diag: array = np.concatenate(([0.], 2/3*np.ones(self.M+1)))
        Y_udiag: array = np.concatenate(([0.],
            [self.h_half[m]/6./self.h_full[m]
                for m in range(1, self.M+1)]))
        Y_ldiag: array =  np.concatenate((\
            [self.h_half[m-1]/6./self.h_full[m]
                for m in range(1, self.M+1)], [0.]))

        D_diag: array = np.concatenate(([-K_full[0]], #Kdzu=r_D|u|
            [-2 * K_full[m] / self.h_half[m] / self.h_half[m-1]
                                for m in range(1, self.M+1)], [0.]))
        D_udiag: array = np.concatenate(([0.],
            [K_full[m+1]/self.h_full[m] / self.h_half[m]
                                for m in range(1, self.M)], [0.]))
        D_lldiag: array = np.zeros(self.M)

        D_ldiag: array = np.concatenate((\
            [K_full[m-1]/self.h_full[m] / self.h_half[m-1]
                                for m in range(1, self.M+1)], [0.]))


        c_B = 0. # r_D * |u| for bottom drag
        c = np.concatenate(([c_B], np.diff(forcing) / \
                                    self.h_full[1:-1], [0., 0.]))
        Y = (Y_ldiag, Y_diag, Y_udiag)
        D = (D_lldiag, D_ldiag, D_diag, D_udiag)
        return Y, D, c


    def __matrices_theta_FV(self, K_full: array, forcing: array):
        """
            Creates the matrices D, Y, c such that the
            semi-discrete in space Ekman Stratified equation writes
            (d/dt Y - D) phi = c
        """
        Y_diag: array = np.concatenate(([0.], 2/3*np.ones(self.M+1)))
        Y_udiag: array = np.concatenate(([0.],
            [self.h_half[m]/6./self.h_full[m]
                for m in range(1, self.M+1)]))
        Y_ldiag: array =  np.concatenate((\
            [self.h_half[m-1]/6./self.h_full[m]
                for m in range(1, self.M+1)], [0.]))

        D_diag: array = np.concatenate(([-K_full[0]],
            [-2 * K_full[m] / self.h_half[m] / self.h_half[m-1]
                                for m in range(1, self.M+1)], [0.]))
        D_udiag: array = np.concatenate(([0.],
            [K_full[m+1]/self.h_full[m] / self.h_half[m]
                                for m in range(1, self.M)], [0.]))
        D_lldiag: array = np.zeros(self.M)

        D_ldiag: array = np.concatenate((\
            [K_full[m-1]/self.h_full[m] / self.h_half[m-1]
                                for m in range(1, self.M+1)], [0.]))


        c_B = 0. # Gamma_T
        c = np.concatenate(([c_B], np.diff(forcing) / \
                                        self.h_full[1:-1], [0., 0.]))
        Y = (Y_ldiag, Y_diag, Y_udiag)
        D = (D_lldiag, D_ldiag, D_diag, D_udiag)
        return Y, D, c

    def __apply_sf_scheme(self, func, Y, D, c, Y_nm1=None, **kwargs):
        """
            Changes matrices Y, D, c on the first levels
            to use the surface flux scheme.
            _, func = self.dictsf_scheme[sf_scheme]
        """
        Y_sf, D_sf, c_sf, Y_nm1_sf = func(**kwargs)
        for y, y_sf in zip(Y, Y_sf): #strict=True, from python3.10
            y_sf = np.array(y_sf)
            if y_sf.shape[0] > 0:
                y[-y_sf.shape[0]:] = y_sf

        if Y_nm1 is not None: # Y_nm1 may be different from Y
            for y, y_sf in zip(Y_nm1, Y_nm1_sf):
                y_sf = np.array(y_sf)
                if y_sf.shape[0] > 0:
                    y[-y_sf.shape[0]:] = y_sf
        else: # otherwise we check the sf gives indeed the same Y
            assert Y_nm1_sf is Y_sf

        for d, d_sf in zip(D, D_sf):
            d_sf = np.array(d_sf)
            if d_sf.shape[0] > 0:
                d[-d_sf.shape[0]:] = d_sf

        c_sf = np.array(c_sf)
        c[-c_sf.shape[0]:] = c_sf

    def initialization(self, u_0, t_0, delta_sl_o, wind10m, t10m,
            u_star, t_star, Q_sw, Q_lw, delta_sl_a=10.,
            sf_scheme="FV free"):
        """
            initialize for FV free scheme. If this is not used,
            the continuity of the reconstruction cannot be
            guaranteed.
        """
        k = bisect.bisect_left(self.z_full, delta_sl_o)
        k_const = k-1 # change it if high-resolution
        # For high-res simulation, putting a quadratic profile
        #  between MOST and the constant profile :
        def func_z(z):
            z_const = self.z_full[k_const]
            return 1-((z - z_const) / (delta_sl_o - z_const))**2
        tilde_h = delta_sl_o - self.z_full[k-1]
        u_const, t_const = u_0[k_const], t_0[k_const]
        u_km1, t_km1 = u_0[k-1], t_0[k-1]
        phi, dz_theta = np.zeros(self.M+1) + 0j, np.zeros(self.M+1)
        phi_m, phi_h, psi_m, psi_h, *_ = large_ocean()
        SL = process_friction_scales_oce(wind10m, delta_sl_a,
                t10m, businger(), u_star, t_star,
                u_const, delta_sl_o, t_const,
                large_ocean(), sf_scheme, Q_sw, Q_lw, k, False)
        # no loop from here because of prescribed u_star
        zeta = -SL.delta_sl*SL.inv_L_MO
        phi[k] = SL.u_star / self.kappa / \
                (SL.z_0M-SL.delta_sl) * phi_m(zeta)
        t_star_rad = SL.t_star + SL.Q_lw / SL.u_star \
                / self.rho0 / self.C_p
        dz_theta[k] = t_star_rad / self.kappa / \
                (SL.z_0H-SL.delta_sl) * phi_h(zeta)
        u_tilde = u_km1 + tilde_h / 6 * (phi[k] + 2*phi[k-1])
        t_tilde = t_km1 + tilde_h / 6 * (dz_theta[k] + \
                2*dz_theta[k-1])
        u_delta = u_tilde + tilde_h / 6 * (2*phi[k] + phi[k-1])
        t_delta = t_tilde + tilde_h / 6 * (2*dz_theta[k] + \
                dz_theta[k-1])
        SL = process_friction_scales_oce(wind10m, delta_sl_a,
            t10m, businger(), u_star, t_star, u_delta,
            delta_sl_o, t_delta,
            large_ocean(), sf_scheme, SL.Q_sw, SL.Q_lw, k, False)

        # profiles under MOST : going smoothly to {u,t}_const
        u_km1 = u_delta + (u_const - u_delta) * \
                func_z(self.z_full[k-1])
        t_km1 = t_delta + (t_const - t_delta) * \
                func_z(self.z_full[k-1])
        u_0[k_const:k] = u_delta + (u_const-u_delta) *\
                func_z(self.z_half[k_const:k])
        t_0[k_const:k] = t_delta + (t_const-t_delta) *\
                func_z(self.z_half[k_const:k])
        def compute_dz(top_cond, var, h_half):
            """ solving the system of finite volumes:
            dz_{m-1}/6 + 2 dz_m / 3 + dz_{m+1} / 6 =
                    (var_{m+1/2} - var_{m-1/2})/h
            """
            ldiag = h_half[:-1] /6.
            diag = (h_half[1:] + h_half[:-1]) * 1/3.
            udiag = h_half[1:] /6.
            diag = np.concatenate(([1.], diag, [1.]))
            udiag = np.concatenate(([0.], udiag))
            ldiag = np.concatenate((ldiag, [0.]))
            rhs = np.concatenate(([0.],
                np.diff(var), [top_cond]))
            return solve_linear((ldiag, diag, udiag), rhs)
        phi[:k+1] = compute_dz(phi[k],
                np.concatenate((u_0[:k-1], [u_tilde])),
                np.concatenate((self.h_half[:k-1], [tilde_h])))
        dz_theta[:k+1] = compute_dz(dz_theta[k],
                np.concatenate((t_0[:k-1], [t_tilde])),
                np.concatenate((self.h_half[:k-1], [tilde_h])))
        # no endloop here because of prescribed u_star

        tau_u, tau_t = self.__tau_sl(SL, large_ocean())
        alpha_u = tilde_h / self.h_half[k-1] + tau_u
        alpha_t = tilde_h / self.h_half[k-1] + tau_t
        u_0[k-1] = alpha_u * u_tilde + tau_u * tilde_h * \
                (phi[k]/3 + phi[k-1]/6) + (1-alpha_u)*SL.u_0
        t_0[k-1] = alpha_t * t_tilde + tau_t * tilde_h * \
                (dz_theta[k]/3 + dz_theta[k-1]/6) + \
                (1-alpha_t)*SL.t_0
        return u_0, phi, t_0, dz_theta, u_delta, t_delta

    def __tau_sl(self, SL: SurfaceLayerData,
            universal_funcs) -> (float, float):
        if SL in self.dict_tau_sl:
            return self.dict_tau_sl[SL]
        delta_sl, inv_L_MO = SL.delta_sl, SL.inv_L_MO
        Q_sw, Q_lw = SL.Q_sw, SL.Q_lw
        _, phi_h, psi_m, psi_h, Psi_m, Psi_h = universal_funcs
        assert self.z_full[SL.k-1] < delta_sl
        assert delta_sl <= self.z_full[SL.k]
        zk = self.z_full[SL.k]
        def brackets_u(z):
            return (-z+SL.z_0M)*np.log(1-z/SL.z_0M) + z - \
                    z*Psi_m(-z*inv_L_MO)
        def brackets_theta(z):
            return (-z+SL.z_0H)*np.log(1-z/SL.z_0H) + z - \
                    z*Psi_h(-z*inv_L_MO)

        QH = SL.t_star * SL.u_star * self.rho0 * self.C_p
        term_lw = QH + SL.Q_lw

        # numerical integration of Qws_E:
        integral_Qsw_E = integrate.quad(Qsw_E, delta_sl,
                zk - (zk-delta_sl)*1e-5, args=(SL,),
                limit=20, epsrel=1e-4)[0]
        numer_theta = term_lw * (brackets_theta(zk) - \
                brackets_theta(delta_sl)) - integral_Qsw_E

        denom_u = np.log(1-delta_sl/SL.z_0M) \
                - psi_m(-delta_sl*inv_L_MO)
        denom_theta = term_lw * (np.log(1-delta_sl/SL.z_0H)\
                - psi_h(-delta_sl*inv_L_MO)) - Qsw_E(delta_sl, SL)


        if abs(denom_u) < 1e-10:
            tau_slu = 0.
        else:
            tau_slu = (brackets_u(zk) - brackets_u(delta_sl)) / \
                    self.h_half[SL.k-1] / denom_u
        if abs(denom_theta) < 1e-10:
            tau_slt = 0.
        else:
            tau_slt = numer_theta / denom_theta / self.h_half[SL.k-1]

        self.dict_tau_sl[SL] = tau_slu, tau_slt
        return tau_slu, tau_slt


    ####### DEFINITION OF SF SCHEMES : VALUE OF u(delta_sl) #####
    # The method must use the prognostic variables and delta_sl
    # to return u(delta_sl).
    # the prognostic variables are u for FD and 
    # (u_{1/2}, ... u_{k+1/2}, phi_k, ...phi_M) for FV.

    def __sf_udelta_FDpure(self, prognostic, **_):
        """
            u(delta) = u_{-1/2}
        """
        return prognostic[-1]

    def __sf_udelta_FD2(self, prognostic, **_):
        """
            u(delta) = (u_{-1/2} + u_{-3/2}) / 2
        """
        return (prognostic[-1] + prognostic[-2])/2

    def __sf_udelta_FDtest(self, prognostic, **_):
        """
            u(delta) = u_{-1/2}
        """
        return prognostic[-1]

    def __sf_udelta_FVpure(self, prognostic, **_):
        """
            u(delta) = Bar{u}_{-1/2} - h (phi_0 - phi_{-1})/24
        """
        return prognostic[-1] - self.h_half[self.M-1]* \
                (prognostic[-2] - prognostic[-3])/24

    def __sf_udelta_FV1(self, prognostic, **_):
        """
            u(delta) = Bar{u}_{-1/2}
        """
        return prognostic[-1]

    def __sf_udelta_FVtest(self, prognostic, **_): # Dirichlet cond
        """
            u(delta) = Bar{u}_{-1/2} + h phi_0 / 3 + h phi_{-1} / 6
        """
        return prognostic[-1] + self.h_half[self.M-1]* \
                (prognostic[-2]/3 + prognostic[-3]/6)

    def __sf_udelta_FVfree(self, prognostic, SL,
            universal_funcs, **_):
        """
            u(delta) = 1/alpha ( Bar{u}_{k-1/2} - u(0) * (1 - alpha)
                    + ~h^2/h (phi_k / 3 + phi_{k-1} / 6))
        """
        tau_slu, _ = self.__tau_sl(SL, universal_funcs)
        tilde_h = SL.delta_sl - self.z_full[SL.k-1]
        alpha = tilde_h/self.h_half[SL.k-1] + tau_slu
        return (prognostic[SL.k + 1] + \
                tilde_h * tilde_h / self.h_half[SL.k-1] * \
                (prognostic[SL.k] / 3 + prognostic[SL.k-1] / 6)\
                - (1 - alpha) * SL.u_0) / alpha

    ####### DEFINITION OF SF SCHEMES : FIRST LINES OF Y,D,c #####
    # each method must return Y, D, c:
    # Y: 3-tuple of tuples of sizes (j-1, j, j)
    # D: 3-tuple of tuples of sizes (j-1, j, j)
    # c: tuple of size j
    # they represent the first j lines of the matrices.
    # for Y and D, the tuples are (lower diag, diag, upper diag)

    def __sf_YDc_FDpure(self, K_u, forcing, SL, **_):
        """
            Y = (0    ,             1                         )
            D = (K/h^2, - K/h^2 - u*^2 / (h|u(z_a) - u(z_o)|) )
            c = (F + u*^2 u(z_a) / (h|u(z_a) - u(z_o)|)       )
        """
        wind_atm = SL.SL_other.u_delta
        u_star, u_delta = SL.u_star, SL.u_delta
        norm_jump = np.abs(wind_atm - u_delta)
        Y = ((0.,), (1.,), ())
        D = ((K_u[self.M-1] / self.h_full[self.M-1] / \
                    self.h_half[self.M-1],),
            (- u_star**2 / norm_jump / self.h_half[self.M-1] - \
            K_u[self.M-1] / self.h_full[self.M-1] / \
                    self.h_half[self.M-1],), ())
        c = (forcing[self.M - 1] + u_star**2*wind_atm / norm_jump / \
                    self.h_half[self.M-1],)
        return Y, D, c, Y

    def __sf_YDc_FD2(self, K_u, SL, **_):
        """
            Y = (            0           ,             0           )
            D = ( K/h - u*^2/(2|ua - uo|),-K/h - u*^2/(2|ua - uo|) )
            c = (    u*^2 ua / |ua - uo|    )
            where ua, uo = u(z_a), u(z_o).
        """
        wind_atm = SL.SL_other.u_delta
        u_star, u_delta = SL.u_star, SL.u_delta
        Y = ((0.,), (0.,), ())
        norm_jump =  np.abs(wind_atm - u_delta)
        D = ((K_u[self.M-1]/self.h_full[self.M-1] -\
                u_star**2 / norm_jump / 2,),
            (-K_u[self.M-1]/self.h_full[self.M-1] -\
                u_star**2 / norm_jump / 2,), ())
        c = (u_star**2 * wind_atm / norm_jump,)
        return Y, D, c, Y

    def __sf_YDc_FDtest(self, K_u, forcing, SL, **_):
        """
            Y = (0     ,    1     )
            D = (K/h^2 , -K/h^2   )
            c = (  F + u*^2/h   )
        """
        Y = ((0.,), (1.,), ())
        D = ((K_u[self.M-1] / self.h_full[self.M-1] \
                / self.h_half[self.M-1],),
            (- K_u[self.M-1] / self.h_full[self.M-1] \
                / self.h_half[self.M-1],), ())
        c = (forcing[self.M - 1] + \
                SL.u_star**2/self.h_half[self.M-1] ,)
        return Y, D, c, Y


    def __sf_YDc_FVpure(self, K_u, forcing, SL, **_):
        """
            Y = (0     ,    0     , 1 )
                (0     ,    0     , 0 )
            D = ( -K/h  ,       K/h             ,   0)
                (- h/24 , h/24 - K|ua-uo|/u*^2  ,  -1)
            c = (  F  )
                (  ua )
        """
        wind_atm = SL.SL_other.u_delta
        u_star, u_delta = SL.u_star, SL.u_delta
        norm_jump = np.abs(wind_atm - u_delta)
        Y = ((0., 0.), (0., 0.), (1.,))
        D = ((-self.h_half[self.M-1]/24,),
            (-K_u[self.M-1]/self.h_half[self.M-1],
            -K_u[self.M]*norm_jump/u_star**2 \
                    + self.h_half[self.M-1]/24),
            (K_u[self.M] / self.h_half[self.M-1], -1.), (0.,))
        c = (forcing[self.M-1], wind_atm)
        return Y, D, c, Y

    def __sf_YDc_FV1(self, K_u, forcing, SL, **_):
        """
            Y = (0     ,    0     , 1 )
                (0     ,    0     , 0 )
            D = (-K/h ,       K/h        ,   0)
                (  0  , - K|ua-uo|/u*^2  ,  -1)
            c = (  F  )
                (  ua )
        """
        wind_atm = SL.SL_other.u_delta
        u_star, u_delta = SL.u_star, SL.u_delta
        norm_jump = np.abs(wind_atm - u_delta)
        Y = ((0., 0.), (0., 0.), (1.,))
        D = ((0.,),
            (-K_u[self.M-1]/self.h_half[self.M-1],
            -K_u[self.M]*norm_jump/u_star**2),
            (K_u[self.M] / self.h_half[self.M-1], -1.), (0.,))
        c = (forcing[self.M-1], wind_atm)
        return Y, D, c, Y

    def __sf_YDc_FVtest(self, K_u, forcing, SL, **_):
        """
            Y = (0     ,    0     , 1 )
                (0     ,    0     , 0 )
            D = (-K/h , K/h  ,  0)
                (  0  , - K  ,  0)
            c = (   F   )
                (  u*^2 )
        """
        Y = ((0., 0.), (0., 0.), (1.,))
        # dont forget equation is dtYu - Du = c
        D = ((0.,),
            (-K_u[self.M-1]/self.h_half[self.M-1], -K_u[self.M]),
            (K_u[self.M] / self.h_half[self.M-1], 0.),)
        c = (forcing[self.M-1], SL.u_star**2)
        return Y, D, c, Y

    def __sf_YDc_FVfree(self, K_u, forcing, universal_funcs,
            SL, SL_nm1, **_):
        """
            Y  = (h/(6h),  (~h+h)/(3h) , ~h/(6h)       , 0   )
                 ( 0    , -tau ~h/(6a) , -tau ~h/(3a)  , 1/a )
                 ( 0    ,           0  ,    0,         , 0   )

            D = (K/h^2,-K(1/h+1/~h)/h,         K/(h~h)         , 0 )
                ( 0   ,     -K/~h    ,         K/~h            , 0 )
                ( 0   ,    ~h^2/(6h) , ~h^2/(3h)+Ka|u0-uo|/u*^2, 1 )

            c = (        1/h (forcing_{k-1/2} - forcing_{k-3/2})     )
                ( forcing_{k-1/2} + (partial_t + if) ( u(0) (1-a)/a ))
                (                     -u(0)                          )
            after that, the matrices are filled for every M > m > k
            with 0 for Y
            D: (ldiag, diag, udiag) = (ratio_norms, -1, 0)
            where ratio_norms is given by the log law.
            this last part is actually overriden in the end so
            it should not be considered too seriously.
        """
        u_star, u_delta, delta_sl, inv_L_MO, k = SL.u_star, \
                SL.u_delta, SL.delta_sl, SL.inv_L_MO, SL.k
        tilde_h = delta_sl - self.z_full[k-1]
        tau, _ = self.__tau_sl(SL, universal_funcs)
        alpha = tilde_h/self.h_half[k-1] + tau
        tau_nm1, _ = self.__tau_sl(SL_nm1, universal_funcs)
        alpha_nm1 = tilde_h/self.h_half[k-1] + tau_nm1
        # note that delta_sl is assumed constant here.
        # its variation would need much more changes.
        Y = ( (self.h_half[k-2]/6./self.h_full[k-1],
            - tilde_h * tau / 6. / alpha, 0.),# LOWER DIAG
            ((tilde_h + self.h_half[k-2])/3/self.h_full[k-1],
            - tilde_h * tau / 3. / alpha, 0.),# DIAG
            (tilde_h/6./self.h_full[k-1], 1/alpha))# UPPER DIAG
        Y_nm1 = ( (self.h_half[k-2]/6./self.h_full[k-1],
            - tilde_h * tau_nm1 / 6. / alpha_nm1, 0.),# LOWER DIAG
            ((tilde_h + self.h_half[k-2])/3/self.h_full[k-1],
            - tilde_h * tau_nm1 / 3. / alpha_nm1, 0.),# DIAG
            (tilde_h/6./self.h_full[k-1], 1/alpha_nm1))# UPPER DIAG
        D = ((0., tilde_h**2 / 6 / self.h_half[k-1]),#LLOWER DIAG
                (K_u[k-2]/self.h_half[k-2]/self.h_full[k-1],
                    -K_u[k-1]/tilde_h,
                    tilde_h**2 / 3 / self.h_half[k-1] + alpha * \
                        np.abs(SL.u_0-SL.u_delta)/SL.u_star**2 * \
                        K_u[k]),#LOWER DIAG
                (-K_u[k-1] / self.h_full[k-1] * \
                        (1/self.h_half[k-2] + 1/tilde_h),
                        K_u[k] / tilde_h, 1.),#DIAG
                (K_u[k] / tilde_h / self.h_full[k-1], 0.))#UPPER DIAG

        rhs_n = SL.u_0 * (1 - alpha)/alpha
        rhs_nm1 = SL_nm1.u_0 * (1 - alpha_nm1)/alpha_nm1
        rhs_part_tilde = (rhs_n - rhs_nm1)/self.dt + 1j*self.f * \
                (self.implicit_coriolis*rhs_n + \
                (1 - self.implicit_coriolis) * rhs_nm1)
        c = ( (forcing[k-1] - forcing[k-2])/self.h_full[k-1],
                forcing[k-1] + rhs_part_tilde, -SL.u_0)

        Y = (np.concatenate((y, np.zeros(self.M - k))) for y in Y)
        Y_nm1 = (np.concatenate((y, np.zeros(self.M - k))) for y in Y_nm1)

        *_, Psi_m, _ = universal_funcs
        def f(z):
            return (-z+SL.z_0M)*np.log(1-z/SL.z_0M) + z - \
                    z * Psi_m(-z*inv_L_MO)
        try:
            # for m >= k: ratio between |u-u0| at m+1/2 and m-1/2.
            ratio_norms = np.array([(f(self.z_full[m+1]) - \
                    f(self.z_full[m])) / \
                    (f(self.z_full[m]) - f(self.z_full[m-1])) \
                        for m in range(k, self.M)])
        except ZeroDivisionError:
            ratio_norms = np.zeros(self.M - k)

        D = (np.concatenate((D[0], np.zeros(self.M - k))),
                np.concatenate((D[1], ratio_norms)),
                np.concatenate((D[2], - np.ones(self.M - k))),#DIAG
                np.concatenate((D[3], np.zeros(self.M - k))))
        c = np.concatenate((c, SL.u_0 * \
                (self.h_half[k:-1] - ratio_norms)))
        return Y, D, c, Y_nm1


    ####### DEFINITION OF SF SCHEMES : VALUE OF theta(delta_sl) ##
    # The method must use the prognostic variables and delta_sl
    # to return theta(delta_sl).
    # the prognostic variables are theta for FD and
    # (theta_{1/2}, ... theta_{k+1/2}, phit_k, ...phit_M) for FV.
    def __sf_thetadelta_FDpure(self, prognostic, **_):
        """
            t(delta) = t_{-1/2}
        """
        return prognostic[self.M-1]

    def __sf_thetadelta_FD2(self, prognostic, **_):
        """
            t(delta) = (t_{-1/2} + t_{-3/2})/2
        """
        return (prognostic[self.M-1] + prognostic[self.M-2])/2

    def __sf_thetadelta_FDtest(self, prognostic, **_):
        """
            t(delta) = t_{-1/2}
        """
        return prognostic[self.M-1]

    def __sf_thetadelta_FVpure(self, prognostic, **_):
        """
            t(delta) = Bar{t}_{-1/2} - h (dzt_0 - dzt_{-1})/24
        """
        return prognostic[-1] - self.h_half[self.M-1]* \
                (prognostic[self.M] - prognostic[self.M-1])/24

    def __sf_thetadelta_FV1(self, prognostic, **_):
        """
            t(delta) = Bar{t}_{-1/2}
        """
        return prognostic[-1]

    def __sf_thetadelta_FVtest(self, prognostic, **_):
        """
            t(delta) = Bar{t}_{-1/2} + h dzt_0 / 3 + h dzt_{-1} / 6
        """
        return prognostic[-1] + self.h_half[self.M-1]* \
                (prognostic[-2]/3 + prognostic[-3]/6)

    def __sf_thetadelta_FVfree(self, prognostic, SL,
            universal_funcs, **_):
        """
            t(delta) = Bar{t}_{k-1/2} - t(0) * (1 - alpha) / alpha
                    + ~h^2/h (dzt_k / 3 + dzt_{k-1} / 6)
        """
        _, tau_slt = self.__tau_sl(SL, universal_funcs)
        tilde_h = SL.delta_sl - self.z_full[SL.k-1]
        alpha = tilde_h/self.h_half[SL.k-1] + tau_slt
        return (prognostic[SL.k + 1] + \
                tilde_h * tilde_h / self.h_half[SL.k-1] * \
                (prognostic[SL.k] / 3 + prognostic[SL.k-1] / 6)\
                - (1 - alpha) * SL.t_0)  / alpha

    ####### DEFINITION OF SF SCHEMES : FIRST LINES OF Y,D,c (theta)
    # each method must return Y_n, D, c, Y_nm1:
    # Y_*: 3-tuple of tuples of sizes (j-1, j, j)
    # D: 3-tuple of tuples of sizes (j-1, j, j)
    # c: tuple of size j
    # they represent the first j lines of the matrices.
    # for Y and D, the tuples are (lower diag, diag, upper diag)
    # Now that Y_n and Y_nm1 can be different,
    # the framework with Y, D and c is far less convenient.
    # I'll change this in a future update.

    def __sf_YDc_FDpure_theta(self, K_theta, SL, forcing_theta,
            **kwargs):
        """
            Y = (0     ,    1     )
            D = (K/h^2 , -K/h^2   )
            c = (  F + (QH + Q_sw + Q_lw) / (h rho cp)   )
        """
        QH = self.rho0 * self.C_p * SL.t_star*SL.u_star
        Y = ((0.,), (1.,), ())
        D = ((K_theta[self.M-1] / self.h_full[self.M-1] \
                / self.h_half[self.M-1],),
            (- K_theta[self.M-1] / self.h_full[self.M-1] \
                / self.h_half[self.M-1],), ())
        c = (forcing_theta[self.M - 1] + \
            (QH + SL.Q_sw + SL.Q_lw) / self.rho0 / self.C_p / \
            self.h_half[self.M-1] ,)
        return Y, D, c, Y

    def __sf_YDc_FD2_theta(self, K_theta, SL, **kwargs):
        """
            Y = (  0   ,   0  )
            D = ( K/h  , -K/h )
            c = ( (QH + Q_sw(delta_sl) + Q_lw) / (rho cp) )
        """
        Y = ((0.,), (0.,), ())
        D = ((K_theta[self.M-1]/self.h_full[self.M-1],),
                (-K_theta[self.M-1]/self.h_full[self.M-1],),
                ())
        QH = self.rho0 * self.C_p * SL.t_star*SL.u_star
        Q_sw = SL.Q_sw * shortwave_frac_sl(SL.delta_sl)
        c = ((QH + Q_sw + SL.Q_lw) / self.rho0 / self.C_p,)
        return Y, D, c, Y

    def __sf_YDc_FVpure_theta(self, K_theta, SL, forcing_theta,
            **kwargs):
        assert NotImplementedError("Please do not use FV pure" + \
                " surface flux scheme in ocean")

    def __sf_YDc_FV1_theta(self, K_theta, SL, forcing_theta,
            **kwargs):
        assert NotImplementedError("Please do not use FV1" + \
                " surface flux scheme in ocean")

    def __sf_YDc_FDtest_theta(self, K_theta, SL,
            forcing_theta, **_):
        """
            Y = (0     ,    1     )
            D = (K/h^2 , -K/h^2   )
            c = (  F + (QH + Q_sw + Q_lw) / (h rho cp)   )
        """
        QH = self.rho0 * self.C_p * SL.t_star*SL.u_star
        Y = ((0.,), (1.,), ())
        D = ((K_theta[self.M-1] / self.h_full[self.M-1] \
                / self.h_half[self.M-1],),
            (- K_theta[self.M-1] / self.h_full[self.M-1] \
                / self.h_half[self.M-1],), ())
        c = (forcing_theta[self.M - 1] + \
            (QH + SL.Q_sw + SL.Q_lw) / self.rho0 / self.C_p / \
            self.h_half[self.M-1] ,)
        return Y, D, c, Y

    def __sf_YDc_FVtest_theta(self, K_theta,
            SL, universal_funcs,
            forcing_theta, **_):
        """
            Y = (0     ,    0     , 1 )
                (0     ,    0     , 0 )
            D = (-K/h , K/h  ,  0)
                (  0  , - K  ,  0)
            c = (               F                 )
                (   (QH + Q_sw + Q_lw)/ (rho cp)  )
        """
        QH = self.rho0 * self.C_p * SL.t_star*SL.u_star
        Y = ((0., 0.), (0., 0.), (1.,))
        D = ((0.,),
            (-K_theta[self.M-1]/self.h_half[self.M-1],
                -K_theta[self.M]), #KdzT = (QH-Qs)/(rho Cp)
                (K_theta[self.M] / self.h_half[self.M-1], 0))
        # QH: total heat
        # QS: solar part
        c = (forcing_theta[self.M-1], (QH + SL.Q_sw + \
                SL.Q_lw)/(self.rho0*self.C_p))
        return Y, D, c, Y

    def __sf_YDc_FVfree_theta(self, K_theta, SL, SL_nm1,
            forcing_theta, universal_funcs, **kwargs):
        """
            Y  = (h/(6h),  (~h+h)/(3h) , ~h/(6h)       , 0   )
                 ( 0    , -tau ~h/(6a) , -tau ~h/(3a)  , 1/a )
                 ( 0    ,           0  ,    0,         , 0   )

            D = (K/h^2,-K(1/h+1/~h)/h,         K/(h~h)          , 0)
                ( 0   ,     -K/~h    ,         K/~h             , 0)
                ( 0   ,       0      ,         -K               , 0)

            c = (         1/h (forcing_{k-1/2} - forcing_{k-3/2})  )
                ( forcing_{k-1/2} + (partial_t + if) (u(0) (1-a)/a))
                (          (QH + Q_sw + Q_lw)/ (rho cp)            )
            after that, the matrices are filled for every M > m > k
            with 0 for Y
            D: (ldiag, diag, udiag) = (ratio_norms, -1, 0)
            where ratio_norms is given by the log law.
            this last part is actually overriden in the end so
            it should not be considered too seriously.
        """
        # Il faut changer ça pour utiliser seulement universal_funcs
        # Donc enlever CHdu du standalone_chapter à priori,
        # pour utiliser un truc plus compatible
        delta_sl, inv_L_MO, k = SL.delta_sl, SL.inv_L_MO, SL.k
        tilde_h = delta_sl - self.z_full[k-1]
        _, tau = self.__tau_sl(SL, universal_funcs)
        alpha = tilde_h/self.h_half[k-1] + tau
        _, tau_nm1 = self.__tau_sl(SL_nm1, universal_funcs)
        alpha_nm1 = tilde_h/self.h_half[k-1] + tau_nm1
        # note that delta_sl is assumed constant here.
        # its variation would need much more changes.
        Y = ( (self.h_half[k-2]/6./self.h_full[k-1],
            - tilde_h * tau / 6. / alpha, 0.),# LOWER DIAG
            ((tilde_h + self.h_half[k-2])/3/self.h_full[k-1],
            - tilde_h * tau / 3. / alpha, 0.),# DIAG
            (tilde_h/6./self.h_full[k-1], 1/alpha))# UPPER DIAG
        Y_nm1 = ( (self.h_half[k-2]/6./self.h_full[k-1],
            - tilde_h * tau_nm1 / 6. / alpha_nm1, 0.),# LOWER DIAG
            ((tilde_h + self.h_half[k-2])/3/self.h_full[k-1],
            - tilde_h * tau_nm1 / 3. / alpha_nm1, 0.),# DIAG
            (tilde_h/6./self.h_full[k-1], 1/alpha_nm1))# UPPER DIAG
        D = ((0., 0.),#LLOWER DIAG
                (K_theta[k-2]/self.h_half[k-2]/self.h_full[k-1],
                    -K_theta[k-1]/tilde_h, -K_theta[k]),#LOWER DIAG
                (-K_theta[k-1] / self.h_full[k-1] * \
                        (1/self.h_half[k-2] + 1/tilde_h),
                        K_theta[k] / tilde_h, 0.),#DIAG
                (K_theta[k] / tilde_h / self.h_full[k-1],
                    0.))#UPPER DIAG

        QH = self.rho0 * self.C_p * SL.t_star*SL.u_star
        Q_sw = SL.Q_sw * shortwave_frac_sl(SL.delta_sl)

        rhs_n = SL.t_0 * (1 - alpha)/alpha
        rhs_nm1 = SL_nm1.t_0 * (1 - alpha_nm1)/alpha_nm1
        rhs_part_tilde = (rhs_n - rhs_nm1)/self.dt
        c = ( (forcing_theta[k-1] - forcing_theta[k-2])/self.h_full[k-1],
                forcing_theta[k-1] + rhs_part_tilde,
                (QH + Q_sw + SL.Q_lw) / self.rho0 / self.C_p)

        Y = (np.concatenate((y, np.zeros(self.M - k))) for y in Y)
        Y_nm1 = (np.concatenate((y, np.zeros(self.M - k))) for y in Y_nm1)

        *_, Psi_m, _ = universal_funcs
        def f(z):
            return (-z+SL.z_0H)*np.log(1-z/SL.z_0H) + z - \
                    z * Psi_m(-z*inv_L_MO)
        try:
            # for m >= k: ratio between |u-u0| at m+1/2 and m-1/2.
            ratio_norms = np.array([(f(self.z_full[m+1]) - \
                    f(self.z_full[m])) / \
                    (f(self.z_full[m]) - f(self.z_full[m-1])) \
                        for m in range(k, self.M)])
        except ZeroDivisionError:
            ratio_norms = np.zeros(self.M - k)

        D = (np.concatenate((D[0], np.zeros(self.M - k))),
                np.concatenate((D[1], ratio_norms)),
                np.concatenate((D[2], - np.ones(self.M - k))),#DIAG
                np.concatenate((D[3], np.zeros(self.M - k))))
        c = np.concatenate((c, SL.t_0 * \
                (self.h_half[k:-1] - ratio_norms)))
        return Y, D, c, Y_nm1
