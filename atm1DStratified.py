"""
    This module defines the class Atm1dStratified
    which simulates a 1D Ekman stratified problem with various
    discretisations.
"""

from typing import Tuple, List, NamedTuple
import bisect
import numpy as np
from utils_linalg import multiply, scal_multiply as s_mult
from utils_linalg import add_banded as add
from utils_linalg import solve_linear
from utils_linalg import full_to_half
from bulk import SurfaceLayerData, friction_scales
from universal_functions import Businger_et_al_1971 as businger
from universal_functions import Large_et_al_2019 as large_ocean
def pr(var, name="atm"):
    print(var, name)

array = np.ndarray
# TKE coefficients:
coeff_FV_big = 1/3. # 1/3 or 5/12
coeff_FV_small = 1/6. # 1/6 or 1/12

class Atm1dStratified():
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
                "FV test" : (self.__sf_udelta_FVpure,
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
                "FV test" : (self.__sf_thetadelta_FVpure,
                                self.__sf_YDc_FVpure_theta),
                "FV1" : (self.__sf_thetadelta_FV1,
                                self.__sf_YDc_FV1_theta),
                "FV2" : (self.__sf_thetadelta_FV2,
                                self.__sf_YDc_FV2_theta),
                "FV free" : (self.__sf_thetadelta_FVfree,
                                self.__sf_YDc_FVfree_theta),
                }

    def FV(self, u_t0: array, phi_t0: array, theta_t0: array,
            dz_theta_t0: array, forcing: array, Q_sw: array,
            Q_lw: array, u_o: array, SST:array, delta_sl: float,
            delta_sl_o: float,
            sf_scheme: str="FV pure",
            u_delta: float=8.+0j, t_delta: float=265.,
            Neutral_case: bool=False, turbulence: str="TKE",
            store_all: bool=False):
        """
            Integrates in time with Backward Euler the model with TKE
            and Finite volumes.

            u_t0 : average of cells (centered on self.z_half[:-1])
            phi_t0 : derivative of u (given at self.z_full)
            forcing: averaged forcing for u on each volume for all times
            SST: Surface Temperature for all times
            u_o: Surface momentum for all times
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
        assert forcing.shape[1] == self.M
        assert t_delta > 0
        assert SST.shape[0] == forcing.shape[0]
        assert sf_scheme in self.dictsf_scheme_theta
        assert sf_scheme in self.dictsf_scheme
        if sf_scheme in {"FV2",}:
            assert abs(delta_sl - self.z_full[1]) < 1e-10
        elif sf_scheme in {"FV1", "FV pure"}:
            assert abs(delta_sl - self.z_full[1]/2) < 1e-10
        assert turbulence in {"TKE", "KPP"}
        N: int = forcing.shape[0] - 1 # number of time steps
        forcing_theta = np.zeros_like(forcing[0])

        k = bisect.bisect_right(self.z_full[1:], delta_sl)
        SL: SurfaceLayerData = friction_scales(u_delta,
                delta_sl, t_delta, businger(),
                u_o[0], delta_sl_o, SST[0], large_ocean(), sf_scheme,
                Q_sw[0], Q_lw[0],
                k, True)
        ignore_tke_sl = sf_scheme in {"FV pure", "FV1"}

        import tkeAtm1D
        tke = tkeAtm1D.TkeAtm1D(self, "FV",
                ignore_tke_sl, Neutral_case, SL)

        theta, dz_theta = np.copy(theta_t0), np.copy(dz_theta_t0)

        Ku_full: array = self.K_min + np.zeros(self.M+1)
        Ktheta_full: array = self.Ktheta_min + np.zeros(self.M+1)

        z_levels_sl = np.copy(self.z_full)
        z_levels_sl[k] = self.z_full[k] if ignore_tke_sl else delta_sl
        l_m, l_eps = self.__mixing_lengths( tke.tke_full,
                np.abs(phi_t0*phi_t0), 9.81*dz_theta/283.,
                z_levels_sl, SL, businger())

        phi, old_phi = phi_t0, np.copy(phi_t0)
        u_current: array = np.copy(u_t0)
        all_u_star, all_t_star = [], []
        ret_u_current, ret_tke, ret_dz_tke, ret_SL = [], [], [], []
        ret_tke_bar = []
        ret_u_delta, ret_t_delta = [u_delta], [t_delta]
        ret_phi, ret_theta, ret_dz_theta, ret_leps = [], [], [], []

        for n in range(1,N+1):
            # Compute friction scales
            SL_nm1, SL = SL, friction_scales(u_delta, delta_sl,
                    t_delta, businger(), u_o[n], delta_sl_o, SST[n],
                    large_ocean(), sf_scheme, Q_sw[n], Q_lw[n],
                    k, True)
            all_u_star += [SL.u_star]
            all_t_star += [SL.t_star]

            # Compute viscosities
            Ku_full, Ktheta_full = self.__visc_turb_FV(
                    SL, turbulence=turbulence, phi=phi,
                    old_phi=old_phi, l_m=l_m, l_eps=l_eps,
                    K_full=Ku_full, tke=tke,
                    dz_theta=dz_theta, Ktheta_full=Ktheta_full,
                    universal_funcs=businger(),
                    ignore_tke_sl=ignore_tke_sl)

            old_phi = phi
            # integrate in time momentum
            u_current, phi, u_delta = self.__step_u(u=u_current,
                    phi=phi, Ku_full=Ku_full,
                    forcing=forcing[n], SL=SL, SL_nm1=SL_nm1)

            if not Neutral_case:
                # integrate in time potential temperature
                theta, dz_theta, t_delta = self.__step_theta(theta,
                        dz_theta, Ktheta_full, forcing_theta,
                        SL, SL_nm1)

            ret_u_delta += [u_delta]
            ret_t_delta += [t_delta]
            if store_all:
                ret_u_current += [np.copy(u_current)]
                ret_tke += [np.copy(tke.tke_full)]
                ret_tke_bar += [np.copy(tke.tke)]
                ret_dz_tke += [np.copy(tke.dz_tke)]
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
            ret_dict['all_tke_bar'] = ret_tke_bar
            ret_dict['all_dz_tke'] = ret_dz_tke
            ret_dict['all_theta'] = ret_theta
            ret_dict['all_dz_theta'] = ret_dz_theta
            ret_dict['all_leps'] = ret_leps
            ret_dict['all_SL'] = ret_SL

        ret_dict['u'] = u_current
        ret_dict['phi'] = phi
        ret_dict['tke'] = tke.tke_full
        ret_dict['all_u_star'] = all_u_star
        ret_dict['all_t_star'] = all_t_star
        ret_dict['theta'] = theta
        ret_dict['dz_theta'] = dz_theta
        ret_dict['l_eps'] = l_eps
        ret_dict['SL'] = SL
        ret_dict['Ktheta'] = Ktheta_full
        return ret_dict

    def FD(self, u_t0: array, theta_t0: array, Q_sw: array,
            u_o: array, Q_lw: array, forcing: array, SST:array,
            delta_sl_o: float,
            turbulence: str="TKE", sf_scheme: str="FD pure",
            Neutral_case: bool=False, store_all: bool=False):
        """
            Integrates in time with Backward Euler the model with KPP
            and Finite differences.

            u_t0 should be given at half-levels (follow self.z_half)
            forcing should be given at half-levels for all times
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
        assert forcing.shape[1] == self.M
        assert SST.shape[0] == forcing.shape[0]
        assert sf_scheme in self.dictsf_scheme_theta
        assert sf_scheme in self.dictsf_scheme
        assert sf_scheme in {"FD2", "FD pure"}
        assert turbulence in {"TKE", "KPP"}
        N: int = forcing.shape[0] - 1 # number of time steps
        # methods to get u(delta) and theta(delta):
        func_un, _ = self.dictsf_scheme[sf_scheme]
        func_theta, _ = self.dictsf_scheme_theta[sf_scheme]
        # height of the surface layer:
        delta_sl = self.z_half[0] if sf_scheme == "FD pure" \
                else self.z_full[1]
        forcing_theta = np.zeros_like(forcing[0]) # no temp forcing
        ###### Initialization #####
        import tkeAtm1D
        tke = tkeAtm1D.TkeAtm1D(self, "FD", Neutral_case=Neutral_case)
        theta: array = np.copy(theta_t0)

        # Initializing viscosities and mixing lengths:
        Ku_full: array = self.K_min + np.zeros(self.M+1)
        Ktheta_full: array = self.Ktheta_min + np.zeros(self.M+1)
        l_m = self.lm_min*np.ones(self.M+1)
        l_eps = self.leps_min*np.ones(self.M+1)

        u_current: array = np.copy(u_t0)
        old_u: array = np.copy(u_current)
        all_u_star, all_t_star = [], []
        all_u, all_tke, all_theta, all_leps = [], [], [], []
        ret_u_delta, ret_t_delta = [], []
        for n in range(1,N+1):
            forcing_current: array = forcing[n]
            u_delta = func_un(prognostic=u_current, delta_sl=delta_sl)
            t_delta = func_theta(prognostic=theta)
            # Compute friction scales
            SL: SurfaceLayerData= friction_scales(u_delta, delta_sl,
                    t_delta, businger(), u_o[n], delta_sl_o, SST[n],
                    large_ocean(), sf_scheme, Q_sw[n], Q_lw[n],
                    0, True)
            all_u_star += [SL.u_star]
            all_t_star += [SL.t_star]

            # Compute viscosities
            Ku_full, Ktheta_full = self.__visc_turb_FD(SL=SL,
                    u_current=u_current, old_u=old_u,
                    K_full=Ku_full, tke=tke, theta=theta,
                    Ktheta_full=Ktheta_full, l_m=l_m, l_eps=l_eps,
                    universal_funcs=businger())

            # integrate in time momentum
            Y, D, c = self.__matrices_u_FD(Ku_full, forcing_current)
            self.__apply_sf_scheme(Y=Y, D=D, c=c, K_u=Ku_full,
                    func=self.dictsf_scheme[sf_scheme][1],
                    forcing=forcing_current, SL=SL)
            next_u = self.__backward_euler(Y=Y, D=D, c=c,
                    u=u_current, f=self.f)
            u_current, old_u = next_u, u_current

            if not Neutral_case:
                # integrate in time potential temperature
                Y_theta, D_theta, c_theta = self.__matrices_theta_FD(
                        Ktheta_full, np.zeros(self.M))
                self.__apply_sf_scheme(\
                        func=self.dictsf_scheme_theta[sf_scheme][1],
                        Y=Y_theta, D=D_theta, c=c_theta, SL=SL,
                        K_theta=Ktheta_full, forcing=forcing_theta,
                        universal_funcs=businger())

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
        ret_dict['all_t_star'] = all_t_star
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
        prognostic: array = np.concatenate((u[:SL.k+1], phi[SL.k:]))
        Y, D, c = self.__matrices_u_FV(Ku_full, forcing)
        Y_nm1 = tuple(np.copy(y) for y in Y)
        self.__apply_sf_scheme(\
                func=func_YDc, Y=Y, D=D, c=c, K_u=Ku_full,
                forcing=forcing, SL=SL, SL_nm1=SL_nm1, Y_nm1=Y_nm1,
                universal_funcs=businger())

        prognostic = self.__backward_euler(Y=Y, D=D, c=c,
                u=prognostic, f=self.f, Y_nm1=Y_nm1)

        next_u = 1/(1+self.dt*1j*self.f*self.implicit_coriolis) * \
                ((1 - self.dt*1j*self.f*(1-self.implicit_coriolis)) * \
                u + self.dt * \
                (np.diff(prognostic[1:] * Ku_full) / self.h_half[:-1] \
                + forcing))

        next_u[:SL.k+1] = prognostic[:SL.k+1]
        phi = prognostic[SL.k+1:]
        if SL.k > 0: # constant flux layer : K[:k] phi[:k] = K0 phi0
            phi = np.concatenate(( Ku_full[SL.k]* \
                    prognostic[SL.k+1]/Ku_full[:SL.k], phi))

        u_delta: complex = func_un(prognostic=prognostic,
                SL=SL, universal_funcs=businger())

        return next_u, phi, u_delta

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
                    theta[:SL.k+1], dz_theta[SL.k:]))
        Y_theta, D_theta, c_theta = self.__matrices_theta_FV(
                Ktheta_full, forcing_theta)
        Y_nm1 = tuple(np.copy(y) for y in Y_theta)
        self.__apply_sf_scheme(\
                func=self.dictsf_scheme_theta[SL.sf_scheme][1],
                Y=Y_theta, D=D_theta, c=c_theta, Y_nm1=Y_nm1,
                K_theta=Ktheta_full, forcing=forcing_theta,
                universal_funcs=businger(), SL=SL, SL_nm1=SL_nm1)
        prognostic_theta[:] = np.real(self.__backward_euler(Y=Y_theta,
                D=D_theta, c=c_theta, u=prognostic_theta, f=0.,
                Y_nm1=Y_nm1))

        next_theta = theta + self.dt * \
                np.diff(prognostic_theta[1:] * Ktheta_full) \
                / self.h_half[:-1]

        next_theta[:SL.k+1] = prognostic_theta[:SL.k+1]
        dz_theta = prognostic_theta[SL.k+1:]
        if SL.k > 0: # const flux layer : K[:k] phi[:k] = K[0] phi[0]
            dz_theta = np.concatenate(( Ktheta_full[SL.k]* \
                prognostic_theta[SL.k+1]/Ktheta_full[:SL.k],dz_theta))
        func_theta, _ = self.dictsf_scheme_theta[SL.sf_scheme]
        t_delta: float = func_theta(prognostic=prognostic_theta,
                universal_funcs=businger(), SL=SL)
        return next_theta, dz_theta, t_delta

    def initialize_theta(self, Neutral_case: bool):
        """
        Initialization method for theta.
        It is not crucial to have a perfect initial
        profile, but dz_theta is computed with care to
        have a continuous profile.
        """
        # approximate profile of theta: 265 then linear increasing
        theta: array = 265 + np.maximum(0, 0.01 * (self.z_half[:-1]-100))
        # But there's an angle, which is not a quadratic spline:
        index_angle = bisect.bisect_right(theta, theta[0] + 1e-4)
        theta[index_angle] = 265 + 0.01 * \
                (self.z_full[index_angle+1] - 100.)/2 * \
                (self.z_full[index_angle+1] - 100.)/ \
                (self.h_half[index_angle])

        # inversion of a system to find phi:
        ldiag, diag, udiag, rhs = \
                self.h_half[:-1] * np.ones(self.M)/6, \
                self.h_full * np.ones(self.M+1)*2/3, \
                self.h_half[:-1] * np.ones(self.M)/6, \
                np.diff(theta, prepend=0, append=0.)
        udiag[0] = rhs[0] = 0. # bottom Neumann condition
        ldiag[-1], diag[-1], rhs[-1] = 0., 1., 0.01 # top Neumann
        dz_theta: array = solve_linear((ldiag, diag, udiag), rhs)

        if Neutral_case: # If there's no stratification, theta=const
            theta[:] = 265.
            dz_theta[:] = 0.
        return theta, dz_theta

    def __mixing_lengths(self, tke: array, dzu2: array, N2: array,
            z_levels: array, SL: SurfaceLayerData, universal_funcs):
        """
            returns the mixing lengths (l_m, l_eps)
            for given entry parameters.
            dzu2 =||du/dz||^2 should be computed similarly
            to the shear
            all input and output are given on full levels.
            z_levels[k] can be set to delta_sl
            for a better link with the surface layer.
        """
        l_down = (self.kappa/self.C_m) * (self.C_m*self.c_eps)**.25 \
                * (SL.z_0M + np.zeros_like(z_levels))
        l_up = np.copy(l_down)
        l_up[-1] = l_down[-1] = self.lm_min
        assert z_levels[0] <= SL.delta_sl
        # to take into account surface layer we allow to change
        # z levels in this method
        h_half = np.diff(z_levels)
        k_modif = bisect.bisect_right(z_levels, SL.delta_sl)
        z_sl = z_levels[:k_modif]
        phi_m, *_ = universal_funcs
        #  Mixing length computation
        Rod = 0.2
        buoyancy = np.maximum(1e-12, N2)
        mxlm = np.maximum(self.lm_min, 2.*np.sqrt(tke) / \
                (Rod*np.sqrt(dzu2) + np.sqrt(Rod**2*dzu2+2.*buoyancy)))
        mxlm[-1] = self.lm_min
        # should not exceed linear mixing lengths:
        for j in range(self.M - 1, -1, -1):
            l_up[j] = min(l_up[j+1] + h_half[j], mxlm[j])

        g, theta_ref = 9.81, 283.
        ratio = SL.u_star**(4/3) * (self.kappa/self.C_m *
                (z_sl + SL.z_0M) / phi_m(z_sl * SL.inv_L_MO))**2 / \
                        ((SL.u_star**2 * phi_m(z_sl * SL.inv_L_MO) \
                        /self.c_eps/self.kappa/(z_sl + SL.z_0M) \
                        - g/theta_ref/self.c_eps*SL.t_star) \
                        **2)**(1/3)

        mxlm[:k_modif] = (ratio/l_up[:k_modif])**(3/5)
        mask_min_is_lup = mxlm[:k_modif]>l_up[:k_modif]
        mxlm[:k_modif][mask_min_is_lup] = \
                (ratio/ l_up[:k_modif]**(5/3))[mask_min_is_lup]
        if (mxlm[:k_modif] < l_up[:k_modif])[mask_min_is_lup].any():
            print("no solution of the link MOST-TKE")
        # TODO use directly 1/(lup tke) * (Ku_MOST)**2
        # mxlm[:k_modif] = 1/l_up[:k_modif] / tke[:k_modif] * \
        #       (SL.u_star * self.kappa * (z_sl + SL.z_0M) \
        #       / self.C_m / phi_m(z_sl * SL.inv_L_MO))**2

        # limiting l_down with the distance to the bottom:
        l_down[k_modif-1] = z_levels[k_modif-1]
        for j in range(k_modif, self.M+1):
            l_down[j] = min(l_down[j-1] + h_half[j-1], mxlm[j])
        l_down[:k_modif] = mxlm[:k_modif]

        l_m = np.maximum(np.sqrt(l_up*l_down), self.lm_min)
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
        z_min = 0.2
        xi = [np.linspace(-h/2, h/2, 15) for h in self.h_half[:-1]]
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
        u_star, t_star, _, _, inv_L_MO, _, _, \
                u_z0, t_z0, delta_sl, k1, sf_scheme, \
                Q_sw, Q_lw, SL_o = SL
        if sf_scheme in {"FV1", "FV pure"} or ignore_loglaw:
            allxi = [np.array(xi[m]) + self.z_half[m] for m in range(self.M)]
            k_1m: int = bisect.bisect_right(allxi[0], z_min)
            allxi[0] = allxi[0][k_1m:]
            sub_discrete_theta[0] = sub_discrete_theta[0][k_1m:]
            sub_discrete[0] = sub_discrete[0][k_1m:]
            return np.concatenate(allxi), \
                    np.concatenate(sub_discrete), \
                    np.concatenate(sub_discrete_theta), \


        if delta_sl is None:
            delta_sl = self.z_half[0] if sf_scheme in\
                    {"FV pure", "FV1"} else self.z_full[1]

        prognostic: array = np.concatenate((u_bar[:SL.k+1],
            phi[SL.k:]))

        # getting information of the surface layer (from which
        # we get the MOST profiles)
        z_log: array = np.geomspace(z_min, delta_sl, 20)

        _, _, psi_m, psi_h, *_ = businger()

        func_un, _ = self.dictsf_scheme[SL.sf_scheme]
        u_delta: complex = func_un(prognostic=prognostic,
                SL=SL, universal_funcs=businger())

        Pr = 1.# 4.8/7.8
        u_log: complex = u_z0 + u_star/self.kappa * \
                (np.log(1+z_log/SL.z_0M) - \
                psi_m(z_log*inv_L_MO) + psi_m(SL.z_0M*inv_L_MO)) \
                            * (u_delta - u_z0)/np.abs(SL.u_delta-u_z0)
        # u_delta/|SL.udelta| is u^{n+1}/|u^n| like in the
        # formulas
        theta_log: complex = t_z0 + Pr * t_star / self.kappa * \
                (np.log(1+z_log/SL.z_0H) - \
                psi_h(z_log*inv_L_MO) + psi_h(SL.z_0H*inv_L_MO))

        k2: int = bisect.bisect_right(z_oversampled, self.z_full[k1+1])

        z_freepart = []
        u_freepart = []
        theta_freepart = []

        if sf_scheme in {"FV free", }:
            # between the log profile and the next grid level:
            tilde_h = self.z_full[k1+1] - delta_sl
            assert 0 < tilde_h <= self.h_half[k1]
            xi = np.linspace(-tilde_h/2, tilde_h/2, 15)
            tau_slu, tau_slt = self.__tau_sl(SL, businger())
            alpha_slu = tilde_h/self.h_half[k1] + tau_slu
            alpha_slt = tilde_h/self.h_half[k1] + tau_slt

            u_tilde = 1/alpha_slu * (u_bar[k1] + tilde_h * tau_slu * \
                    (phi[k1]/3 + phi[k1+1]/6) - (1-alpha_slu)*u_z0)
            theta_tilde = 1/alpha_slt * (theta[k1] + tilde_h * tau_slt * \
                    (dz_theta[k1]/3 + dz_theta[k1+1]/6) - (1-alpha_slt)*t_z0)

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
            l_eps: array=None, universal_funcs=None):
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
            h_cl: float = self.defaulth_cl if np.abs(self.f) < 1e-10 \
                    else c_1*u_star/self.f
            G_full: array = self.kappa * self.z_full * \
                    (1 - self.z_full/h_cl)**2 \
                    * np.heaviside(1 - self.z_full/h_cl, 1)

            K_full: array = u_star * G_full + self.K_mol # viscosity
            K_full[0] = u_star * self.kappa * (delta_sl + SL.z_0M)
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
            tke.integrate_tke(self, SL, universal_funcs,
                    shear, K_full, l_eps, Ktheta_full, N2)
                    
            l_m[:], l_eps[:] = self.__mixing_lengths(tke.tke_full,
                    shear/K_full, N2, self.z_full, SL, universal_funcs)

            phi_z = self.__stability_temperature_phi_z(
                    C_1=self.C_1, l_m=l_m, l_eps=l_eps, N2=N2,
                    TKE=tke.tke_full)


            Ktheta_full: array = np.maximum(self.Ktheta_min,
                    self.C_s * phi_z * l_m * np.sqrt(tke.tke_full))

            Ku_full: array = np.maximum(self.K_min,
                    self.C_m * l_m * np.sqrt(tke.tke_full))

        else:
            raise NotImplementedError("Wrong turbulence scheme")
        return Ku_full, Ktheta_full

    def __stability_temperature_phi_z(self, C_1: float,
            l_m: array, l_eps: array, N2: array, TKE: array):
        return 1/(1+np.maximum(-0.5455,
                C_1*l_m*l_eps*np.maximum(1e-12, N2)/TKE))

    def __matrices_u_FD(self, K_full: array, forcing: array):
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

        c: array = np.copy(forcing)
        c[-1] += K_full[self.M] * 0. / self.h_half[self.M-1]
        Y = (np.zeros(self.M-1), np.ones(self.M), np.zeros(self.M-1))
        D = D_ldiag, D_diag, D_udiag
        return Y, D, c

    def __matrices_theta_FD(self, K_full: array, forcing: array):
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

        c: array = np.copy(forcing)
        c[-1] += K_full[self.M] * 0. / self.h_half[self.M-1]
        Y = (np.zeros(self.M-1), np.ones(self.M), np.zeros(self.M-1))
        D = D_ldiag, D_diag, D_udiag
        return Y, D, c

    def __visc_turb_FV(self, SL: SurfaceLayerData,
            turbulence: str="TKE", phi: array=None,
            old_phi: array=None, K_full: array=None, tke=None,
            l_m: array=None, l_eps: array=None,
            dz_theta: array=None, Ktheta_full: array=None,
            universal_funcs: tuple=None,
            ignore_tke_sl: bool=False):
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
            h_cl: float = self.defaulth_cl if np.abs(self.f) < 1e-10 \
                    else np.abs(c_1*u_star/self.f)
            G_full: array = self.kappa * self.z_full * \
                    (1 - self.z_full/h_cl)**2 \
                    * np.heaviside(1 - self.z_full/h_cl, 1)
            K_full = u_star * G_full + self.K_mol # viscosity
            K_full[0] = u_star * self.kappa * (delta_sl+SL.z_0M)
        elif turbulence == "TKE":
            k = bisect.bisect_right(self.z_full[1:], delta_sl)
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

            # computing buoyancy on half levels:
            g, theta_ref = 9.81, 283.
            shear_full = np.abs(K_full * phi * phi)
            # dzu2 and buoyancy at full levels for mixing lenghts:
            N2_full = g/theta_ref*dz_theta

            tke.integrate_tke(self, SL, universal_funcs,
                    shear_full, K_full, l_eps, Ktheta_full,
                    N2_full)


            # in surface layer we use MOST profiles:
            phi_m, phi_h, *_ = universal_funcs
            N2_full[:k+1] = g/theta_ref * SL.t_star * phi_h(\
                z_levels[:k+1] * SL.inv_L_MO) / self.kappa / \
                (z_levels[:k+1] + SL.z_0H)
            dzu2_full = np.concatenate((\
                    [np.abs(phi[0] * phi_second[0])],
                np.abs(phi[1:-1]*(phi_prime[:-1]+phi_second[1:])),
                [np.abs(phi[-1] * phi_prime[-1])]))
            # MOST profiles cause troubles with FV free (high res):
            # dzu2_full[:k+1] = SL.u_star**2 * \
            #         phi_m(z_levels[:k+1]*SL.inv_L_MO)**2 / \
            #         self.kappa**2 * (z_levels[:k+1] + SL.z_0M)**2
            # without this, We don't have equivalence between
            # FV2 and FV free because of dzu2_full[k].

            l_m[:], l_eps[:] = self.__mixing_lengths(tke.tke_full,
                    dzu2_full, N2_full,
                    z_levels, SL, universal_funcs)

            phi_z = self.__stability_temperature_phi_z(
                    C_1=self.C_1, l_m=l_m, l_eps=l_eps,
                    N2=g/theta_ref*dz_theta, TKE=tke.tke_full)
            phi_z[:k+1] = self.C_m * phi_m(z_levels[:k+1] * \
                    SL.inv_L_MO) / self.C_s / \
                    phi_h(z_levels[:k+1] * SL.inv_L_MO)

            Ktheta_full: array = np.maximum(self.Ktheta_min,
                    self.C_s * phi_z * l_m * np.sqrt(tke.tke_full))
            K_full = self.C_m * l_m * np.sqrt(tke.tke_full)

            if ignore_tke_sl:
                # When we ignore SL, K_0 is ~ the molecular viscosity:
                Ktheta_full[0] = (self.kappa * u_star*(SL.z_0M))\
                        / phi_h(SL.z_0M*inv_L_MO)
                K_full[0] = (self.kappa * u_star*(SL.z_0M)\
                        ) / phi_m(SL.z_0M*inv_L_MO)
                # since it does not work well, one can replace with:
                # Ktheta_full[0] = (self.kappa * u_star*(delta_sl+SL.z_0H))\
                #         / phi_h(delta_sl*inv_L_MO)
                # K_full[0] = (self.kappa * u_star*(delta_sl+SL.z_0M)\
                #         ) / phi_m(delta_sl*inv_L_MO)
                # by doing this, phi_0 would not mean correspond
                # to dzu(z0) neither to dzu(delta_sl)


        else:
            raise NotImplementedError("Wrong turbulence scheme")

        return K_full, Ktheta_full

    def __matrices_u_FV(self, K_full, forcing):
        """
            Creates the matrices D, Y, c such that the
            semi-discrete in space Ekman Stratified equation writes
            (d/dt Y - D) phi = c
        """
        Y_diag: array = np.concatenate(([0, 0.], 2/3*np.ones(self.M-1),
                                                        [0.]))
        Y_udiag: array = np.concatenate(([0, 0.],
            [self.h_half[m]/6./self.h_full[m] for m in range(1, self.M)]))
        Y_ldiag: array =  np.concatenate(([0.],
            [self.h_half[m-1]/6./self.h_full[m] for m in range(1, self.M)],
            [0.]))

        D_diag: array = np.concatenate(([0., 0.],
            [-2 * K_full[m] / self.h_half[m] / self.h_half[m-1]
                                        for m in range(1, self.M)],
            [-1.]))
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


    def __matrices_theta_FV(self, K_full: array, forcing: array):
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

    def __apply_sf_scheme(self, func, Y, D, c, Y_nm1=None, **kwargs):
        """
            Changes matrices Y, D, c on the first levels
            to use the surface flux scheme.
            _, func = self.dictsf_scheme[sf_scheme]
        """
        Y_sf, D_sf, c_sf, Y_nm1_sf = func(**kwargs)
        for y, y_sf in zip(Y, Y_sf):
            y_sf = np.array(y_sf)
            y[:y_sf.shape[0]] = y_sf

        if Y_nm1 is not None: # Y_nm1 may be different from Y
            for y, y_sf in zip(Y_nm1, Y_nm1_sf):
                y_sf = np.array(y_sf)
                y[:y_sf.shape[0]] = y_sf
        else: # otherwise we check the sf gives indeed the same Y
            assert Y_nm1_sf is Y_sf

        for d, d_sf in zip(D, D_sf):
            d_sf = np.array(d_sf)
            d[:d_sf.shape[0]] = d_sf

        c_sf = np.array(c_sf)
        c[:c_sf.shape[0]] = c_sf

    def initialization(self, u_0, phi_0, t_0, dz_theta,
            delta_sl, u_o, t_o, Q_sw, Q_lw, z_constant,
            delta_sl_o=0.):
        """
            initialize for FV free scheme. If this is not used,
            the continuity of the reconstruction cannot be
            guaranteed.
        """
        z_levels = self.z_full
        u_kp1 = u_const = 8.
        k = bisect.bisect_right(z_levels[1:], delta_sl)
        k_constant = bisect.bisect_right(z_levels[1:], z_constant)
        t_kp1 = t_const = t_0[k_constant]
        zkp1 = z_levels[k+1]
        z_constant = max(zkp1, z_constant)
        h_tilde = z_levels[k+1] - delta_sl
        phi_m, phi_h, *_ = businger()
        SL = friction_scales(u_const, delta_sl,
                t_const, businger(), u_o, delta_sl_o, t_o,
                large_ocean(), None, Q_sw, Q_lw, k, True)
        for _ in range(15):
            zeta = delta_sl * SL.inv_L_MO
            phi_0[k] = SL.u_star / self.kappa / \
                    (SL.z_0M+SL.delta_sl) * phi_m(zeta)
            dz_theta[k] = SL.t_star / self.kappa / \
                    (SL.z_0H+SL.delta_sl) * phi_h(zeta)
            # u_tilde + h_tilde (phi_0 / 6 + phi_1 / 3) = u_kp1
            # (subgrid reconstruction at the top of the volume)
            u_tilde = u_kp1 - h_tilde/6 * (phi_0[k]+2*phi_0[k+1])
            t_tilde = t_kp1 - h_tilde / 6 * (dz_theta[k] + \
                    2*dz_theta[k+1])
            u_delta = u_tilde - h_tilde / 6 * (2*phi_0[k]+phi_0[k+1])
            t_delta = t_tilde - h_tilde / 6 * (2*dz_theta[k] + \
                    dz_theta[k+1])

            SL = friction_scales(u_delta, delta_sl,
                t_delta, businger(), u_o, delta_sl_o, t_o,
                large_ocean(), None, SL.Q_sw, SL.Q_lw, k, True)
            # For LES simulation, putting a quadratic profile between
            # the log law and the constant profile :
            def func_z(z):
                return 1-((z_constant - z) / (z_constant - delta_sl))**2

            u_kp1 = u_delta + (u_const - u_delta) * func_z(zkp1)
            t_kp1 = t_delta + (t_const - t_delta) * func_z(zkp1)
            u_0[k+1:k_constant] = u_delta + (u_const-u_delta) *\
                    func_z(self.z_half[k+1:k_constant])
            t_0[k+1:k_constant] = t_delta + (t_const-t_delta) *\
                    func_z(self.z_half[k+1:k_constant])
            # compute_phi: with phi[k] = phi_0[k],
            # with phi[k_constant] = 0,
            # and the FV approximation
            def compute_dz(bottom_cond, var, h_half):
                """ solving the system of finite volumes:
                phi_{m-1}/12 + 10 phi_m / 12 + phi_{m+1} / 12 =
                        (tke_{m+1/2} - tke_{m-1/2})/h
                """
                ldiag = h_half[:-1] /6.
                diag = (h_half[1:] + h_half[:-1]) * 1/3.
                udiag = h_half[1:] /6.
                diag = np.concatenate(([1.], diag, [1.]))
                udiag = np.concatenate(([0.], udiag))
                ldiag = np.concatenate((ldiag, [0.]))
                rhs = np.concatenate(([bottom_cond],
                    np.diff(var), [0.]))
                return solve_linear((ldiag, diag, udiag), rhs)

            phi_0[k:] = compute_dz(phi_0[k],
                    np.concatenate(([u_tilde], u_0[k+1:])),
                    np.concatenate(([h_tilde], self.h_half[k+1:-1])))
            dz_theta[k:] = compute_dz(dz_theta[k],
                    np.concatenate(([t_tilde], t_0[k+1:])),
                    np.concatenate(([h_tilde], self.h_half[k+1:-1])))

        tau_u, tau_t = self.__tau_sl(SL, businger())
        alpha_u = h_tilde / self.h_half[k] + tau_u
        alpha_t = h_tilde / self.h_half[k] + tau_t

        u_0[k] = alpha_u * u_tilde - tau_u*h_tilde*(phi_0[k]/3 + \
                phi_0[k+1]/6) + (1-alpha_u) * SL.u_0
        t_0[k] = alpha_t * t_tilde - tau_t*h_tilde*(dz_theta[k]/3 + \
                dz_theta[k+1]/6) + (1 - alpha_t) * SL.t_0
        return u_0, phi_0, t_0, dz_theta, u_delta, t_delta

    def __tau_sl(self, SL: SurfaceLayerData,
            universal_funcs) -> (float, float):
        delta_sl, inv_L_MO = SL.delta_sl, SL.inv_L_MO
        _, _, psi_m, psi_h, Psi_m, Psi_h = universal_funcs
        k = bisect.bisect_right(self.z_full[1:], delta_sl)
        zk = self.z_full[k]
        def brackets_u(z):
            return (z+SL.z_0M)*np.log(1+z/SL.z_0M) - z + \
                    z*Psi_m(z*inv_L_MO)
        def brackets_theta(z):
            return (z+SL.z_0H)*np.log(1+z/SL.z_0H) - z + \
                    z*Psi_h(z*inv_L_MO)
        denom_u = np.log(1+delta_sl/SL.z_0M) \
                - psi_m(delta_sl*inv_L_MO)
        denom_theta = np.log(1+delta_sl/SL.z_0H)\
                - psi_h(delta_sl*inv_L_MO)

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
        """
            u(delta) = u_{1/2}
        """
        return prognostic[0]

    def __sf_udelta_FD2(self, prognostic, **_):
        """
            u(delta) = (u_{1/2} + u_{3/2}) / 2
        """
        return (prognostic[0] + prognostic[1])/2

    def __sf_udelta_FVpure(self, prognostic, **_):
        """
            u(delta) = Bar{u}_{1/2} - h (phi_1 - phi_0)/24
        """
        return prognostic[0] - self.h_half[0]* \
                (prognostic[2] - prognostic[1])/24

    def __sf_udelta_FV1(self, prognostic, **_):
        """
            u(delta) = Bar{u}_{1/2}
        """
        return prognostic[0]

    def __sf_udelta_FV2(self, prognostic, **_):
        """
            u(delta) = Bar{u}_{3/2} - h (phi_2/6 + phi_1/3)
        """
        return prognostic[1] - self.h_half[1] * \
                (prognostic[3]/6 + prognostic[2]/3)

    def __sf_udelta_FVfree(self, prognostic, SL,
            universal_funcs, **_):
        """
            u(delta) = 1/a ( Bar{u}_{k+1/2}
                    - ~h^2/h (phi_{k} / 3 + phi_{k+1} / 6))
        """
        tau_slu, _ = self.__tau_sl(SL, universal_funcs)
        k = bisect.bisect_right(self.z_full[1:], SL.delta_sl)
        tilde_h = self.z_full[k+1] - SL.delta_sl
        alpha = tilde_h/self.h_half[k]+tau_slu
        return (prognostic[k] - tilde_h*tilde_h/self.h_half[k] * \
                (prognostic[k+1] / 3 + prognostic[k+2] / 6) - \
                (1 - alpha) * SL.u_0) \
                / alpha

    ####### DEFINITION OF SF SCHEMES : FIRST LINES OF Y,D,c #####
    # each method must return Y, D, c:
    # Y: 3-tuple of tuples of sizes (j-1, j, j)
    # D: 3-tuple of tuples of sizes (j-1, j, j)
    # c: tuple of size j
    # they represent the first j lines of the matrices.
    # for Y and D, the tuples are (lower diag, diag, upper diag)

    def __sf_YDc_FDpure(self, K_u, forcing, SL, **_):
        """
            Y = (           1                   ,   0   )
            D = (-K/h^2 - u*^2 / (h|u(z_a)-u0|) , K/h^2 )
            c = ( F + u_0 u*^2 / (h|u(z_a)-u0|) )
        """
        u_star, u_delta = SL.u_star, SL.u_delta
        jump = np.abs(u_delta - SL.u_0)
        Y = ((), (1.,), (0.,))
        D = ((), (-K_u[1]/self.h_full[1]/self.h_half[0] - \
                u_star**2 / jump /self.h_half[0], ),
                (K_u[1]/self.h_full[1]/self.h_half[0],))
        c = (forcing[0] + u_star**2 * SL.u_0 / jump / self.h_half[0],)
        return Y, D, c, Y

    def __sf_YDc_FD2(self, K_u, SL, **_):
        """
            Y = (            0          ,              0         )
            D = ( -K/h - u*^2/(2|ua-u0|),  K/h - u*^2/(2|ua-u0|) )
            c = (   u0 u*^2/|ua-u0|    )
        """
        u_star, u_delta = SL.u_star, SL.u_delta
        jump = np.abs(u_delta - SL.u_0)
        Y = ((), (0.,), (0.,))
        D = ((), (-K_u[1]/self.h_full[1] - u_star**2 / jump / 2,),
                (K_u[1]/self.h_full[1] - u_star**2 / jump / 2,))
        c = (u_star**2 * SL.u_0 / jump,)
        return Y, D, c, Y

    def __sf_YDc_FVpure(self, K_u, forcing, SL, **_):
        """
            Y = (0     ,    0     , 0 )
                (1     ,    0     , 0 )
            D = ( 1  ,   h/24 - K|ua-u0|/u*^2 ,  - h/24  )
                ( 0  ,          -K/h          ,    K/h   )
            c = ( -u0  )
                (  F   )
        """
        u_star, u_delta = SL.u_star, SL.u_delta
        jump = np.abs(u_delta - SL.u_0)
        Y = ((1.,), (0., 0.), (0., 0.))
        D = ((0.,), (1, -K_u[0] / self.h_half[0]),
                (-K_u[0]*jump/u_star**2+self.h_half[0]/24,
                    K_u[1]/self.h_half[0]),
                (-self.h_half[0]/24,))
        c = (-SL.u_0, forcing[0])
        return Y, D, c, Y

    def __sf_YDc_FV1(self, K_u, forcing, SL, **_):
        """
            Y = (0     ,    0     , 0 )
                (1     ,    0     , 0 )
            D = ( 1  ,  - K|ua-u0|/u*^2 ,  0  )
                ( 0  ,     -K/h         , K/h )
            c = (  -u0  )
                (   F   )
        """
        u_star, u_delta = SL.u_star, SL.u_delta
        jump = np.abs(u_delta - SL.u_0)
        Y = ((1.,), (0., 0.), (0., 0.))
        D = ((0.,), (1, -K_u[0] / self.h_half[0]),
                (-K_u[0]*jump/u_star**2, K_u[1]/self.h_half[0]))
        c = (-SL.u_0, forcing[0])
        return Y, D, c, Y

    def __sf_YDc_FV2(self, K_u, forcing, SL, universal_funcs, **_):
        """
            Y = (0 , 0  , 0 ,  0)
                (0 , 0  , 0 ,  0)
                (0 , 1  , 0 ,  0)
            D = (1 , -R ,           0          ,  0   )
                (0 , 1  ,  -K|u-u0|/u*^2 - h/3 ,  -h/6)
                (0 , 0  ,          -K/h        ,  K/h )
            c = (  u0 (R-1) )
                (  -u0      )
                (   F       )
            where R is the ratio of norms defined by MOST profiles.
            (Bar{u}_{1/2} is overriden so no need to overthink this)
        """
        u_star, u_delta, t_star, t_delta = SL.u_star, \
                SL.u_delta, SL.t_star, SL.t_delta
        jump = np.abs(u_delta - SL.u_0)
        _, _, _, _, Psi_m, _ = universal_funcs
        inv_L_MO = t_star / t_delta / u_star**2 * self.kappa * 9.81
        def f(z):
            return (z+SL.z_0M)*np.log(1+z/SL.z_0M) - z + \
                    z * Psi_m(z*inv_L_MO)

        try:
            ratio_norms = (f(self.z_full[1]) - f(0)) / \
                    (f(self.z_full[2]) - f(self.z_full[1]))
        except ZeroDivisionError:
            ratio_norms = 0.
        Y = ((0., 1.), (0., 0., 0.), (0., 0., 0.))
        D = ((0., 0.), (1., 1., -K_u[1] / self.h_half[1]),
                (-ratio_norms, -K_u[1]*jump/u_star**2 - \
                        self.h_half[1]/3, K_u[2]/self.h_half[1]),
                (0., -self.h_half[1]/6))
        c = (SL.u_0 * (ratio_norms-1), -SL.u_0, forcing[1])
        return Y, D, c, Y

    def __sf_YDc_FVfree(self, K_u, forcing, universal_funcs,
            SL, SL_nm1, **_):
        """
            Y  = ( 0    ,           0  ,    0,         ,  0   )
                 ( 1/a  ,  tau ~h/(3a) ,  tau ~h/(6a)  ,  0   )
                 (  0   ,   ~h/(6h)    ,  (~h+h)/(3h)  ,h/(6h))

            D = ( -1  , ~h^2/(3h)+Ka|ua-u0|/u*^2, ~h^2/(6h)    , 0     )
                ( 0   ,-K/~h                    ,  K/~h        , 0     )
                ( 0   , K/(h~h)                 ,-K(1/h+1/~h)/h, K/h^2 )

            c = (                   u_0                          )
                (  forcing_{k+1/2} + (partial_t+if) (u0(1-a)/ a) )
                (      1/h (forcing_{k+3/2} - forcing_{k+1/2})   )
            after that, the matrices are filled for every 0 <= m < k
            with 0 for Y
            D: (ldiag, diag, udiag) = (0, -1, R)
            where R is a ratio of norms given by the log law.
            this last part is actually overriden in the end so
            it should not be considered too seriously.
        """
        u_star, u_delta, delta_sl, inv_L_MO = SL.u_star, \
                SL.u_delta, SL.delta_sl, SL.inv_L_MO
        k = bisect.bisect_right(self.z_full[1:], delta_sl)
        tilde_h = self.z_full[k+1] - delta_sl
        tau_slu, _ = self.__tau_sl(SL, universal_funcs)
        alpha_sl = tilde_h/self.h_half[k] + tau_slu
        tau_slu_nm1, _ = self.__tau_sl(SL_nm1, universal_funcs)
        alpha_sl_nm1 = tilde_h/self.h_half[k] + tau_slu_nm1
        # note that delta_sl is assumed constant here.
        # its variation would need much more changes.

        Y = ((1/alpha_sl, tilde_h / 6 / self.h_full[k+1]),
                (0., tilde_h*tau_slu/3/alpha_sl,
                    tilde_h/3/self.h_full[k+1] + \
                            self.h_half[k+1]/3/self.h_full[k+1]),
                (0., tilde_h*tau_slu/6/alpha_sl,
                    self.h_half[k+1]/6/self.h_full[k+1]))
        Y_nm1 = ((1/alpha_sl_nm1, tilde_h / 6 / self.h_full[k+1]),
                (0., tilde_h*tau_slu_nm1/3/alpha_sl_nm1,
                    tilde_h/3/self.h_full[k+1] + \
                            self.h_half[k+1]/3/self.h_full[k+1]),
                (0., tilde_h*tau_slu_nm1/6/alpha_sl_nm1,
                    self.h_half[k+1]/6/self.h_full[k+1]))

        D = ((0., K_u[k+0]/tilde_h/self.h_full[k+1]),
                (-1., -K_u[k+0] / tilde_h,
                    -K_u[k+1]/tilde_h/self.h_full[k+1] - \
                            K_u[k+1] / self.h_half[k+1] / self.h_full[k+1]),
                (K_u[k+0]*np.abs(u_delta - SL.u_0)*alpha_sl/u_star**2 + \
                        tilde_h**2 / 3 / self.h_half[k],
                    K_u[k+1]/tilde_h, K_u[k+2]/self.h_full[k+1]/self.h_half[k+1]),
                (tilde_h**2 / 6 / self.h_half[k], 0.))
        u01ma_a_n = SL.u_0 * (1 - alpha_sl)/alpha_sl
        u01ma_a_nm1 = SL_nm1.u_0 * (1-alpha_sl_nm1) / alpha_sl_nm1
        partial_tpif_u01ma_a = (u01ma_a_n - u01ma_a_nm1)/self.dt + \
                1j*self.f*(self.implicit_coriolis*u01ma_a_n + \
                (1 - self.implicit_coriolis) * u01ma_a_nm1)
        c = (SL.u_0, forcing[k+0] + partial_tpif_u01ma_a,
                (forcing[k+1] - forcing[k+0])/self.h_full[k+1])

        Y = (np.concatenate((np.zeros(k), y)) for y in Y)
        Y_nm1 = (np.concatenate((np.zeros(k), y)) for y in Y_nm1)

        *_, Psi_m, _ = universal_funcs
        def f(z):
            return (z+SL.z_0M)*np.log(1+z/SL.z_0M) - z + \
                    z * Psi_m(z*inv_L_MO)

        try:
            ratio_norms = np.array([(f(self.z_full[m+1]) - \
                    f(self.z_full[m])) / \
                    (f(self.z_full[m+2]) - f(self.z_full[m+1])) \
                        for m in range(k)])
        except ZeroDivisionError:
            ratio_norms = np.zeros(k)

        D = (np.concatenate((np.zeros(k), D[0])),
                np.concatenate((-np.ones(k), D[1])),
                np.concatenate((ratio_norms, D[2])),
                np.concatenate((np.zeros(k), D[3])))
        c = np.concatenate(((ratio_norms - 1)*SL.u_0, c))
        return Y, D, c, Y_nm1


    ####### DEFINITION OF SF SCHEMES : VALUE OF theta(delta_sl) ##
    # The method must use the prognostic variables and delta_sl
    # to return theta(delta_sl).
    # the prognostic variables are theta for FD and 
    # (theta_{1/2}, ... theta_{k+1/2}, phit_k, ...phit_M) for FV.
    def __sf_thetadelta_FDpure(self, prognostic, **_):
        """
            t(delta) = t_{1/2}
        """
        return prognostic[0]

    def __sf_thetadelta_FD2(self, prognostic, **_):
        """
            t(delta) = (t_{1/2} + t_{1/2})/2
        """
        return (prognostic[0] + prognostic[1])/2

    def __sf_thetadelta_FVpure(self, prognostic, **_):
        """
            t(delta) = Bar{t}_{1/2} - h (dzt_1 - dzt_0) / 24
        """
        return prognostic[0] - self.h_half[0]* \
                (prognostic[2] - prognostic[1])/24

    def __sf_thetadelta_FV1(self, prognostic, **_):
        """
            t(delta) = Bar{t}_{1/2}
        """
        return prognostic[0]

    def __sf_thetadelta_FV2(self, prognostic, **_):
        """
            t(delta) = Bar{t}_{1/2}
        """
        return prognostic[1] - self.h_half[1] * \
                (prognostic[3]/6 + prognostic[2]/3)

    def __sf_thetadelta_FVfree(self, prognostic, SL,
            universal_funcs, **_):
        """
            t(delta) = 1/a ( Bar{t}_{k+1/2} - (1 - a) * t_0
                    - ~h^2/h (dzt_{k} / 3 + dzt_{k+1} / 6))
        """
        _, tau_slt = self.__tau_sl(SL, universal_funcs)
        k = bisect.bisect_right(self.z_full[1:], SL.delta_sl)
        zk = self.z_full[k]
        tilde_h = self.z_full[k+1] - SL.delta_sl
        alpha = tilde_h/self.h_half[k]+tau_slt
        return (prognostic[k] - tilde_h*tilde_h/self.h_half[k] * \
                (prognostic[k+1] / 3 + prognostic[k+2] / 6) - \
                (1 - alpha) * SL.t_0) \
                / alpha


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
    def __sf_YDc_FDpure_theta(self, K_theta, SL, universal_funcs, **_):
        """
            Y = (           1         ,   0   )
            D = (-K/h^2 - C_H|ua| / h , K/h^2 )
            c = ( t_0 * C_H|u| / h )
        """
        inv_L_MO = SL.t_star / SL.t_delta / SL.u_star**2 * self.kappa * 9.81
        _, _, _, psi_h, _, _ = universal_funcs
        phi_stab = psi_h(inv_L_MO * SL.delta_sl)
        ch_du = SL.u_star * self.kappa / \
                (np.log(1+SL.delta_sl/SL.z_0H)-phi_stab)
        Y = ((), (1.,), (0.,))
        D = ((), (-K_theta[1]/self.h_full[1]/self.h_half[0] - \
                ch_du /self.h_half[0],),
                (K_theta[1]/self.h_full[1]/self.h_half[0],))
        c = (SL.t_0*ch_du / self.h_half[0],)
        return Y, D, c, Y

    def __sf_YDc_FD2_theta(self, K_theta, SL, universal_funcs, **_):
        """
            Y = (            0       ,          0        )
            D = ( -K/h - C_H|u| / 2  ,  K/h - C_H|u| / 2 )
            c = (    C_H|u| * t_0    )
        """
        inv_L_MO = SL.t_star / SL.t_delta / SL.u_star**2 * self.kappa * 9.81
        _, _, _, psi_h, _, _ = universal_funcs
        phi_stab = psi_h(inv_L_MO * SL.delta_sl)
        ch_du = SL.u_star * self.kappa / (np.log(1+SL.delta_sl/SL.z_0H)-phi_stab)
        Y = ((), (0.,), (0.,))
        D = ((), (-K_theta[1]/self.h_full[1] - ch_du / 2,),
                (K_theta[1]/self.h_full[1] - ch_du / 2,))
        c = (ch_du * SL.t_0,)
        return Y, D, c, Y

    def __sf_YDc_FV1_theta(self, K_theta, SL, universal_funcs, **_):
        """
            Y = (0     ,    0     , 0 )
                (1     ,    0     , 0 )
            D = ( -1 ,    K /(C_H|u|),  0  )
                ( 0  ,     -K/h      , K/h )
            c = (  t_0)
                (  0  )
        """
        _, _, _, psi_h, _, _ = universal_funcs
        inv_L_MO = SL.t_star / SL.t_delta / SL.u_star**2 * self.kappa * 9.81
        ch_du = SL.u_star * self.kappa / \
                (np.log(1+SL.delta_sl/SL.z_0H)-psi_h(SL.delta_sl*inv_L_MO))

        Y = ((1.,), (0., 0.), (0., 0.))
        D = ((0.,), (-1, -K_theta[0] / self.h_half[0]),
                (K_theta[0]/ch_du, K_theta[1]/self.h_half[0]),
                (0.,))
        c = (SL.t_0, 0.)
        return Y, D, c, Y

    def __sf_YDc_FVpure_theta(self, K_theta, SL,
            universal_funcs, **_):
        """
            Y = (0     ,    0     , 0 )
                (1     ,    0     , 0 )
            D = (-1  , -h/24 + K / (C_H|u|) ,   h/24  )
                ( 0  ,          -K/h        ,    K/h  )
            c = (  t_0 )
                (   0  )
        """
        _, _, _, psi_h, _, _ = universal_funcs
        inv_L_MO = SL.t_star / SL.t_delta / SL.u_star**2 * self.kappa * 9.81
        ch_du = SL.u_star * self.kappa / \
                (np.log(1+SL.delta_sl/SL.z_0H)-psi_h(SL.delta_sl*inv_L_MO))

        Y = ((1.,), (0., 0.), (0., 0.))
        D = ((0.,), (-1, -K_theta[0] / self.h_half[0]),
                (K_theta[0]/ch_du-self.h_half[0]/24,
                    K_theta[1]/self.h_half[0]),
                (self.h_half[0]/24,))
        c = (SL.t_0, 0.)
        return Y, D, c, Y

    def __sf_YDc_FV2_theta(self, K_theta, SL, universal_funcs, **_):
        _, _, _, psi_h, _, Psi_h = universal_funcs
        """
            Y = (0 , 0  , 0 ,  0)
                (0 , 0  , 0 ,  0)
                (0 , 1  , 0 ,  0)
            D = (1 , -R ,           0       , 0   )
                (0 ,-1  ,   K / (C_H|u|)+h/3, h/6 )
                (0 , 0  ,          -K/h     , K/h )
            c = (  t_0 (R-1) )
                (     t_0    )
                (      0     )
            where R is the ratio of norms defined by MOST profiles.
            (Bar{u}_{1/2} is overriden so no need to overthink this)
        """
        def f(z):
            return (z+SL.z_0H)*np.log(1+z/SL.z_0H) \
                - z + z* Psi_h(z*SL.inv_L_MO)

        ch_du = SL.u_star * self.kappa / \
                (np.log(1+SL.delta_sl/SL.z_0H)-psi_h(SL.delta_sl*SL.inv_L_MO))

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
        c = (SL.t_0*(ratio_norms-1), SL.t_0, 0.)
        return Y, D, c, Y

    def __sf_YDc_FVfree_theta(self, K_theta, universal_funcs, forcing,
            SL, SL_nm1, **_):
        """
            Y  = ( 0    ,           0  ,    0,         ,  0   )
                 ( 1/a  ,  tau ~h/(3a) ,  tau ~h/(6a)  ,  0   )
                 (  0   ,   ~h/(6h)    ,  (~h+h)/(3h)  ,h/(6h))

            D = ( -1  , ~h^2/(3h)+Ka/(C_H|u|), ~h^2/(6h)    , 0     )
                ( 0   ,-K/~h                 ,  K/~h        , 0     )
                ( 0   , K/(h~h)              ,-K(1/h+1/~h)/h, K/h^2 )

            c = (                  t_0                     )
                (  forcing_{k+1/2} + partial_t(t_0(1-a)/ a))
                (  1/h (forcing_{k+3/2} - forcing_{k+1/2}) )
            after that, the matrices are filled for every 0 <= m < k
            with 0 for Y
            D: (ldiag, diag, udiag) = (0, 1, -R)
            where R is a ratio of norms given by the log law.
            this last part is actually overriden in the end so
            it should not be considered too seriously.
        """
        k = bisect.bisect_right(self.z_full[1:], SL.delta_sl)
        tilde_h = self.z_full[k+1] - SL.delta_sl
        _, _, _, psi_h, _, Psi_h = universal_funcs
        _, tau_slt = self.__tau_sl(SL, universal_funcs)
        alpha_slt = tilde_h/self.h_half[k] + tau_slt
        _, tau_slt_nm1 = self.__tau_sl(SL_nm1, universal_funcs)
        alpha_slt_nm1 = tilde_h/self.h_half[k] + tau_slt_nm1
        ch_du = SL.u_star * self.kappa / \
                (np.log(1+SL.delta_sl/SL.z_0H) - \
                psi_h(SL.delta_sl*SL.inv_L_MO))

        Y = ((1/alpha_slt, tilde_h / 6 / self.h_full[k+1]),
                (0., tilde_h*tau_slt/3/alpha_slt,
                    tilde_h/3/self.h_full[k+1] + \
                            self.h_half[k+1]/3/self.h_full[k+1]),
                (0., tilde_h*tau_slt/6/alpha_slt,
                    self.h_half[k+1]/6/self.h_full[k+1]))
        Y_nm1 = ((1/alpha_slt_nm1, tilde_h / 6 / self.h_full[k+1]),
                (0., tilde_h*tau_slt_nm1/3/alpha_slt_nm1,
                    tilde_h/3/self.h_full[k+1] + \
                            self.h_half[k+1]/3/self.h_full[k+1]),
                (0., tilde_h*tau_slt_nm1/6/alpha_slt_nm1,
                    self.h_half[k+1]/6/self.h_full[k+1]))

        D = ((0., K_theta[k+0]/tilde_h/self.h_full[k+1]),
                (-1., -K_theta[k+0] / tilde_h,
                    -K_theta[k+1]/tilde_h/self.h_full[k+1] - \
                            K_theta[k+1] / self.h_half[k+1] / self.h_full[k+1]),
                (K_theta[k+0]*alpha_slt/ch_du + \
                        tilde_h**2 / 3 / self.h_half[k],
                    K_theta[k+1]/tilde_h, K_theta[k+2]/self.h_full[k+1]/self.h_half[k+1]),
                (tilde_h**2 / 6 / self.h_half[k], 0.))
        rhs_part_tilde = (SL.t_0 * (1 - alpha_slt)/alpha_slt - \
                SL_nm1.t_0 *(1-alpha_slt_nm1)/alpha_slt_nm1)/self.dt
        c = (SL.t_0, forcing[k+0] + rhs_part_tilde,
                (forcing[k+1] - forcing[k+0])/self.h_full[k+1])
        Y = (np.concatenate((np.zeros(k), y)) for y in Y)
        Y_nm1 = (np.concatenate((np.zeros(k), y)) for y in Y_nm1)

        def f(z):
            return (z+SL.z_0H)*np.log(1+z/SL.z_0H) \
                - z + z* Psi_h(z*SL.inv_L_MO)

        try:
            ratio_norms = np.array([(f(self.z_full[m+1]) - \
                    f(self.z_full[m])) / \
                    (f(self.z_full[m+2]) - f(self.z_full[m+1])) \
                        for m in range(k)])
        except ZeroDivisionError:
            ratio_norms = np.zeros(k)

        D = (np.concatenate((np.zeros(k), D[0])),
                np.concatenate((np.ones(k), D[1])),
                np.concatenate((-ratio_norms, D[2])),
                np.concatenate((np.zeros(k), D[3])))
        c = np.concatenate(((ratio_norms - 1)*SL.t_0, c))
        return Y, D, c, Y_nm1
