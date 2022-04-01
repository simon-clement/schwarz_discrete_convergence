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
from universal_functions import Businger_et_al_1971 as businger
def pr(var, name="atm"):
    print(var, name)

array = np.ndarray
# TKE coefficients:
coeff_FV_big = 1/3. # 1/3 or 5/12
coeff_FV_small = 1/6. # 1/6 or 1/12

class SurfaceLayerData(NamedTuple):
    """
        Handles the data of the SL at one time step.
    """
    u_star: float # friction scale u*
    t_star: float # friction scale t*
    z_0M: float # roughness length for momentum
    z_0H: float # roughness length for theta
    inv_L_MO: float # inverse of Obukhov length:
    u_delta: complex # value of u at z=delta_sl
    t_delta: float # value of theta at z=delta_sl
    SST: float # Sea Surface Temperature (z=0)
    delta_sl: float # Height of the Surface Layer
    k: int # index for which z_k < delta_sl < z_{k+1}
    sf_scheme: str # Name of the surface flux scheme

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
                "FV1" : (self.__sf_thetadelta_FV1,
                                self.__sf_YDc_FV1_theta),
                "FV2" : (self.__sf_thetadelta_FV2,
                                self.__sf_YDc_FV2_theta),
                "FV free" : (self.__sf_thetadelta_FVfree,
                                self.__sf_YDc_FVfree_theta),
                }

    def FV(self, u_t0: array, phi_t0: array, forcing: array,
            SST:array, delta_sl: float, sf_scheme: str="FV pure",
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
        SL: SurfaceLayerData = self.__friction_scales(u_delta,
                delta_sl, t_delta, SST[0], businger(), sf_scheme, k)
        ignore_tke_sl = sf_scheme in {"FV pure", "FV1"}

        tke, dz_tke = self.__initialize_tke(SL,
                Neutral_case, ignore_tke_sl)
        theta, dz_theta = self.__initialize_theta(Neutral_case)

        Ku_full: array = self.K_min + np.zeros(self.M+1)
        Ktheta_full: array = self.Ktheta_min + np.zeros(self.M+1)

        z_levels_sl = np.copy(self.z_full)
        z_levels_sl[k] = self.z_full[k] if ignore_tke_sl else delta_sl
        l_m, l_eps = self.__mixing_lengths(
                np.concatenate((tke, [self.e_min])),
                phi_t0*phi_t0, 9.81*dz_theta/283.,
                z_levels_sl, SL, businger())

        phi, old_phi = phi_t0, np.copy(phi_t0)
        u_current: array = np.copy(u_t0)
        all_u_star = []
        ret_u_current, ret_tke, ret_dz_tke, ret_SL = [], [], [], []
        ret_phi, ret_theta, ret_dz_theta, ret_leps = [], [], [], []

        for n in range(1,N+1):
            # Compute friction scales
            SL_nm1, SL = SL, self.__friction_scales(u_delta, delta_sl,
                    t_delta, SST[n], businger(), sf_scheme, k)
            all_u_star += [SL.u_star]

            # Compute viscosities
            Ku_full, Ktheta_full, tke, dz_tke = self.__visc_turb_FV(
                    SL, turbulence=turbulence, phi=phi,
                    old_phi=old_phi, l_m=l_m, l_eps=l_eps,
                    K_full=Ku_full, tke=tke, dz_tke=dz_tke,
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

            if store_all:
                ret_u_current += [np.copy(u_current)]
                ret_tke += [np.copy(tke)]
                ret_dz_tke += [np.copy(dz_tke)]
                ret_phi += [np.copy(phi)]
                ret_theta += [np.copy(theta)]
                ret_dz_theta += [np.copy(dz_theta)]
                ret_leps += [np.copy(l_eps)]
                ret_SL += [SL]

        if store_all:
            return ret_u_current, ret_phi, ret_tke, ret_dz_tke, \
                    all_u_star, ret_theta, ret_dz_theta, ret_leps, \
                    ret_SL
        return u_current, phi, tke, dz_tke, all_u_star, theta, \
                dz_theta, l_eps, SL


    def FD(self, u_t0: array, forcing: array, SST:array,
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
        tke = np.ones(self.M+1) * self.e_min
        tke[self.z_half <= 250] = self.e_min + 0.4*(1 - \
                self.z_half[self.z_half <= 250] / 250)**3
        theta: array = 265 + np.maximum(0, # potential temperature
                0.01 * (self.z_half[:-1]-100))
        if Neutral_case:
            theta[:] = 265.
            tke[:] = self.e_min

        # Initializing viscosities and mixing lengths:
        Ku_full: array = self.K_min + np.zeros_like(tke)
        Ktheta_full: array = self.Ktheta_min + np.zeros_like(tke)
        l_m = self.lm_min*np.ones(self.M+1)
        l_eps = self.leps_min*np.ones(self.M+1)

        u_current: array = np.copy(u_t0)
        old_u: array = np.copy(u_current)
        all_u_star = []
        all_u, all_tke, all_theta, all_leps = [], [], [], []
        for n in range(1,N+1):
            forcing_current: array = forcing[n]
            u_delta = func_un(prognostic=u_current, delta_sl=delta_sl)
            t_delta = func_theta(prognostic=theta)
            # Compute friction scales
            SL: SurfaceLayerData= self.__friction_scales(u_delta,
                    delta_sl, t_delta, SST[n], businger(),sf_scheme,0)

            all_u_star += [SL.u_star]

            # Compute viscosities
            Ku_full, Ktheta_full, tke = self.__visc_turb_FD(SL=SL,
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

            if store_all:
                all_u += [np.copy(u_current)]
                all_tke += [np.copy(tke)]
                all_theta += [np.copy(theta)]
                all_leps += [np.copy(l_eps)]

        if store_all:
            return all_u, all_tke, all_u_star, all_theta, all_leps
        return u_current, tke, all_u_star, theta, l_eps

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


    def __initialize_tke(self, SL: SurfaceLayerData,
            Neutral_case: bool, ignore_sl: bool):
        """
        Initialization method for the tke.
        It is not crucial to have a perfect initial
        profile, but dz_tke is computed with care to
        have a continuous profile.
        """
        tke = np.ones(self.M) * self.e_min
        if not Neutral_case:
            # N2 and inv_L_MO are 0 at initialization
            # so we use a neutral e_sl assuming l_m=l_eps
            e_sl = np.maximum(SL.u_star**2 / \
                    np.sqrt(self.C_m*self.c_eps), self.e_min)
            z_half, k = np.copy(self.z_half), SL.k
            z_half[k] = z_half[k] if ignore_sl else SL.delta_sl
            tke[z_half[:-1] <= 250] = self.e_min + e_sl*(1 - \
                (z_half[:-1][z_half[:-1] <= 250] - \
                SL.delta_sl) / (250.- SL.delta_sl))**3
            # inversion of a system to find dz_tke:
            return tke, self.__compute_dz_tke(e_sl, tke,
                    SL.delta_sl, SL.k, ignore_sl)

        return tke, np.zeros(self.M+1)

    def __initialize_theta(self, Neutral_case: bool):
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

    def __compute_tke_full(self, tke: array, dz_tke: array,
            SL: SurfaceLayerData, ignore_sl: bool,
            l_eps: array, universal_funcs):
        """
            projection of the tke reconstruction on z_full
            if ignore_sl is False, the closest z level below
            delta_sl is replaced with delta_sl.
        """
        h_half, k = np.copy(self.h_half), SL.k
        z_sl = np.copy(self.z_full[:k])
        if not ignore_sl:
            h_half[k] = self.z_full[k+1] - SL.delta_sl

        phi_m, *_ = universal_funcs

        KN2 = 9.81/283. * SL.t_star * SL.u_star
        shear = SL.u_star**3 * phi_m(z_sl* SL.inv_L_MO) / \
                self.kappa / (z_sl + SL.z_0M)
        # TKE inside the surface layer: 
        tke_sl = np.maximum(((l_eps[:k]/self.c_eps * \
                (shear - KN2))**2)**(1/3), self.e_min)
        # TKE at the top of the surface layer:
        tke_k = [tke[k] - h_half[k]*dz_tke[k]*coeff_FV_big - \
                    h_half[k]*dz_tke[k+1]*coeff_FV_small]
        # Combining both with the TKE above the surface layer:
        return np.concatenate((tke_sl, tke_k,
            tke[k:] + h_half[k:-1]*dz_tke[k+1:]*coeff_FV_big + \
                    h_half[k:-1]*dz_tke[k:-1]*coeff_FV_small))

    def __compute_dz_tke(self, e_sl: float, tke: array,
            delta_sl: float, k: int, ignore_sl: bool):
        """ solving the system of finite volumes:
                phi_{m-1}/12 + 10 phi_m / 12 + phi_{m+1} / 12 =
                    (tke_{m+1/2} - tke_{m-1/2})/h
            The coefficients 1/12, 5/12 are replaced
                by coeff_FV_small, coeff_FV_big.
        """
        ldiag = self.h_half[:-2] *coeff_FV_small
        diag = self.h_full[1:-1] * 2*coeff_FV_big
        udiag = self.h_half[1:-1] *coeff_FV_small
        h_tilde = self.z_full[k+1] - delta_sl
        if ignore_sl: # If the tke is reconstructed until z=0:
            assert k == 0
            h_tilde = self.h_half[k]
        diag = np.concatenate(([h_tilde*coeff_FV_big], diag, [1.]))
        udiag = np.concatenate(([h_tilde*coeff_FV_small], udiag))
        ldiag = np.concatenate((ldiag, [0.]))
        rhs = np.concatenate(([tke[0]-e_sl],
            np.diff(tke), [0.]))

        # GRID LEVEL k-1 AND BELOW: dz_tke=0:
        diag[:k], udiag[:k] = 1., 0.
        rhs[:k] = 0.
        ldiag[:k] = 0. # ldiag[k-1] is for cell k
        # GRID level k: tke(z=delta_sl) = e_sl
        diag[k] = h_tilde*coeff_FV_big
        udiag[k] = h_tilde*coeff_FV_small
        rhs[k] = tke[k] - e_sl
        # GRID LEVEL k+1: h_tilde used in continuity equation
        ldiag[k] = h_tilde *coeff_FV_small
        diag[k+1] = (h_tilde+self.h_half[k+1]) * coeff_FV_big
        dz_tke = solve_linear((ldiag, diag, udiag), rhs)
        return dz_tke

    def __integrate_tke_FV(self, tke: array, dz_tke: array,
            shear_half: array, K_full: array,
            SL: SurfaceLayerData, l_eps: array,
            buoy_half: array, ignore_sl: bool,
            universal_funcs):
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
        Ke_full = K_full * self.C_e / self.C_m
        # Bottom value:
        phi_m, *_ = universal_funcs
        k = bisect.bisect_right(self.z_full[1:], SL.delta_sl)
        z_sl = np.copy(self.z_full[:k+1])
        z_sl[-1] = z_sl[-1] if ignore_sl else SL.delta_sl
        # buoyancy and shear in surface layer:
        KN2_sl = 9.81/283. * SL.t_star * SL.u_star
        shear_sl = SL.u_star**3 * phi_m(z_sl * SL.inv_L_MO) / \
                self.kappa / (z_sl + SL.z_0M)
        e_sl = np.maximum(((l_eps[:k+1]/self.c_eps * \
                (shear_sl - KN2_sl))**2)**(1/3), self.e_min)

        l_eps_half = full_to_half(l_eps)

        h_tilde = self.z_full[k+1] - SL.delta_sl
        if ignore_sl:
            h_tilde = self.h_half[k]

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
        even_diag = np.concatenate(([-self.h_half[0]*coeff_FV_big],
                2*coeff_FV_big*self.h_full[1:-1] / self.dt + \
                Ke_full[1:-1]/self.h_half[1:-1] + \
                Ke_full[1:-1] / self.h_half[:-2],
                [1]
            ))
        even_udiag = np.concatenate(([1],
                buoy_half[1:]*PATANKAR[1:] / tke[1:] + \
                self.c_eps * np.sqrt(tke[1:]) / l_eps_half[1:]))
        even_uudiag = np.concatenate(([-self.h_half[0]*coeff_FV_small],
                self.h_half[1:-1]*coeff_FV_small/self.dt - \
                Ke_full[2:]/self.h_half[1:-1]))

        even_ldiag = np.concatenate((
                - buoy_half[:-1]*PATANKAR[:-1] / tke[:-1] - \
                self.c_eps * np.sqrt(tke[:-1]) / l_eps_half[:-1],
                [0]))
        even_lldiag = np.concatenate((
                self.h_half[:-2]*coeff_FV_small/self.dt - \
                Ke_full[:-2]/self.h_half[:-2],
                [0]
                ))

        even_rhs = np.concatenate(( [e_sl[0]],
                1/self.dt * (self.h_half[:-2]* dz_tke[:-2]*coeff_FV_small \
                + 2*self.h_full[1:-1]* dz_tke[1:-1]*coeff_FV_big \
                + self.h_half[1:-1]* dz_tke[2:]*coeff_FV_small) \
                + np.diff(shear_half - buoy_half*(1-PATANKAR)),
                [0]))
        # e(delta_sl) = e_sl :
        even_diag[k] = -h_tilde*coeff_FV_big
        even_udiag[k] = 1.
        even_uudiag[k] = -h_tilde*coeff_FV_small
        even_rhs[k] = e_sl[k]
        # first grid levels above delta_sl:
        even_diag[k+1] = (self.h_half[k+1]+h_tilde)*coeff_FV_big/self.dt + \
                Ke_full[k+1]/ h_tilde + \
                Ke_full[k+1] / self.h_half[k+1]
        even_lldiag[k] = h_tilde*coeff_FV_small/self.dt - Ke_full[k]/h_tilde
        even_rhs[k+1] = (h_tilde* dz_tke[k]*coeff_FV_small \
                +(self.h_half[k+1]+h_tilde)* dz_tke[k+1]*coeff_FV_big \
                +self.h_half[k+1]*dz_tke[k+2]*coeff_FV_small)/self.dt \
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

        tke, dz_tke = deinterlace(\
                solve_linear((lldiag, ldiag, diag, udiag, uudiag),
                                rhs))

        tke_full = self.__compute_tke_full(tke, dz_tke, SL, ignore_sl,
                l_eps, universal_funcs)
        return tke, dz_tke, tke_full

    def __integrate_tke(self, tke: array, shear: array,
            K_full: array, SL: SurfaceLayerData,
            l_eps: array, Ktheta_full: array,
            N2: array, universal_funcs):
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

        phi_m, *_ = universal_funcs
        KN2_sl = 9.81/283. * SL.t_star * SL.u_star
        shear_sl = SL.u_star**3 * phi_m(SL.delta_sl * SL.inv_L_MO) / \
                self.kappa / (SL.delta_sl + SL.z_0M)

        e_sl = np.maximum(((l_eps[SL.k]/self.c_eps * \
                (shear_sl - KN2_sl))**2)**(1/3), self.e_min)

        rhs_e = np.concatenate(([e_sl], [tke[m]/self.dt + shear[m]
                for m in range(1, self.M)],
            [tke[self.M]/self.dt]))
        for m in range(1, self.M):
            if shear[m] <= Ktheta_full[m] * N2[m]: # Patankar trick
                diag_e[m] += Ktheta_full[m] * N2[m] / tke[m]
            else: # normal handling of buoyancy
                rhs_e[m] -=  Ktheta_full[m] * N2[m]
        # if delta_sl is inside the computational domain:
        k = bisect.bisect_right(self.z_full[1:], SL.delta_sl)
        rhs_e[:k+1] = e_sl # prescribe e=e(delta_sl)
        diag_e[:k+1] = 1. # because lower levels are not used
        udiag_e[:k+1] = ldiag_e[:k] = 0.

        return solve_linear((ldiag_e, diag_e, udiag_e), rhs_e)

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

        # limiting l_down with the distance to the bottom:
        l_down[k_modif-1] = z_levels[k_modif-1]
        for j in range(k_modif, self.M+1):
            l_down[j] = min(l_down[j-1] + h_half[j-1], mxlm[j])
        l_down[:k_modif] = mxlm[:k_modif]

        l_m = np.maximum(np.sqrt(l_up*l_down), self.lm_min)
        l_eps = np.minimum(l_down, l_up)
        return l_m, l_eps

    def reconstruct_TKE(self, tke: array, dz_tke: array,
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
        xi = [np.linspace(-h/2, h/2, 15) for h in self.h_half[:-1]]
        xi[1] = np.linspace(-self.h_half[1]/2, self.h_half[1]/2, 40)
        xi[0] = np.linspace(z_min-self.h_half[0]/2, self.h_half[0]/2, 40)
        sub_discrete: List[array] = [tke[m] + \
                (dz_tke[m+1] + dz_tke[m]) * xi[m]/2 \
                + (dz_tke[m+1] - dz_tke[m]) / (2 * self.h_half[m]) * \
                (xi[m]**2 - self.h_half[m]**2/12) \
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
                h = self.h_half[m]
                values = np.array((tke[m], h*dz_tke[m],
                    h*dz_tke[m+1], \
                    tke[m] - 5/12*h*dz_tke[m] - h*dz_tke[m+1]/12,
                    tke[m] + h*dz_tke[m]/12 + h*dz_tke[m+1]*5/12))
                polynomial = coefficients @ values
                sub_discrete[m] = np.sum([(rihi/h**i) * xi[m]**i\
                    for i, rihi in enumerate(polynomial)], axis=0)

        # If the SL is not handled, we can stop now:
        if ignore_tke_sl:
            tke_oversampled = np.concatenate(sub_discrete)
            z_oversampled = np.concatenate([np.array(xi[m]) + \
                    self.z_half[m] for m in range(self.M)])

            return z_oversampled, tke_oversampled
        tke_oversampled = np.concatenate(sub_discrete[1:])
        z_oversampled = np.concatenate([np.array(xi[m]) + \
                self.z_half[m] for m in range(1, self.M)])
        # The part above the SL starts at z_oversampled[k2:].
        k2: int = bisect.bisect_right(z_oversampled,
                self.z_full[SL.k+1])

        # We now compute the reconstruction inside the SL:
        phi_m, *_ = universal_funcs
        z_log: array = np.array((z_min, SL.delta_sl))
        # buoyancy and shear inside the SL:
        KN2 = 9.81/283. * SL.t_star * SL.u_star
        shear = SL.u_star**3 * phi_m(z_log * SL.inv_L_MO) / \
                self.kappa / (z_log + SL.z_0M)
        from scipy.interpolate import interp1d
        z_levels = np.copy(self.z_full)
        if not ignore_tke_sl:
            z_levels[SL.k] = SL.delta_sl
        f = interp1d(z_levels, l_eps, fill_value="extrapolate")
        l_eps_log = np.maximum(f(z_log), 0.)
        # With the quasi-equilibrium hypothesis, we get the tke:
        tke_log = np.maximum(((l_eps_log/self.c_eps * \
                (shear - KN2))**2)**(1/3), self.e_min)

        # We finally compute the reconstruction just above SL:
        z_freepart = [] # we call "freepart" the zone
        k = SL.k # between the log profile and the next grid level:
        tilde_h = self.z_full[k+1] - SL.delta_sl
        assert 0 < tilde_h <= self.h_half[k]
        xi = np.linspace(-tilde_h/2, tilde_h/2, 15)

        tke_freepart = tke[k] + (dz_tke[k+1] + dz_tke[k]) * xi/2 \
                + (dz_tke[k+1] - dz_tke[k]) / (2 * tilde_h) * \
                (xi**2 - tilde_h**2/12)

        if abs(coeff_FV_small - 1./12) < 1e-4:
            values = np.array((tke[k], tilde_h*dz_tke[k],
                tilde_h*dz_tke[k+1], \
                tke[k] - 5/12*tilde_h*dz_tke[k] - tilde_h*dz_tke[k+1]/12,
                tke[k] + tilde_h*dz_tke[k]/12 + tilde_h*dz_tke[k+1]*5/12))
            polynomial = coefficients @ values
            tke_freepart = np.sum([(rihi/tilde_h**i) * xi**i\
                for i, rihi in enumerate(polynomial)], axis=0)

        z_freepart = SL.delta_sl + xi + tilde_h / 2
        # we return the concatenation of the 3 zones:
        return np.concatenate((z_log,
                            z_freepart, z_oversampled[k2:])), \
                np.concatenate((tke_log,
                            tke_freepart, tke_oversampled[k2:]))


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
                SST, delta_sl, k1, sf_scheme = SL
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
        u_log: complex = u_star/self.kappa * \
                (np.log(1+z_log/SL.z_0M) - \
                psi_m(z_log*inv_L_MO) + psi_m(SL.z_0M*inv_L_MO)) \
                            * u_delta/np.abs(SL.u_delta)
        # u_delta/|SL.udelta| is u^{n+1}/|u^n| like in the
        # formulas
        theta_log: complex = SST + Pr * t_star / self.kappa * \
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
            tke: array=None, theta: array=None,
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
            tke[:] = np.maximum(self.e_min,
                    self.__integrate_tke(tke, shear, K_full, SL,
                    l_eps, Ktheta_full, N2, universal_funcs))
            l_m[:], l_eps[:] = self.__mixing_lengths(tke,
                    shear/K_full, N2, self.z_full, SL, universal_funcs)

            phi_z = self.__stability_temperature_phi_z(
                    C_1=self.C_1, l_m=l_m, l_eps=l_eps, N2=N2,
                    TKE=tke)


            Ktheta_full: array = np.maximum(self.Ktheta_min,
                    self.C_s * phi_z * l_m * np.sqrt(tke))

            Ku_full: array = np.maximum(self.K_min,
                    self.C_m * l_m * np.sqrt(tke))

        else:
            raise NotImplementedError("Wrong turbulence scheme")
        return Ku_full, Ktheta_full, tke

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

        c: array = forcing
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

        c: array = forcing
        c[-1] += K_full[self.M] * 0. / self.h_half[self.M-1]
        Y = (np.zeros(self.M-1), np.ones(self.M), np.zeros(self.M-1))
        D = D_ldiag, D_diag, D_udiag
        return Y, D, c

    def __visc_turb_FV(self, SL: SurfaceLayerData,
            turbulence: str="TKE", phi: array=None,
            old_phi: array=None, K_full: array=None, tke: array=None,
            dz_tke: array=None, l_m: array=None, l_eps: array=None,
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
            buoy_half = full_to_half(Ktheta_full*g/theta_ref*dz_theta)

            tke[:], dz_tke[:], tke_full = \
                    self.__integrate_tke_FV(tke, dz_tke,
                    shear_half=shear_half, K_full=K_full,
                    SL=SL, l_eps=l_eps, buoy_half=buoy_half,
                    ignore_sl=ignore_tke_sl,
                    universal_funcs=universal_funcs)

            if (tke_full < self.e_min).any() or \
                    (tke < self.e_min).any():
                tke_full = np.maximum(tke_full, self.e_min)
                tke = np.maximum(tke, self.e_min)
                dz_tke = self.__compute_dz_tke(tke_full[k],
                                    tke, delta_sl, k, ignore_tke_sl)

            # dzu2 and buoyancy at full levels for mixing lenghts:
            N2_full = g/theta_ref*dz_theta
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

            l_m[:], l_eps[:] = self.__mixing_lengths(tke_full,
                    dzu2_full, N2_full,
                    z_levels, SL, universal_funcs)

            phi_z = self.__stability_temperature_phi_z(
                    C_1=self.C_1, l_m=l_m, l_eps=l_eps,
                    N2=g/theta_ref*dz_theta, TKE=tke_full)
            phi_z[:k+1] = self.C_m * phi_m(z_levels[:k+1] * \
                    SL.inv_L_MO) / self.C_s / \
                    phi_h(z_levels[:k+1] * SL.inv_L_MO)

            Ktheta_full: array = np.maximum(self.Ktheta_min,
                    self.C_s * phi_z * l_m * np.sqrt(tke_full))
            K_full = self.C_m * l_m * np.sqrt(tke_full)

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

        return K_full, Ktheta_full, tke, dz_tke

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

    def __friction_scales(self, u_delta: float, delta_sl: float,
            t_delta: float, SST: float,
            universal_funcs, sf_scheme: str, k: int) -> SurfaceLayerData:
        """
        Computes (u*, t*) with a fixed point algorithm.
        returns a SurfaceLayerData containing all the necessary data.
        universal_funcs is the tuple (phim, phih, psim, psih, Psim, Psih)
        defined in universal_functions.py
        other parameters are defined in the SurfaceLayerData class.
        """
        _, _, psim, psis, *_ = universal_funcs
        t_star: float = (t_delta-SST) * \
                (0.0180 if t_delta > SST else 0.0327)
        z_0M = 0.1
        z_0H = 0.1
        u_star: float = (self.kappa *np.abs(u_delta) / \
                np.log(1 + delta_sl/z_0M ) )
        for _ in range(12):
            zeta = self.kappa * delta_sl * 9.81*\
                    (t_star / t_delta) / u_star**2
            Cd    = self.kappa**2 / \
                    (np.log(1+delta_sl/z_0M) - psim(zeta))**2
            Ch    = self.kappa * np.sqrt(Cd) / \
                    (np.log(1+delta_sl/z_0H) - psis(zeta))
            u_star = np.sqrt(Cd) * np.abs(u_delta)
            t_star = ( Ch / np.sqrt(Cd) ) * (t_delta - SST)
            z_0M = z_0H = self.K_mol / self.kappa / u_star
        inv_L_MO = t_star / t_delta / u_star**2 * \
                self.kappa * 9.81
        return SurfaceLayerData(u_star, t_star, z_0M, z_0H,
                inv_L_MO, u_delta, t_delta, SST, delta_sl, k,
                sf_scheme)

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

    def __sf_udelta_FVfree(self, prognostic, SL,
            universal_funcs, **_):
        tau_slu, _ = self.__tau_sl(SL, universal_funcs)
        k = bisect.bisect_right(self.z_full[1:], SL.delta_sl)
        tilde_h = self.z_full[k+1] - SL.delta_sl
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

    def __sf_YDc_FDpure(self, K_u, forcing, SL, **_):
        u_star, u_delta = SL.u_star, SL.u_delta
        Y = ((), (1.,), (0.,))
        D = ((), (-K_u[1]/self.h_full[1]/self.h_half[0] - \
                u_star**2 / np.abs(u_delta)/self.h_half[0], ),
                (K_u[1]/self.h_full[1]/self.h_half[0],))
        c = (forcing[0],)
        return Y, D, c, Y

    def __sf_YDc_FD2(self, K_u, SL, **_):
        u_star, u_delta = SL.u_star, SL.u_delta
        Y = ((), (0.,), (0.,))
        D = ((), (-K_u[1]/self.h_full[1] - u_star**2 / np.abs(u_delta) / 2,),
                (K_u[1]/self.h_full[1] - u_star**2 / np.abs(u_delta) / 2,))
        c = (0.,)
        return Y, D, c, Y

    def __sf_YDc_FVpure(self, K_u, forcing, SL, **_):
        u_star, u_delta = SL.u_star, SL.u_delta
        Y = ((1.,), (0., 0.), (0., 0.))
        D = ((0.,), (1, -K_u[0] / self.h_half[0]),
                (-K_u[0]*np.abs(u_delta)/u_star**2+self.h_half[0]/24,
                    K_u[1]/self.h_half[0]),
                (-self.h_half[0]/24,))
        c = (0., forcing[0])
        return Y, D, c, Y

    def __sf_YDc_FV1(self, K_u, forcing, SL, **_):
        u_star, u_delta = SL.u_star, SL.u_delta
        Y = ((1.,), (0., 0.), (0., 0.))
        D = ((0.,), (1, -K_u[0] / self.h_half[0]),
                (-K_u[0]*np.abs(u_delta)/u_star**2, K_u[1]/self.h_half[0]))
        c = (0., forcing[0])
        return Y, D, c, Y

    def __sf_YDc_FV2(self, K_u, forcing, SL, universal_funcs, **_):
        u_star, u_delta, t_star, t_delta = SL.u_star, \
                SL.u_delta, SL.t_star, SL.t_delta
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
                (-ratio_norms, -K_u[1]*np.abs(u_delta)/u_star**2 - \
                        self.h_half[1]/3, K_u[2]/self.h_half[1]),
                (0., -self.h_half[1]/6))
        c = (0., 0., forcing[1])
        return Y, D, c, Y

    def __sf_YDc_FVfree(self, K_u, forcing, universal_funcs,
            SL, SL_nm1, **_):
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
                (K_u[k+0]*np.abs(u_delta)*alpha_sl/u_star**2 + \
                        tilde_h**2 / 3 / self.h_half[k],
                    K_u[k+1]/tilde_h, K_u[k+2]/self.h_full[k+1]/self.h_half[k+1]),
                (tilde_h**2 / 6 / self.h_half[k], 0.))
        c = (0.+0j, forcing[k+0],
                (forcing[k+1] - forcing[k+0])/self.h_full[k+1])

        Y = (np.concatenate((np.zeros(k), y)) for y in Y)
        Y_nm1 = (np.concatenate((np.zeros(k), y)) for y in Y_nm1)

        *_, Psi_m, _ = universal_funcs
        def f(z):
            return (z+SL.z_0M)*np.log(1+z/SL.z_0M) - z + \
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
        return Y, D, c, Y_nm1


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

    def __sf_thetadelta_FV1(self, prognostic, **_):
        return prognostic[0]

    def __sf_thetadelta_FV2(self, prognostic, **_):
        return prognostic[1] - self.h_half[1] * \
                (prognostic[3]/6 + prognostic[2]/3)

    def __sf_thetadelta_FVfree(self, prognostic, SL,
            universal_funcs, **_):
        _, tau_slt = self.__tau_sl(SL, universal_funcs)
        k = bisect.bisect_right(self.z_full[1:], SL.delta_sl)
        zk = self.z_full[k]
        tilde_h = self.z_full[k+1] - SL.delta_sl
        alpha = tilde_h/self.h_half[k]+tau_slt
        return (prognostic[k] - tilde_h*tilde_h/self.h_half[k] * \
                (prognostic[k+1] / 3 + prognostic[k+2] / 6) - \
                (1 - alpha) * SL.SST) \
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
        inv_L_MO = SL.t_star / SL.t_delta / SL.u_star**2 * self.kappa * 9.81
        _, _, _, psi_h, _, _ = universal_funcs
        phi_stab = psi_h(inv_L_MO * SL.delta_sl)
        ch_du = SL.u_star * self.kappa / \
                (np.log(1+SL.delta_sl/SL.z_0H)-phi_stab)
        Y = ((), (1.,), (0.,))
        D = ((), (-K_theta[1]/self.h_full[1]/self.h_half[0] - \
                ch_du /self.h_half[0],),
                (K_theta[1]/self.h_full[1]/self.h_half[0],))
        c = (SL.SST*ch_du / self.h_half[0],)
        return Y, D, c, Y

    def __sf_YDc_FD2_theta(self, K_theta, SL, universal_funcs, **_):
        inv_L_MO = SL.t_star / SL.t_delta / SL.u_star**2 * self.kappa * 9.81
        _, _, _, psi_h, _, _ = universal_funcs
        phi_stab = psi_h(inv_L_MO * SL.delta_sl)
        ch_du = SL.u_star * self.kappa / (np.log(1+SL.delta_sl/SL.z_0H)-phi_stab)
        Y = ((), (0.,), (0.,))
        D = ((), (-K_theta[1]/self.h_full[1] - ch_du / 2,),
                (K_theta[1]/self.h_full[1] - ch_du / 2,))
        c = (ch_du * SL.SST,)
        return Y, D, c, Y

    def __sf_YDc_FV1_theta(self, K_theta, SL, universal_funcs, **_):
        _, _, _, psi_h, _, _ = universal_funcs
        inv_L_MO = SL.t_star / SL.t_delta / SL.u_star**2 * self.kappa * 9.81
        ch_du = SL.u_star * self.kappa / \
                (np.log(1+SL.delta_sl/SL.z_0H)-psi_h(SL.delta_sl*inv_L_MO))

        Y = ((1.,), (0., 0.), (0., 0.))
        D = ((0.,), (-1, -K_theta[0] / self.h_half[0]),
                (K_theta[0]/ch_du, K_theta[1]/self.h_half[0]),
                (0.,))
        c = (SL.SST, 0.)
        return Y, D, c, Y

    def __sf_YDc_FVpure_theta(self, K_theta, SL,
            universal_funcs, **_):
        _, _, _, psi_h, _, _ = universal_funcs
        inv_L_MO = SL.t_star / SL.t_delta / SL.u_star**2 * self.kappa * 9.81
        ch_du = SL.u_star * self.kappa / \
                (np.log(1+SL.delta_sl/SL.z_0H)-psi_h(SL.delta_sl*inv_L_MO))

        Y = ((1.,), (0., 0.), (0., 0.))
        D = ((0.,), (-1, -K_theta[0] / self.h_half[0]),
                (K_theta[0]/ch_du-self.h_half[0]/24,
                    K_theta[1]/self.h_half[0]),
                (self.h_half[0]/24,))
        c = (SL.SST, 0.)
        return Y, D, c, Y

    def __sf_YDc_FV2_theta(self, K_theta, SL, universal_funcs, **_):
        _, _, _, psi_h, _, Psi_h = universal_funcs
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
        c = (SL.SST*(ratio_norms-1), SL.SST, 0.)
        return Y, D, c, Y

    def __sf_YDc_FVfree_theta(self, K_theta, universal_funcs, forcing,
            SL, SL_nm1, **_):
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
        rhs_part_tilde = (SL.SST * (1 - alpha_slt)/alpha_slt - \
                SL_nm1.SST *(1-alpha_slt_nm1)/alpha_slt_nm1)/self.dt
        c = (SL.SST, forcing[k+0] + rhs_part_tilde,
                (forcing[k+1] - forcing[k+0])/self.h_full[k+1])
        Y = (np.concatenate((np.zeros(k), y)) for y in Y)
        Y_nm1 = (np.concatenate((np.zeros(k), y)) for y in Y_nm1)

        def f(z):
            return (z+SL.z_0H)*np.log(1+z/SL.z_0H) \
                - z + z* Psi_h(z*SL.inv_L_MO)

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
        c = np.concatenate(((ratio_norms - 1)*SL.SST, c))
        return Y, D, c, Y_nm1
