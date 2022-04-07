"""
    defines the bulk function friction_scales()
    and the structure SurfaceLayerData.
"""
import numpy as np
from typing import NamedTuple
from utils_linalg import orientation
from shortwave_absorption import integrated_shortwave_frac_sl

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
    u_0: complex # value of u at z=z_M (surface)
    t_0: float # value of theta at z=z_M (surface)
    delta_sl: float # Depth of the Surface Layer (<0 for ocean)
    k: int # index for which z_k < delta_sl < z_{k+1}
    sf_scheme: str # Name of the surface flux scheme
    Q_sw: float # shortwave radiation flux
    Q_lw: float # longwave radiation flux
    SL_other: 'SurfaceLayerData' # data of other domain

def friction_scales(ua_delta: float, delta_sl_a: float,
        ta_delta: float, univ_funcs_a,
        uo_delta: float, delta_sl_o: float,
        to_delta: float, univ_funcs_o,
        sf_scheme: str, Q_sw: float, Q_lw: float,
        k: int, is_atm: bool) -> SurfaceLayerData:
    """
    Computes (u*, t*) with a fixed point algorithm.
    returns a SurfaceLayerData containing all the necessary data.
    universal_funcs is the tuple (phim, phih, psim, psih, Psim, Psih)
    defined in universal_functions.py
    other parameters are defined in the SurfaceLayerData class.
    It is possible to give to this method
    {u, t}a_delta={u, t}a(z=0) together with delta_sl_a = 0.
    if is_atm returns SL for atmosphere else returns oceanic SL
    """
    _, _, psim_a, psis_a, _, _, = univ_funcs_a
    _, _, psim_o, psis_o, _, _, = univ_funcs_o
    ta_delta_Kelvin = ta_delta
    ta_delta_Kelvin += 273. if ta_delta < 150 else 0.
    rho0 = 1024.
    kappa = 0.4
    c_p_atm = 1004.
    c_p_oce = 3985.
    K_mol_oce = 1e-4
    mu_m = 6.7e-2
    alpha_eos = 1.8e-4
    # ATMOSPHERIC friction scales:
    t_star: float = (ta_delta-to_delta) * \
            (0.0180 if ta_delta > to_delta else 0.0327)
    u_star: float = (kappa *np.abs(ua_delta - uo_delta) / \
            np.log(1 + delta_sl_a/.1 ) )
    lambda_u = np.sqrt(1/rho0) # u_o* = lambda_u u_a*
    lambda_t = np.sqrt(1./rho0)*c_p_atm/c_p_oce
    for _ in range(42):
        uo_star, to_star = lambda_u*u_star, lambda_t*t_star
        inv_L_a = 9.81 * kappa * t_star / ta_delta_Kelvin / u_star**2
        inv_L_o = 9.81 * kappa * alpha_eos * to_star / uo_star**2
        za_0M = za_0H = K_mol_oce / kappa / u_star / mu_m
        zo_0M = zo_0H = K_mol_oce / kappa / uo_star
        zeta_a = np.clip(delta_sl_a*inv_L_a, -50., 50.)
        zeta_o = np.clip(-delta_sl_o*inv_L_o, -50., 50.)
        # Pelletier et al, 2021, equations 31, 32:
        rhs_31 = np.log(1+delta_sl_a/za_0M) - psim_a(zeta_a) + \
                lambda_u * (np.log(1-delta_sl_o/zo_0M) - \
                    psim_o(zeta_o))
        # Radiative fluxes:
        turhocp = to_star * uo_star * rho0 * c_p_oce
        if abs(turhocp) > 1e-30:
            term_lw = 1 - Q_lw / turhocp
            term_Qw = Q_sw * integrated_shortwave_frac_sl(\
                    inv_L_o, delta_sl_o) / turhocp
        else:
            print("Warning (bulk): dividing by t*u* where u*o=",
                    uo_star, "t*o", to_star)
            term_lw = term_Qw = 0.

        # Pelletier et al, 2021, equation (43):
        rhs_32 = np.log(1+delta_sl_a/za_0H) - psis_a(zeta_a) + \
                lambda_t * term_lw * (np.log(1-delta_sl_o/zo_0M)-\
                    psis_o(zeta_o)) - lambda_t * term_Qw

        C_D    = (kappa / rhs_31)**2
        Ch    = kappa * np.sqrt(C_D) / rhs_32
        previous_u_star, previous_t_star = u_star, t_star
        u_star = np.sqrt(C_D) * np.abs(ua_delta-uo_delta)
        t_star = ( Ch / np.sqrt(C_D)) * (ta_delta - to_delta)
    if abs(previous_u_star - u_star) > 1e-10: # we attained
        print("bulk convergence not attained (u*): error of",
                abs(previous_u_star - u_star))
    if abs(previous_t_star - t_star) > 1e-10: # convergence
        print("bulk convergence not attained (t*): error of",
                abs(previous_t_star - t_star))

    uo_star, to_star = lambda_u*u_star, lambda_t*t_star
    inv_L_a = 9.81 * kappa * t_star / ta_delta_Kelvin / u_star**2
    inv_L_o = 9.81 * kappa * alpha_eos * to_star / uo_star**2
    za_0M = za_0H = K_mol_oce / kappa / u_star / mu_m
    zo_0M = zo_0H = K_mol_oce / kappa / uo_star
    u_zM: complex = ua_delta - orientation(ua_delta) * u_star \
            / kappa * (np.log(1+delta_sl_a/za_0M) - \
            psim_a(delta_sl_a*inv_L_a))
    # theta_zM does not see radiative fluxes.
    theta_zM: float = ta_delta - t_star \
            / kappa * (np.log(1+delta_sl_a/za_0H) - \
            psis_a(delta_sl_a*inv_L_a))
    if is_atm:
        SL_o = SurfaceLayerData(uo_star, to_star, zo_0M, zo_0H,
                inv_L_o, uo_delta, to_delta, u_zM, theta_zM,
                delta_sl_o, None, None, Q_sw, Q_lw, None)
        return SurfaceLayerData(u_star, t_star, za_0M, za_0H,
                inv_L_a, ua_delta, ta_delta, u_zM, theta_zM,
                delta_sl_a, k, sf_scheme, Q_sw, Q_lw, SL_o)

    SL_a = SurfaceLayerData(u_star, t_star, za_0M, za_0H,
            inv_L_a, ua_delta, ta_delta, u_zM, theta_zM,
            delta_sl_a, None,
            None, Q_sw, Q_lw, None)
    return SurfaceLayerData(uo_star, to_star, zo_0M, zo_0H,
            inv_L_o, uo_delta, to_delta, u_zM, theta_zM,
            delta_sl_o, k, sf_scheme, Q_sw, Q_lw, SL_a)
