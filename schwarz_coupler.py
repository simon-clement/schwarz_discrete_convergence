#!/usr/bin/python3
"""
    This module is here to simulate the OA coupling.
"""
from typing import NamedTuple, List, Dict
import numpy as np
from scipy import interpolate
from atm1DStratified import Atm1dStratified
from ocean1DStratified import Ocean1dStratified

INIT_U_ATM = 8. + 0j
INIT_THETA_ATM = 280.

class StateOce(NamedTuple):
    """
        all the output variables of the atmosphere (resp. ocean) model
        A part of them will be used by the ocean (resp. atmosphere).
    """
    u_delta: np.ndarray # SL Momentum at all time steps
    t_delta: np.ndarray # SL Temperature at all time steps
    last_tstep: Dict
    other: Dict

class StateAtm(NamedTuple):
    """
        all the output variables of the atmosphere (resp. ocean) model
        A part of them will be used by the ocean (resp. atmosphere).
    """
    u_delta: np.ndarray # SL Momentum at all time steps
    t_delta: np.ndarray # SL Temperature at all time steps
    u_star: np.ndarray
    t_star: np.ndarray
    last_tstep: Dict
    other: Dict

class NumericalSetting(NamedTuple):
    """
        all the input variables of both the atmosphere and ocean
    """
    T: float # Integration time (in seconds)
    sf_scheme_a: str # should be accepted by simulator_oce
    sf_scheme_o: str # should be accepted by simulator_atm
    delta_sl_a: float # height of the top of ASL
    delta_sl_o: float # height of the bottom of OSL (<0)
    Q_sw: np.ndarray # shortwave radiative flux in ocean
    Q_lw: np.ndarray # shortwave radiative flux in ocean


def schwarz_coupling(simulator_oce: Ocean1dStratified,
        simulator_atm: Atm1dStratified,
        parameters: NumericalSetting,
        **kwargs)-> (List[StateAtm], List[StateOce]):
    """
        computes the coupling between the two models
        Atm1dStratified and Ocean1dStratified.
    """
    atm_state, oce_state = [initialization_atmosphere(parameters,
        simulator_atm)], []
    NUMBER_SCHWARZ_ITERATION = 1
    for _ in range(NUMBER_SCHWARZ_ITERATION):
        oce_state += [compute_ocean(simulator_oce,
            atm_state[-1], parameters, **kwargs)]
        atm_state += [compute_atmosphere(simulator_atm,
            oce_state[-1], parameters, **kwargs)]
    return atm_state, oce_state

def initialization_atmosphere(numer_set: NumericalSetting,
        simulator_atm: Atm1dStratified) -> StateAtm:
    """
    returns a State that can be used by ocean model for integration.
    """
    N = int(numer_set.T/simulator_atm.dt) # Number of time steps
    u_star = INIT_U_ATM * simulator_atm.kappa / \
            np.log(numer_set.delta_sl_a / 0.1)
    return StateAtm(u_delta=np.ones(N+1) * INIT_U_ATM,
            t_delta=np.ones(N+1) * INIT_THETA_ATM,
            u_star=np.ones(N+1) * u_star,
            t_star=np.ones(N+1) * 1e-6,
            last_tstep=None, other=None)

def compute_ocean(simulator_oce: Ocean1dStratified,
        atm_state: StateAtm,
        numer_set: NumericalSetting, **kwargs) -> StateOce:
    """
        Integrator in time of the ocean
    """
    T0 = 281. # Reference temperature
    N = int(numer_set.T/simulator_oce.dt) # Number of time steps
    theta_0 = T0 - simulator_oce.N0**2 * \
            np.abs(simulator_oce.z_half[:-1]) \
            / simulator_oce.alpha / 9.81 # Initial temperature
    u_0 = np.zeros(simulator_oce.M)
    phi_0 = np.zeros(simulator_oce.M+1)
    dz_theta_0 = np.ones(simulator_oce.M+1) * simulator_oce.N0**2 \
            / simulator_oce.alpha / 9.81

    Q_sw = projection(np.asarray(numer_set.Q_sw), N)
    Q_lw = projection(np.asarray(numer_set.Q_lw), N)
    delta_sl = numer_set.delta_sl_o
    sf_scheme = numer_set.sf_scheme_o
    wind_10m = projection(atm_state.u_delta, N)
    temp_10m = projection(atm_state.t_delta, N)
    u_star = projection(atm_state.u_star, N)
    t_star = projection(atm_state.t_star, N)

    if sf_scheme in {"FV free", "FV2"}:
        u_i, phi_i, theta_i, dz_theta_i, u_delta, t_delta = \
                simulator_oce.initialization(\
                np.zeros(simulator_oce.M)+0j, # u_0
                np.copy(theta_0), # theta_0
                delta_sl, wind_10m[0], temp_10m[0],
                u_star[0], t_star[0],
                Q_sw[0], Q_lw[0],
                10., sf_scheme)
    else:
        u_i, phi_i, theta_i, dz_theta_i, u_delta, t_delta = \
                u_0, phi_0, theta_0, dz_theta_0, 0., T0

    if sf_scheme[:2] == "FV":
        ret = simulator_oce.FV(u_t0=u_i, phi_t0=phi_i,
                theta_t0=theta_i, dz_theta_t0=dz_theta_i, Q_sw=Q_sw,
                Q_lw=Q_lw, delta_sl_a=numer_set.delta_sl_a,
                u_star=u_star, t_star=t_star,
                u_delta=u_delta, t_delta=t_delta, delta_sl=delta_sl,
                heatloss=None, wind_10m=wind_10m,
                temp_10m=temp_10m, sf_scheme=sf_scheme,
                **kwargs)

    elif sf_scheme[:2] == "FD":
        ret = simulator_oce.FD(u_t0=u_0, theta_t0=theta_0,
                delta_sl_a=numer_set.delta_sl_a,
                u_star=u_star, t_star=t_star,
                Q_sw=Q_sw, Q_lw=Q_lw, wind_10m=wind_10m,
                temp_10m=temp_10m,
                heatloss=None, sf_scheme=sf_scheme,
                **kwargs)
    else:
        raise NotImplementedError("Cannot infer discretization " + \
                "from surface flux scheme name " + sf_scheme)
    last_tstep = {key: ret[key] for key in ("u", "theta", "tke")}

    if sf_scheme[:2] == "FV":
        last_tstep["dz_theta"] = ret["dz_theta"]
        last_tstep["phi"] = ret["phi"]

    return StateOce(u_delta=np.array(ret["u_delta"]),
            t_delta=np.array(ret["t_delta"]),
            last_tstep=last_tstep, other=ret)

def compute_atmosphere(simulator_atm: Atm1dStratified,
        oce_state: StateOce,
        numer_set: NumericalSetting, **kwargs) -> StateAtm:
    """
        Integrator in time of the atmosphere
    """
    N = int(numer_set.T/simulator_atm.dt) # Number of time steps
    M = simulator_atm.M # Number of grid points
    u_0 = INIT_U_ATM*np.ones(M)
    phi_0 = np.zeros(M+1) + 0j
    theta_0 = INIT_THETA_ATM*np.ones(M)
    dz_theta_0 = np.zeros(M+1)
    forcing = 1j*simulator_atm.f*simulator_atm.u_g*np.ones((N+1, M))

    delta_sl = numer_set.delta_sl_a
    sf_scheme = numer_set.sf_scheme_a
    uo_delta = projection(oce_state.u_delta, N)
    to_delta = projection(oce_state.t_delta, N)
    Q_sw = projection(np.asarray(numer_set.Q_sw), N)
    Q_lw = projection(np.asarray(numer_set.Q_lw), N)
    z_constant = 15

    if sf_scheme in {"FV free", "FV2"}:
        u_i, phi_i, theta_i, dz_theta_i, u_delta, t_delta = \
                simulator_atm.initialization(\
                u_0, phi_0, theta_0, dz_theta_0, delta_sl,
                uo_delta[0], to_delta[0], Q_sw[0], Q_lw[0],
                z_constant, numer_set.delta_sl_o)
    else:
        u_i, phi_i, theta_i, dz_theta_i, u_delta, t_delta = \
                u_0, phi_0, theta_0, dz_theta_0, u_0[0], theta_0[0]

    if sf_scheme[:2] == "FV":
        ret = simulator_atm.FV(u_t0=u_i, phi_t0=phi_i,
                        SST=to_delta, sf_scheme=sf_scheme,
                        theta_t0=theta_i, dz_theta_t0=dz_theta_i,
                        u_o=uo_delta, u_delta=u_delta,
                        t_delta=t_delta, Q_sw=Q_sw, Q_lw=Q_lw,
                        delta_sl_o=numer_set.delta_sl_o,
                        forcing=forcing, delta_sl=delta_sl,
                        **kwargs)
    elif sf_scheme[:2] == "FD":
        ret = simulator_atm.FD(u_t0=u_i, theta_t0=theta_i,
                delta_sl_o=numer_set.delta_sl_o,
                Q_sw=Q_sw, Q_lw=Q_lw, u_o=uo_delta,
                SST=to_delta, forcing=forcing,
                sf_scheme=sf_scheme,
                **kwargs)
    else:
        raise NotImplementedError("Cannot infer discretization " + \
                "from surface flux scheme name " + sf_scheme)
    last_tstep = {key: ret[key] for key in ("u", "theta", "tke")}

    if sf_scheme[:2] == "FV":
        last_tstep["dz_theta"] = ret["dz_theta"]
        last_tstep["phi"] = ret["phi"]
    return StateAtm(u_delta=np.array(ret["u_delta"]),
            t_delta=np.array(ret["t_delta"]),
            u_star=np.array(ret["all_u_star"]),
            t_star=np.array(ret["all_t_star"]),
            last_tstep=last_tstep, other=ret)


def projection(array: np.ndarray, N: int)-> np.ndarray:
    """
        projects an array of size array.shape[0] onto
        an other array of shape N+1.
    """
    return interpolate.interp1d(np.linspace(0, 1, array.shape[0]),
            array)(np.linspace(0, 1, N+1))
